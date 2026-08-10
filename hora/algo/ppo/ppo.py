"""PPO trainer for Stage1 teacher policy.

Training loop: collect horizon_length steps × num_envs → GAE returns →
  PPO clipped loss with KL-adaptive learning rate.

Value bootstrap: when episode truncates (timeout, not termination), the last
  value estimate bootstraps the return to avoid penalizing unfinished episodes.

Gotcha — minibatch_size must divide batch_size (num_envs × horizon) exactly.
  With the current 16-step horizon, train.py permits 2048, 4096, ... envs.

Gotcha — reward_scale: total reward × 0.01 before GAE. Env extras (scalar means)
  are logged to TensorBoard via extra_info dict.
"""
import os
import time
import math
import torch

from hora.algo.ppo.experience import ExperienceBuffer
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd

from hora.utils.misc import AverageScalarMeter, tprint

from tensorboardX import SummaryWriter


class PPO(object):
    def __init__(self, env, output_dif, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        # ---- Priv Info ----
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.priv_info = self.ppo_config['priv_info']
        self.proprio_adapt = self.ppo_config['proprio_adapt']
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': self.proprio_adapt,
            'priv_info_dim': self.priv_info_dim,
            'obs_per_step': self.obs_shape[0] // 3,
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Output Dir ----
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
        self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.last_lr, weight_decay=self.weight_decay)
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        self.reward_scale = float(self.ppo_config.get('reward_scale', 0.01))
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.kl_threshold = self.ppo_config['kl_threshold']
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        self.full_gravity_magnitude = float(self.ppo_config.get('full_gravity_magnitude', 9.81))
        self.full_gravity_tolerance = float(self.ppo_config.get('full_gravity_tolerance', 0.02))
        self.full_gravity_max_reset_rate = float(self.ppo_config.get('full_gravity_max_reset_rate', 0.003))
        self.full_gravity_eval_epochs = int(self.ppo_config.get('full_gravity_eval_epochs', 25))
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(100)
        self.episode_raw_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.obs = None
        self.epoch_num = 0
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size, self.obs_shape[0],
            self.actions_num, self.priv_info_dim, self.device,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_raw_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000
        self.best_curriculum_rewards = -10000
        self.full_gravity_epochs = 0
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        def _mean_or_none(items):
            if len(items) == 0:
                return None
            return torch.mean(torch.stack(items)).item()

        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)

        actor_loss = _mean_or_none(a_losses)
        bounds_loss = _mean_or_none(b_losses)
        critic_loss = _mean_or_none(c_losses)
        entropy = _mean_or_none(entropies)
        if actor_loss is not None:
            self.writer.add_scalar('losses/actor_loss', actor_loss, self.agent_steps)
        if bounds_loss is not None:
            self.writer.add_scalar('losses/bounds_loss', bounds_loss, self.agent_steps)
        if critic_loss is not None:
            self.writer.add_scalar('losses/critic_loss', critic_loss, self.agent_steps)
        if entropy is not None:
            self.writer.add_scalar('losses/entropy', entropy, self.agent_steps)

        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        kl_mean = _mean_or_none(kls)
        if kl_mean is not None:
            self.writer.add_scalar('info/kl', kl_mean, self.agent_steps)
        for k, v in self.extra_info.items():
            self.writer.add_scalar(f'{k}', v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        input_dict = {
            'obs': processed_obs,
            'priv_info': obs_dict['priv_info'],
        }
        res_dict = self.model.act(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        if self.agent_steps == 0:
            self.agent_steps = self.batch_size
        total_iters = max(1, math.ceil(self.max_agent_steps / self.batch_size))

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            iter_start_t = time.time()
            a_losses, c_losses, b_losses, entropies, kls, collect_t, learn_t = self.train_epoch()
            self.storage.data_dict = None

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            self.write_stats(a_losses, c_losses, b_losses, entropies, kls)

            mean_rewards = self.episode_rewards.get_mean()
            mean_raw_rewards = self.episode_raw_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_rewards_raw/step', mean_raw_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            gravity_magnitude = float(self.extra_info.get('gravity_magnitude', 0.0))
            gravity_reset_rate = float(self.extra_info.get('gravity_reset_rate_window', 1.0))
            at_full_gravity = (
                gravity_magnitude >= self.full_gravity_magnitude - self.full_gravity_tolerance
                and gravity_reset_rate <= self.full_gravity_max_reset_rate
            )
            self.full_gravity_epochs = self.full_gravity_epochs + 1 if at_full_gravity else 0
            full_gravity_evaluated = self.full_gravity_epochs >= self.full_gravity_eval_epochs
            self.writer.add_scalar('gravity/full_eval_epochs', self.full_gravity_epochs, self.agent_steps)
            self.writer.add_scalar('gravity/full_eval_ready', float(full_gravity_evaluated), self.agent_steps)

            checkpoint_name = (
                f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M_'
                f'g_{gravity_magnitude:.2f}_reward_{mean_rewards:.2f}'
            )
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | Full-g Best: {self.best_rewards:.2f} | ' \
                          f'Curriculum Best: {self.best_curriculum_rewards:.2f}'
            tprint(info_string)
            print("", flush=True)
            self._print_epoch_log(
                total_iters=total_iters,
                collect_t=collect_t,
                learn_t=learn_t,
                iter_t=time.time() - iter_start_t,
                elapsed=time.time() - _t,
                mean_rewards=mean_rewards,
                mean_lengths=mean_lengths,
            )
            # Keep a diagnostic best across the curriculum, but never expose it
            # as best.pth for Stage2.  best.pth is reserved for policies that
            # have survived a fixed full-gravity evaluation period.
            if mean_rewards > self.best_curriculum_rewards and self.epoch_num >= self.save_best_after:
                self.best_curriculum_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, 'best_curriculum'))

            if full_gravity_evaluated and mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                print(
                    f'save full-gravity best reward: {mean_rewards:.2f} '
                    f'(g={gravity_magnitude:.2f}, reset_rate={gravity_reset_rate:.5f})',
                    flush=True,
                )
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, 'best'))
                self.save(os.path.join(self.nn_dir, 'best_full_gravity'))

            if self.save_freq > 0 and self.epoch_num % self.save_freq == 0:
                self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, 'last'))

        print('max steps achieved')

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'agent_steps': int(self.agent_steps),
            'epoch_num': int(self.epoch_num),
            'best_rewards': float(self.best_rewards),
            'best_curriculum_rewards': float(self.best_curriculum_rewards),
            'full_gravity_epochs': int(self.full_gravity_epochs),
            'gravity_magnitude': float(getattr(self.env, '_gravity_magnitude', 0.0)),
            'gravity_reset_rate_window': float(getattr(self.env, '_gravity_window_reset_rate', 1.0)),
            'priv_info_dim': int(self.priv_info_dim),
            'last_lr': float(self.last_lr),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        required_keys = [
            'model',
            'running_mean_std',
            'value_mean_std',
            'optimizer',
            'agent_steps',
            'epoch_num',
            'best_rewards',
            'last_lr',
        ]
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            raise RuntimeError(
                f"Strict Stage1 resume failed: missing keys {missing} in checkpoint: {fn}"
            )
        checkpoint_priv_dim = checkpoint.get('priv_info_dim')
        if checkpoint_priv_dim is not None and int(checkpoint_priv_dim) != self.priv_info_dim:
            raise RuntimeError(
                f"Stage1 checkpoint privileged-observation mismatch: checkpoint={checkpoint_priv_dim}, "
                f"current={self.priv_info_dim}. Retrain Stage1 after changing privileged observations."
            )

        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.value_mean_std.load_state_dict(checkpoint['value_mean_std'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.agent_steps = int(checkpoint['agent_steps'])
        self.epoch_num = int(checkpoint['epoch_num'])
        self.best_rewards = float(checkpoint['best_rewards'])
        self.best_curriculum_rewards = float(checkpoint.get('best_curriculum_rewards', self.best_rewards))
        self.full_gravity_epochs = int(checkpoint.get('full_gravity_epochs', 0))
        self.last_lr = float(checkpoint['last_lr'])
        if 'gravity_magnitude' in checkpoint and hasattr(self.env, 'set_gravity_magnitude'):
            self.env.set_gravity_magnitude(float(checkpoint['gravity_magnitude']))
        if 'gravity_reset_rate_window' in checkpoint and hasattr(self.env, '_gravity_window_reset_rate'):
            self.env._gravity_window_reset_rate = float(checkpoint['gravity_reset_rate_window'])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.last_lr
        print(
            f"[INFO] Restored train state: agent_steps={self.agent_steps}, "
            f"epoch_num={self.epoch_num}, best_rewards={self.best_rewards:.4f}, lr={self.last_lr:.6g}",
            flush=True,
        )

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def test(self, max_steps: int = 0, real_time: bool = False):
        """Evaluate a checkpoint at the gravity configured by train.py.

        ``max_steps=0`` retains interactive/infinite play.  A positive value
        produces a finite, reproducible summary suitable for checkpoint
        comparison; train.py configures test environments at 9.81 m/s².
        """
        self.set_eval()
        obs_dict = self.env.reset()
        step = 0
        reward_sum = 0.0
        height_reset_count = 0.0
        timeout_count = 0.0
        tilt_sum = 0.0
        step_dt = float(getattr(self.env, 'step_dt', 0.0))
        while max_steps <= 0 or step < max_steps:
            step_start = time.time()
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'priv_info': obs_dict['priv_info'],
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            step += 1
            reward_sum += float(r.mean().item())
            height_reset_count += self._info_scalar(info, 'height_reset_lower') * self.num_actors
            height_reset_count += self._info_scalar(info, 'height_reset_upper') * self.num_actors
            timeout_count += self._info_scalar(info, 'time_out') * self.num_actors
            tilt_sum += self._info_scalar(info, 'cylinder_tilt_deg')
            sleep_time = step_dt - (time.time() - step_start)
            if real_time and sleep_time > 0:
                time.sleep(sleep_time)

        if max_steps > 0:
            transitions = max_steps * self.num_actors
            print(
                "[FULL-GRAVITY EVAL]\n"
                f"  gravity       : {float(getattr(self.env, '_gravity_magnitude', 9.81)):.3f} m/s^2\n"
                f"  policy steps  : {max_steps}\n"
                f"  transitions   : {transitions}\n"
                f"  mean reward   : {reward_sum / max_steps:.6f}\n"
                f"  mean tilt     : {tilt_sum / max_steps:.3f} deg\n"
                f"  height resets : {height_reset_count:.0f} ({height_reset_count / transitions:.6%}/step)\n"
                f"  timeouts      : {timeout_count:.0f}",
                flush=True,
            )

    @staticmethod
    def _info_scalar(info: dict, key: str) -> float:
        value = info.get(key, 0.0)
        if isinstance(value, torch.Tensor):
            return float(value.float().mean().item())
        return float(value)

    def train_epoch(self):
        _t = time.time()
        self.set_eval()
        self.play_steps()
        collect_t = time.time() - _t
        self.data_collect_time += collect_t
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []

        for mini_epoch in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs, priv_info = self.storage[i]

                obs = self.running_mean_std(obs)
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'priv_info': priv_info,
                }
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    b_loss = action_bounds_loss(mu, soft_bound)
                else:
                    b_loss = torch.zeros_like(a_loss)
                a_loss, c_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

                self.optimizer.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            if len(ep_kls) == 0:
                av_kls = torch.tensor(0.0, device=self.device)
            else:
                av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr
            kls.append(av_kls)

        learn_t = time.time() - _t
        self.rl_train_time += learn_t
        return a_losses, c_losses, b_losses, entropies, kls, collect_t, learn_t

    def _print_epoch_log(self, total_iters, collect_t, learn_t, iter_t, elapsed, mean_rewards, mean_lengths):
        width = 100
        pad = 30
        fps = int(self.batch_size / max(1e-6, collect_t + learn_t))
        eta_sec = max(0.0, (total_iters - self.epoch_num) * (elapsed / max(1, self.epoch_num)))

        rew_items = []
        for k in sorted(self.extra_info.keys()):
            v = self.extra_info[k]
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (int, float)):
                rew_items.append((k, float(v)))

        header = f" Learning iteration {self.epoch_num}/{total_iters} "
        lines = [
            "#" * width,
            header.center(width, " "),
            "",
            f"{'Computation:':>{pad}} {fps} steps/s (collection: {collect_t:.3f}s, learning: {learn_t:.3f}s)",
            f"{'Mean reward:':>{pad}} {mean_rewards:.4f}",
            f"{'Mean episode length:':>{pad}} {mean_lengths:.4f}",
        ]
        for k, v in rew_items:
            lines.append(f"{k + ':':>{pad}} {v:.6f}")
        lines.extend([
            "-" * width,
            f"{'Total timesteps:':>{pad}} {self.agent_steps}",
            f"{'Iteration time:':>{pad}} {iter_t:.2f}s",
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}",
            f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta_sec))}",
        ])
        print("\n".join(lines))

    def play_steps(self):
        extra_sums = {}
        extra_counts = {}
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('priv_info', n, self.obs['priv_info'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = self.reward_scale * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += shaped_rewards
            self.current_raw_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_raw_rewards.update(self.current_raw_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            assert isinstance(infos, dict), 'Info Should be a Dict'
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    if isinstance(k, str) and k.startswith("rew/"):
                        v = float(v) * self.reward_scale
                    else:
                        v = float(v)
                    extra_sums[k] = extra_sums.get(k, 0.0) + v
                    extra_counts[k] = extra_counts.get(k, 0) + 1

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_raw_rewards = self.current_raw_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        self.extra_info = {
            k: extra_sums[k] / extra_counts[k]
            for k in extra_sums
        }

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps += self.batch_size
        self.storage.computer_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)
    return kl.mean()


def action_bounds_loss(mu: torch.Tensor, soft_bound: float = 1.1) -> torch.Tensor:
    """Penalize only policy means outside ``[-soft_bound, soft_bound]``.

    The old implementation used ``clamp_max`` and therefore penalized values
    *inside* the interval while pushing every mean toward +soft_bound.
    """
    return torch.relu(torch.abs(mu) - soft_bound).square().sum(dim=-1)


class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
