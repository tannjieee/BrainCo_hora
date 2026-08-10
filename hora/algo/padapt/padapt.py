"""ProprioAdapt Stage2 student distillation.

Freezes all Stage1 weights except adapt_tconv. Online training combines latent
  MSE with teacher/student action MSE. Rollout starts under the privileged
  teacher and linearly hands control to the tactile-history student.

Checkpoint: .ckpt extension. Warm-start from Stage1 .pth via strict=False.
  Full resume supports optimizer/agent_steps/best_rewards/rms/sa_ms.

Gotcha — sa_mean_std stays in train() mode (accumulating proprio_hist statistics
  online), while running_mean_std stays in eval() mode (Stage1 stats frozen).
"""
import os
import time
import math
import torch
import torch.nn.functional as F
from termcolor import cprint

from hora.utils.misc import AverageScalarMeter, tprint
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from tensorboardX import SummaryWriter


class ProprioAdapt(object):
    def __init__(self, env, output_dir, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        self.stage2_config = full_config.train.get('stage2', {})
        self.source_checkpoint = str(full_config.train.load_path)
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        # ---- Priv Info ----
        self.priv_info = self.ppo_config['priv_info']
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.proprio_adapt = self.ppo_config['proprio_adapt']
        self.proprio_hist_dim = self.env.prop_hist_len
        self.obs_per_step = self.obs_shape[0] // 3
        if not bool(getattr(self.env.cfg, 'enable_tactile', False)):
            raise RuntimeError('Tactile Stage2 requires env.enable_tactile=True')
        if not bool(getattr(self.env.cfg, 'enable_contact_in_obs', False)):
            raise RuntimeError('Tactile Stage2 requires contact forces in actor observations')
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': self.proprio_adapt,
            'priv_info_dim': self.priv_info_dim,
            'obs_per_step': self.obs_per_step,
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((self.proprio_hist_dim, self.obs_per_step)).to(self.device)
        self.sa_mean_std.train()
        # ---- Output Dir ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'stage2_nn')
        self.tb_dir = os.path.join(self.output_dir, 'stage2_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        writer = SummaryWriter(self.tb_dir)
        self.writer = writer
        self.direct_info = {}
        # ---- Misc ----
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.best_rewards = -10000
        self.best_latent_loss = float('inf')
        self.completed_episodes = 0
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.save_interval_steps = int(self.stage2_config.get('save_interval_steps', 25_000_000))
        self.next_save_step = self.save_interval_steps
        self.log_interval = max(1, int(self.stage2_config.get('log_interval', 20)))
        self.best_after_steps = int(self.stage2_config.get('best_after_steps', 50_000_000))
        self.best_min_episodes = int(self.stage2_config.get('best_min_episodes', 4096))
        self.teacher_warmup_steps = int(self.stage2_config.get('teacher_warmup_steps', 10_000_000))
        self.teacher_mix_steps = int(self.stage2_config.get('teacher_mix_steps', 40_000_000))
        self.latent_loss_coef = float(self.stage2_config.get('latent_loss_coef', 1.0))
        self.action_loss_coef = float(self.stage2_config.get('action_loss_coef', 0.25))
        self.grad_norm = float(self.stage2_config.get('grad_norm', 1.0))
        if self.save_interval_steps <= 0:
            raise ValueError('stage2.save_interval_steps must be positive')
        if not 1 <= self.best_min_episodes <= self.mean_eps_reward.window_size:
            raise ValueError('stage2.best_min_episodes must fit the reward-meter window')
        if min(self.teacher_warmup_steps, self.teacher_mix_steps, self.best_after_steps) < 0:
            raise ValueError('Stage2 schedule steps must be non-negative')
        if min(self.latent_loss_coef, self.action_loss_coef) < 0:
            raise ValueError('Stage2 loss coefficients must be non-negative')
        if self.grad_norm <= 0:
            raise ValueError('stage2.grad_norm must be positive')
        # ---- Optim ----
        for p in self.model.parameters():
            p.requires_grad = False
        adapt_params = list(self.model.adapt_tconv.parameters())
        if not adapt_params:
            raise RuntimeError('Stage2 has no adapt_tconv parameters to optimize')
        for p in adapt_params:
            p.requires_grad = True
        learning_rate = float(self.stage2_config.get('learning_rate', 3e-4))
        self.optim = torch.optim.Adam(adapt_params, lr=learning_rate)
        # ---- Training Misc
        self.internal_counter = 0
        batch_size = self.num_actors
        self.step_reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()

    def test(self, max_steps: int = 0, real_time: bool = False):
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
                'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            reward_sum += float(r.mean().item())
            height_reset_count += self._info_scalar(info, 'height_reset_lower') * self.num_actors
            height_reset_count += self._info_scalar(info, 'height_reset_upper') * self.num_actors
            timeout_count += self._info_scalar(info, 'time_out') * self.num_actors
            tilt_sum += self._info_scalar(info, 'cylinder_tilt_deg')
            step += 1
            sleep_time = step_dt - (time.time() - step_start)
            if real_time and sleep_time > 0:
                time.sleep(sleep_time)
        if max_steps > 0:
            transitions = max_steps * self.num_actors
            print(
                "[FULL-GRAVITY EVAL]\n"
                f"  policy steps  : {max_steps}\n"
                f"  transitions   : {transitions}\n"
                f"  mean reward   : {reward_sum / max_steps:.6f}\n"
                f"  mean tilt     : {tilt_sum / max_steps:.3f} deg\n"
                f"  height resets : {height_reset_count:.0f} "
                f"({height_reset_count / transitions:.6%}/step)\n"
                f"  timeouts      : {timeout_count:.0f}",
                flush=True,
            )

    @staticmethod
    def _info_scalar(info: dict, key: str) -> float:
        value = info.get(key, 0.0)
        if isinstance(value, torch.Tensor):
            return float(value.float().mean().item())
        return float(value)

    def _teacher_ratio(self) -> float:
        """Linearly hand rollout control from the privileged teacher to the student."""
        if self.agent_steps < self.teacher_warmup_steps:
            return 1.0
        if self.teacher_mix_steps <= 0:
            return 0.0
        progress = (self.agent_steps - self.teacher_warmup_steps) / self.teacher_mix_steps
        return max(0.0, min(1.0, 1.0 - progress))

    def _distillation_step(self, obs_dict):
        """Optimize the adapter and return detached teacher/student actions and metrics."""
        obs = self.running_mean_std(obs_dict['obs']).detach()
        proprio_hist = self.sa_mean_std(obs_dict['proprio_hist'].detach())

        student_latent = torch.tanh(self.model.adapt_tconv(proprio_hist))
        with torch.no_grad():
            teacher_latent = torch.tanh(self.model.env_mlp(obs_dict['priv_info']))
            teacher_features = self.model.actor_mlp(torch.cat([obs, teacher_latent], dim=-1))
            teacher_mu = self.model.mu(teacher_features)

        # The actor is frozen, but retaining the student-latent input graph lets
        # action loss weight latent errors by their effect on the policy output.
        student_features = self.model.actor_mlp(torch.cat([obs, student_latent], dim=-1))
        student_mu = self.model.mu(student_features)
        latent_loss = F.mse_loss(student_latent, teacher_latent)
        action_loss = F.mse_loss(student_mu, teacher_mu)
        loss = self.latent_loss_coef * latent_loss + self.action_loss_coef * action_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f'Non-finite Stage2 loss: latent={latent_loss.item()}, action={action_loss.item()}'
            )

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.adapt_tconv.parameters(), self.grad_norm)
        self.optim.step()

        with torch.no_grad():
            latent_cosine = F.cosine_similarity(student_latent, teacher_latent, dim=-1).mean()
        metrics = {
            'stage2/loss_total': loss.detach(),
            'stage2/loss_latent': latent_loss.detach(),
            'stage2/loss_action': action_loss.detach(),
            'stage2/latent_cosine': latent_cosine.detach(),
            'stage2/grad_norm': torch.as_tensor(grad_norm).detach(),
        }
        return student_mu.detach(), teacher_mu.detach(), metrics

    def train(self):
        _t = time.time()
        _last_t = time.time()
        start_agent_steps = self.agent_steps
        remaining_steps = max(0, self.max_agent_steps - start_agent_steps)
        total_iters = max(1, math.ceil(remaining_steps / self.batch_size))
        iter_num = 0

        obs_dict = self.env.reset()
        while self.agent_steps < self.max_agent_steps:
            iter_num += 1
            iter_start_t = time.time()

            learn_start_t = time.time()
            student_mu, teacher_mu, distill_metrics = self._distillation_step(obs_dict)
            learn_t = time.time() - learn_start_t

            teacher_ratio = self._teacher_ratio()
            mu = teacher_ratio * teacher_mu + (1.0 - teacher_ratio) * student_mu
            mu = torch.clamp(mu, -1.0, 1.0)
            collect_start_t = time.time()
            obs_dict, r, done, info = self.env.step(mu)
            collect_t = time.time() - collect_start_t
            self.agent_steps += self.batch_size

            # ---- statistics
            self.step_reward += r
            self.step_length += 1
            done_mask = done.bool().view(-1)
            self.mean_eps_reward.update(self.step_reward[done_mask])
            if self.completed_episodes < self.best_min_episodes:
                self.completed_episodes += int(done_mask.sum().item())

            not_dones = (~done_mask).float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            should_log = iter_num % self.log_interval == 0
            if should_log:
                for k, v in info.items():
                    if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1):
                        self.direct_info[k] = float(v)
                if 'time_outs' in info:
                    self.direct_info['stage2/timeout_rate'] = float(
                        info['time_outs'].float().mean().detach().cpu().item()
                    )
                self.direct_info.update({k: float(v.item()) for k, v in distill_metrics.items()})
                self.direct_info['stage2/teacher_ratio'] = teacher_ratio
                self.direct_info['stage2/completed_episodes'] = float(self.completed_episodes)
                self.log_tensorboard()

            if self.agent_steps >= self.next_save_step:
                step_m = int(self.agent_steps // 1e6)
                self.save(os.path.join(self.nn_dir, f'{step_m:04d}M'))
                self.save(os.path.join(self.nn_dir, 'model_last'))
                self.next_save_step += self.save_interval_steps

            mean_rewards = self.mean_eps_reward.get_mean()
            best_ready = (
                self.agent_steps >= self.best_after_steps
                and self.completed_episodes >= self.best_min_episodes
                and len(self.mean_eps_reward) >= self.best_min_episodes
            )
            if should_log and best_ready and teacher_ratio <= 0.0 and mean_rewards > self.best_rewards:
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, 'model_best'))

            if should_log and best_ready and teacher_ratio <= 0.0:
                latent_loss_value = float(distill_metrics['stage2/loss_latent'].item())
                if latent_loss_value < self.best_latent_loss:
                    self.best_latent_loss = latent_loss_value
                    self.save(os.path.join(self.nn_dir, 'model_best_latent'))

            all_fps = (self.agent_steps - start_agent_steps) / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            if should_log:
                info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                              f'Last FPS: {last_fps:.1f} | ' \
                              f'Current Best: {self.best_rewards:.2f}'
                tprint(info_string)
                print("", flush=True)
                self._print_epoch_log(
                    iter_num=iter_num,
                    total_iters=total_iters,
                    collect_t=collect_t,
                    learn_t=learn_t,
                    iter_t=time.time() - iter_start_t,
                    elapsed=time.time() - _t,
                    mean_rewards=mean_rewards,
                )

        self.save(os.path.join(self.nn_dir, 'model_final'))
        self.save(os.path.join(self.nn_dir, 'model_last'))
        self.writer.flush()
        self.writer.close()
        print(f'[INFO] Stage2 max steps achieved: {self.agent_steps}', flush=True)

    def log_tensorboard(self):
        self.writer.add_scalar('episode_rewards/step', self.mean_eps_reward.get_mean(), self.agent_steps)
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, self.agent_steps)

    @staticmethod
    def _validate_stage2_tactile_abi(checkpoint, fn) -> None:
        if checkpoint.get('tactile_required') is False:
            raise RuntimeError(
                f'Stage2 checkpoint uses the obsolete no-tactile ABI: {fn}'
            )
        if 'tactile_required' not in checkpoint:
            print(
                f'[WARN] Stage2 checkpoint has no tactile ABI metadata: {fn}',
                flush=True,
            )

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        is_stage2_ckpt = str(fn).endswith(".ckpt") or ("stage2_nn" in str(fn))
        if is_stage2_ckpt:
            self._validate_stage2_tactile_abi(checkpoint, fn)
            required_keys = ["model", "optimizer", "agent_steps", "best_rewards"]
            missing = [k for k in required_keys if k not in checkpoint]
            if missing:
                raise RuntimeError(
                    f"Stage2 resume failed: missing keys {missing} in checkpoint: {fn}"
                )

            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.optim.load_state_dict(checkpoint["optimizer"])
            self.agent_steps = int(checkpoint["agent_steps"])
            self.best_rewards = float(checkpoint["best_rewards"])
            self.best_latent_loss = float(checkpoint.get("best_latent_loss", float('inf')))
            self.completed_episodes = int(checkpoint.get("completed_episodes", 0))
            self.source_checkpoint = str(checkpoint.get("source_checkpoint", self.source_checkpoint))
            self.next_save_step = ((self.agent_steps // self.save_interval_steps) + 1) * self.save_interval_steps
            if "running_mean_std" in checkpoint:
                self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
            if "sa_mean_std" in checkpoint:
                self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
            print(
                f"[INFO] Resumed Stage2: agent_steps={self.agent_steps}, "
                f"best_rewards={self.best_rewards:.4f}",
                flush=True,
            )
            return

        cprint('Warm-starting Stage2 adapter from Stage1 checkpoint', 'yellow', attrs=['bold'])
        incompatible = self.model.load_state_dict(checkpoint['model'], strict=False)
        expected_missing = {f'adapt_tconv.{name}' for name in self.model.adapt_tconv.state_dict()}
        actual_missing = set(incompatible.missing_keys)
        if actual_missing != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                'Unexpected Stage1→Stage2 state mismatch: '
                f'missing={sorted(actual_missing)}, unexpected={incompatible.unexpected_keys}'
            )
        if 'running_mean_std' in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if 'sa_mean_std' in checkpoint:
            self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])
        if float(checkpoint.get('gravity_magnitude', 0.0)) < 9.79:
            print(
                '[WARN] Stage1 source checkpoint was not saved at full gravity; '
                'validate it at 9.81 m/s^2 before trusting Stage2.',
                flush=True,
            )
        if float(checkpoint.get('best_rewards', -10000.0)) <= -9999.0:
            print(
                '[WARN] Stage1 source checkpoint has no validated full-gravity best reward.',
                flush=True,
            )
        print("[INFO] Warm-start Stage2 from non-resume checkpoint.", flush=True)

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self._validate_stage2_tactile_abi(checkpoint, fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])
        self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'agent_steps': int(self.agent_steps),
            'best_rewards': float(self.best_rewards),
            'best_latent_loss': float(self.best_latent_loss),
            'completed_episodes': int(self.completed_episodes),
            'source_checkpoint': self.source_checkpoint,
            'obs_shape': tuple(self.obs_shape),
            'proprio_hist_len': int(self.proprio_hist_dim),
            'priv_info_dim': int(self.priv_info_dim),
            'tactile_required': bool(getattr(self.env.cfg, 'enable_contact_in_obs', True)),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights['sa_mean_std'] = self.sa_mean_std.state_dict()
        torch.save(weights, f'{name}.ckpt')

    def _print_epoch_log(self, iter_num, total_iters, collect_t, learn_t, iter_t, elapsed, mean_rewards):
        width = 100
        pad = 30
        fps = int(self.batch_size / max(1e-6, collect_t + learn_t))
        eta_sec = max(0.0, (total_iters - iter_num) * (elapsed / max(1, iter_num)))

        # Collect numeric extras for display
        rew_items = []
        for k in sorted(self.direct_info.keys()):
            v = self.direct_info[k]
            if isinstance(v, (int, float)):
                rew_items.append((k, float(v)))

        header = f" Learning iteration {iter_num}/{total_iters} "
        lines = [
            "#" * width,
            header.center(width, " "),
            "",
            f"{'Computation:':>{pad}} {fps} steps/s (collection: {collect_t:.3f}s, learning: {learn_t:.3f}s)",
            f"{'Mean reward:':>{pad}} {mean_rewards:.4f}",
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
