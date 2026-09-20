#!/usr/bin/env python3
"""Run two bounded duck PPO experiments sequentially on one idle GPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--suffix', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    budget = 20971520
    kit_args = '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8'
    env = dict(os.environ, PYTHON_BIN=args.python, DUCK_ENVS='2048', DUCK_SPEED='0.5',
               DUCK_CACHE='revo3_right_grasp_rubber_duck_gait_v2.npy',
               PYTHONUNBUFFERED='1', OMNI_KIT_ACCEPT_EULA='YES', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PXR_WORK_THREAD_LIMIT='8')
    # Prefer this interpreter even for helper commands inside shell scripts.
    env['PATH'] = str(Path(args.python).parent) + os.pathsep + env['PATH']
    env.pop('ISAACLAB_PATH', None)
    source_paths = ['train.py', 'hora/algo/ppo/ppo.py', 'scripts/duck_gait.sh',
                    'scripts/duck_gait_compare.sh', 'scripts/run_duck_comparison.py',
                    'configs/train/Revo3HandHora.yaml', 'hora/tasks/isaaclab/revo3_hand_hora_env.py',
                    'hora/tasks/isaaclab/revo3_hand_hora_env_cfg.py', 'hora/utils/finger_gait.py']
    source_hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}
    jobs = []
    for profile, horizon, gamma, seconds in [('baseline', 16, .99, 20), ('long', 64, .995, 40)]:
        directory = run / profile
        directory.mkdir()
        output_name = f'run_rubber_duck_gait_{profile}_s42_{args.suffix}'
        output = root / 'outputs/revo3_right' / output_name
        if output.exists():
            raise FileExistsError(output)
        train_cmd = ['bash', 'scripts/duck_gait_compare.sh', profile,
                     '--device', f'cuda:{args.gpu}', '--kit_args', kit_args]
        checkpoint = output / 'stage1_nn/last.pth'
        eval_cmd = ['bash', 'scripts/duck_gait_compare.sh', 'eval', str(checkpoint),
                    '--device', f'cuda:{args.gpu}', '--kit_args', kit_args]
        manifest = dict(profile=profile, gpu=args.gpu, output=str(output), seed=42,
                        num_envs=2048, max_agent_steps=budget, horizon_length=horizon,
                        gamma=gamma, episode_length_s=seconds, rotation_speed=.5,
                        eval_seed=12345, eval_envs=128, eval_seconds=60,
                        train_command=train_cmd, eval_command=eval_cmd, source_sha256=source_hashes)
        (directory / 'launch.json').write_text(json.dumps(manifest, indent=2) + '\n')
        jobs.append((directory, output_name, checkpoint, train_cmd, eval_cmd))

    for directory, output_name, checkpoint, train_cmd, eval_cmd in jobs:
        for path, expected in source_hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Source changed while queued: {path}')
        row = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}',
              '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, utilization = map(int, row.strip().split(','))
        if memory > 1024 or utilization > 5:
            raise RuntimeError(f'GPU {args.gpu} is occupied; queue stopped: {row.strip()}')
        env['DUCK_RUN'] = output_name
        print(datetime.datetime.now().isoformat(), 'Starting', output_name, flush=True)
        with (directory / 'train.log').open('w') as log:
            result = subprocess.run(train_cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            code = result.returncode
            if code == 0:
                # Isaac shutdown can mask a Python exception: verify the actual artifact.
                check = '''import json,sys,torch
from pathlib import Path
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
assert c['agent_steps']==int(sys.argv[2]), 'Training budget not reached'
assert all(torch.isfinite(v).all() for v in c['model'].values()), 'Nonfinite model'
r=c['env_runtime']
assert r['finger_gait']['enabled'] and r['finger_gait']['target_angvel']==.5
summary=dict(agent_steps=c['agent_steps'],epoch_num=c['epoch_num'],episode_length_s=r['episode_length_s'])
Path(sys.argv[3]).write_text(json.dumps(summary,indent=2)+'\\n')
print('Verified final checkpoint:',json.dumps(summary),flush=True)
'''
                code = subprocess.run([args.python, '-c', check, str(checkpoint), str(budget),
                                       str(directory / 'train_summary.json')], env=env,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
        (directory / 'train.exit').write_text(str(code) + '\n')
        if code:
            raise RuntimeError(f'Training failed: {directory}; exit={code}')
        with (directory / 'eval.log').open('w') as log:
            code = subprocess.run(eval_cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        evaluation = (directory / 'eval.log').read_text(errors='replace')
        if code == 0 and ('[FULL-GRAVITY EVAL]' not in evaluation or 'Traceback (most recent call last)' in evaluation):
            code = 1
        (directory / 'eval.exit').write_text(str(code) + '\n')
        if code:
            raise RuntimeError(f'Evaluation failed: {directory}; exit={code}')
        print('Completed training and evaluation:', output_name, flush=True)
    print('Both screening runs completed; compare eval.log before extending a run.', flush=True)


if __name__ == '__main__':
    main()
