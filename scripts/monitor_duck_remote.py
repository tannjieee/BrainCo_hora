#!/usr/bin/env python3
"""Mirror remote collection/training logs using an authenticated SSH control socket.

No credentials are stored. Reconnect the control master if the connection expires.
"""
import argparse
import datetime
import json
import os
import re
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', default='/tmp/brainco-collect.sock')
    parser.add_argument('--host', default='tanjie@127.0.0.1')
    parser.add_argument('--port', type=int, default=22020)
    parser.add_argument('--remote-run', required=True)
    parser.add_argument('--local-dir', required=True)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--kind', choices=('collection', 'training'), default='collection')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('interval must be at least one second')
    destination = Path(args.local_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'monitor.pid').write_text(str(os.getpid()) + '\n')
    # The directory is passed as a Python literal through stdin, never shell interpolation.
    query = '''import json, subprocess
from pathlib import Path
run = Path(REMOTE_RUN)
files = {}
for name in ('setup.log', 'setup.exit', 'smoke.log', 'smoke.exit', 'collect.log', 'collect.exit', 'collect.pid', 'launch.json', 'train.log', 'train.exit', 'eval.log', 'eval.exit'):
    path = run / name
    if path.is_file():
        files[name] = path.read_text(errors='replace')
queue_exit = run.parent / 'queue.exit'
if queue_exit.is_file():
    files['queue.exit'] = queue_exit.read_text()
gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
print(json.dumps({'files': files, 'gpu': gpu.stdout, 'remote_run': str(run)}))
'''.replace('REMOTE_RUN', repr(args.remote_run))
    command = ['ssh', '-S', args.socket, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-p', str(args.port), args.host, 'python3', '-']
    print(f'Local log mirror: {destination}', flush=True)
    previous_progress = None
    while True:
        timestamp = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
        try:
            result = subprocess.run(command, input=query, capture_output=True, text=True, timeout=45)
            if result.returncode:
                raise RuntimeError(result.stderr.strip())
            snapshot = json.loads(result.stdout)
            snapshot['local_updated_at'] = timestamp
            for name, contents in snapshot['files'].items():
                temporary = destination / (name + '.tmp')
                temporary.write_text(contents)
                temporary.replace(destination / name)
            files = snapshot['files']
            stage = ('collection finished, exit=' + files['collect.exit'].strip() if 'collect.exit' in files
                     else 'collecting' if 'collect.log' in files
                     else 'simulation smoke test' if 'smoke.log' in files
                     else 'runtime setup')
            lines = files.get('collect.log', files.get('smoke.log', files.get('setup.log', ''))).splitlines()
            progress = [line for line in lines if line.startswith(('[PROGRESS]', '[COVERAGE]', '[INCOMPLETE]', '[INFO] Saved'))]
            current = '\n'.join(progress[-2:] if progress else lines[-3:])
            if args.kind == 'training':
                if 'eval.exit' in files:
                    stage = 'evaluation finished, exit=' + files['eval.exit'].strip()
                elif 'eval.log' in files:
                    stage = 'evaluating'
                elif 'train.exit' in files:
                    stage = 'training finished, exit=' + files['train.exit'].strip()
                else:
                    stage = 'training' if 'train.log' in files else 'queued'
                    if stage == 'queued' and 'queue.exit' in files:
                        stage = 'queue stopped before this run, exit=' + files['queue.exit'].strip()
                if 'eval.log' in files:
                    current = '\n'.join(files['eval.log'].splitlines()[-24:])
                else:
                    log = files.get('train.log', '')
                    block = log.rsplit('Learning iteration ', 1)[-1]
                    keys = ('Computation:', 'Mean reward:', 'Mean episode length:',
                            'Total timesteps:', 'ETA:', 'gait/completed_one_turn_rate:',
                            'gait/completed_mean_turns:', 'gait/completed_mean_net_turns:',
                            'gait/blocked_push:', 'gait/support_gate:', 'stable_rotation_rate:')
                    selected = [line.strip() for line in block.splitlines() if line.strip().startswith(keys)]
                    if 'Learning iteration ' in log:
                        selected.insert(0, 'Iteration: ' + block.splitlines()[0].strip())
                    # Trainer HH:MM:SS output wraps at 24 h; show a days-aware estimate.
                    launch = json.loads(files.get('launch.json', '{}'))
                    steps = re.search(r'Total timesteps:\s*(\d+)', block)
                    fps = re.search(r'Computation:\s*(\d+) steps/s', block)
                    if steps and fps and int(fps[1]) > 0 and launch.get('max_agent_steps'):
                        seconds = int(max(0, launch['max_agent_steps'] - int(steps[1])) / int(fps[1]))
                        days, seconds = divmod(seconds, 86400)
                        hours, seconds = divmod(seconds, 3600)
                        minutes, seconds = divmod(seconds, 60)
                        selected = [line for line in selected if not line.startswith('ETA:')]
                        selected.append(f'ETA (recent FPS): {days}d {hours:02}:{minutes:02}:{seconds:02}')
                    current = '\n'.join(selected) if selected else '\n'.join(log.splitlines()[-3:])
            snapshot['stage'] = stage
            snapshot['progress'] = current
            temporary = destination / 'status.json.tmp'
            temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(destination / 'status.json')
            print(f'[{timestamp}] {stage}', flush=True)
            if current != previous_progress:
                print(current, flush=True)
                previous_progress = current
            print('GPU index, memory used, utilization:\n' + snapshot['gpu'].strip(), flush=True)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f'[{timestamp}] Monitor connection error: {exc}', flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
