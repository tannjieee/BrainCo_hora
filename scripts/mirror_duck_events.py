#!/usr/bin/env python3
"""Mirror growing event files; outages keep the last successful local data."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--source', default=(
        'tanjie@127.0.0.1:/home/tanjie/BrainCo_hora/outputs/revo3_right/'
        'run_rubber_duck_gait_h16_p120_16384_i4000_s42_20260920/stage1_tb/'),
        help='Remote TensorBoard event directory (rsync source).')
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    status_path = args.destination.parent / 'tensorboard_sync.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    # Event files only append. Verify existing prefixes and compress transfers;
    # the first full copy can be much slower than subsequent tiny increments.
    command = ['rsync', '-az', '--append-verify', '--timeout=20',
               '-e', 'ssh -S /tmp/brainco-collect.sock -p 22020 -o BatchMode=yes -o ConnectTimeout=10',
               '--include=events.out.tfevents.*', '--exclude=*',
               args.source.rstrip('/') + '/',
               str(args.destination) + '/']
    while True:
        now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
        status['last_attempt'] = now
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(result.stderr.strip()[-500:])
            status.update(last_success=datetime.datetime.now().astimezone().isoformat(timespec='seconds'), error=None)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            status['error'] = str(exc)
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(status_path)
        print(f'[{now}] event sync: {"offline; retaining previous data" if status["error"] else "OK"}', flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
