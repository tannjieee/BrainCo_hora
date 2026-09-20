import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path('/home/tanjie/BrainCo_hora')
RUN = ROOT / 'remote_runs/octagonal_train_h16_p120_20260920_182743'
OUTPUT = 'outputs/revo3_right/run_octagonal_prism_gait_h16_p120_16384_i4000_s42_20260920_182743'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.update(TMPDIR='/home/tanjie/.cache/hora/tmp', OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1', CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PXR_WORK_THREAD_LIMIT='8', TERM='xterm')
os.environ.pop('HORA_SKIP_SIM_CLOSE', None)
COMMAND = [sys.executable, '-u', 'train.py', '--task', 'octagonal_prism', '--algo', 'PPO', '--headless', '--device', 'cuda:2', '--finger_gait', '--rotation_speed', '.5', '--fixed_train_gravity', '9.81', '--cache_file', 'revo3_right_grasp_octagonal_prism.npy', '--num_envs', '16384', '--horizon_length', '16', '--gamma', '.995', '--episode_length_s', '40', '--minibatch_size', '32768', '--max_iterations', '4000', '--save_frequency', '50', '--physics_hz', '120', '--seed', '42', '--output_name', Path(OUTPUT).name, '--kit_args', '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8']
try:
    import numpy as np
    from hora.object_registry import get_object_task_spec
    cache = ROOT / 'cache/revo3_right_grasp_octagonal_prism.npy'
    report = json.loads(cache.with_suffix('.report.json').read_text())
    assert report['complete'] and report['accepted'] == report['targets'] and sum(report['accepted']) == 8192
    assert report['object'] == json.loads(json.dumps(get_object_task_spec('octagonal_prism').metadata()))
    cache_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
    assert cache_sha == report['cache_sha256']
    states = np.load(cache, allow_pickle=False)
    assert states.shape == (8192, 28) and states.dtype == np.float32 and np.isfinite(states).all()
    expected = json.loads((RUN / 'source_sha256.json').read_text())
    for name, sha in expected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    assert not (ROOT / OUTPUT).exists(), 'Refusing to overwrite existing run'
    gpu = subprocess.check_output(['nvidia-smi', '--id=2', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    assert int(gpu.split(',')[0]) >= 24000, f'GPU 2 needs at least 24000 MiB free: {gpu}'
    launch = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), task='octagonal_prism', gpu=2, num_envs=16384, horizon_length=16, gamma=.995, episode_length_s=40, policy_hz=20, physics_hz=120, decimation=6, max_iterations=4000, max_agent_steps=1048576000, save_frequency=50, minibatch_size=32768, rotation_speed=.5, gravity=9.81, seed=42, output=OUTPUT, checkpoint=None, cache_sha256=cache_sha, source_sha256=expected, command=COMMAND, gpu_at_start=gpu)
    (RUN / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    with (RUN / 'train.log').open('w') as log:
        process = subprocess.Popen(COMMAND, stdout=log, stderr=subprocess.STDOUT)
        (RUN / 'train.pid').write_text(str(process.pid) + '\n')
        code = process.wait()
    assert code == 0, f'Train process exited {code}'
    import torch
    checkpoint = torch.load(ROOT / OUTPUT / 'stage1_nn/last.pth', map_location='cpu', weights_only=False)
    assert checkpoint['epoch_num'] == 4000 and checkpoint['agent_steps'] == 1048576000
    assert checkpoint['horizon_length'] == 16 and checkpoint['env_runtime']['decimation'] == 6
    assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())
    (RUN / 'train.exit').write_text('0\n')
    print('Final checkpoint verified: iteration 4000', flush=True)
except BaseException:
    traceback.print_exc()
    (RUN / 'train.exit').write_text('1\n')
    sys.exit(1)
