import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np

ROOT = Path('/home/tanjie/BrainCo_hora')
RUN = ROOT / 'remote_runs/octagonal_collect_20260920_181435'
os.chdir(ROOT)
os.environ.update(TMPDIR='/home/tanjie/.cache/hora/tmp', OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1', CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PXR_WORK_THREAD_LIMIT='8')
Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)
KIT = '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8'
COMMON = [sys.executable, '-u', 'gen_grasp.py', '--task', 'octagonal_prism', '--headless', '--device', 'cuda:2', '--seed', '42', '--noise_scale', '.15', '--min_contact_fingertips', '3', '--min_live_contact_fingertips', '3', '--episode_length_s', '5', '--progress_interval', '10', '--kit_args', KIT]
SMOKE_CACHE = '_smoke_octagonal_20260920_181435.npy'
MAIN_CACHE = 'revo3_right_grasp_octagonal_prism.npy'
SMOKE = COMMON + ['--num_envs', '128', '--target_count', '16', '--yaw_bins', '1', '--max_steps', '500', '--cache_file', SMOKE_CACHE]
MAIN = COMMON + ['--num_envs', '2048', '--target_count', '8192', '--yaw_bins', '8', '--free_fingers', 'index', 'middle', 'ring', 'little', '--max_steps', '20000', '--cache_file', MAIN_CACHE]

def validate(name, count):
    path = ROOT / 'cache' / name
    report = json.loads(path.with_suffix('.report.json').read_text())
    assert report['complete'], report
    assert report['accepted'] == report['targets'], report
    assert sum(report['accepted']) == count
    data = np.load(path, allow_pickle=False)
    assert data.shape == (count, 28), data.shape
    assert data.dtype == np.float32 and np.isfinite(data).all()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report['cache_sha256']
    return {'shape': list(data.shape), 'sha256': report['cache_sha256'], 'accepted': report['accepted']}

def launch(stage, command, cache, count):
    with (RUN / (stage + '.log')).open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        (RUN / (stage + '.pid')).write_text(str(child.pid) + '\n')
        code = child.wait()
    try:
        assert code == 0, f'{stage} process returned {code}'
        result = validate(cache, count)
        (RUN / (stage + '.validation.json')).write_text(json.dumps(result, indent=2) + '\n')
    except BaseException:
        (RUN / (stage + '.exit')).write_text('1\n')
        raise
    (RUN / (stage + '.exit')).write_text('0\n')
    print(stage, 'validated', result, flush=True)

try:
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in gpu.splitlines()}
    assert free[2] >= 16000, f'GPU 2 insufficient free memory: {free[2]} MiB'
    for cache in [SMOKE_CACHE, MAIN_CACHE]:
        assert not (ROOT / 'cache' / cache).exists(), f'Cache already exists: {cache}'
    (RUN / 'launch.json').write_text(json.dumps({'started_at': datetime.datetime.now().astimezone().isoformat(), 'task': 'octagonal_prism', 'gpu': 2, 'num_envs': 2048, 'target_count': 8192, 'smoke_command': SMOKE, 'collect_command': MAIN, 'gpu_at_start': gpu, 'manifest_sha256': hashlib.sha256((ROOT / 'assets/usd/objects/manifest.json').read_bytes()).hexdigest()}, indent=2) + '\n')
    launch('smoke', SMOKE, SMOKE_CACHE, 16)
    launch('collect', MAIN, MAIN_CACHE, 8192)
except BaseException:
    traceback.print_exc()
    if not (RUN / 'collect.exit').exists():
        (RUN / 'collect.exit').write_text('1\n')
    sys.exit(1)
