import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/tan/hora2/BrainCo_hora')
RUN = ROOT / 'outputs/revo3_right/run_octagonal_prism_gait_h16_p120_16384_i4000_s42_20260920_182743'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
expected = json.loads((RUN / 'download_sha256.json').read_text())
for name, sha in expected.items():
    assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == sha, name
print('Verified downloaded files:', len(expected), flush=True)
import torch
from hora.object_registry import get_object_task_spec
from hora.utils.grasp_cache import validate_object_runtime
checkpoint_path = RUN / 'stage1_nn/last.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
assert all(torch.isfinite(t).all() for t in checkpoint['model'].values())
cache = ROOT / 'cache/revo3_right_grasp_octagonal_prism.npy'
validate_object_runtime(checkpoint['env_runtime'], {
    'object': json.loads(json.dumps(get_object_task_spec('octagonal_prism').metadata())),
    'grasp_cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
})
assert checkpoint['env_runtime']['decimation'] == 6
print('last.pth:', checkpoint['epoch_num'], 'epochs;', checkpoint['agent_steps'], 'steps', flush=True)
diag = RUN / 'diagnostics/play_20260920_201027'
diag.mkdir(parents=True, exist_ok=False)
command = [sys.executable, '-u', 'train.py', '--task', 'octagonal_prism', '--algo', 'PPO', '--test', '--num_envs', '1', '--device', 'cuda:0', '--real-time', '--physics_hz', '120', '--episode_length_s', '40', '--finger_gait', '--rotation_speed', '.5', '--cache_file', cache.name, '--checkpoint', str(checkpoint_path), '--output_name', str(diag / 'runtime'), '--camera_eye', '.55', '-.55', '1.85', '--camera_lookat', '0', '-.03', '1.55', '--kit_args', '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=4']
env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PXR_WORK_THREAD_LIMIT='4')
env.pop('HORA_SKIP_SIM_CLOSE', None)
with (diag / 'play.log').open('w') as log:
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(started_at=datetime.datetime.now().astimezone().isoformat(), pid=child.pid, epoch=checkpoint['epoch_num'], agent_steps=checkpoint['agent_steps'], checkpoint_sha256=expected['stage1_nn/last.pth'], command=command)
(diag / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
