"""Real X11 window-close regression; run with the Isaac Lab Python environment.

python tests/check_window_close.py --case editor
python tests/check_window_close.py --case physics
python tests/check_window_close.py --case play --checkpoint /path/to/duck/last.pth
"""
import argparse
import ctypes as C
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def close_window(pid):
    """Send WM_DELETE_WINDOW only to the Isaac Sim window owned by pid."""
    tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
    for line in tree.splitlines():
        if '("Isaac Sim' not in line:
            continue
        window = re.search(r'0x[0-9a-fA-F]+', line).group()
        prop = subprocess.check_output(['xprop', '-id', window, '_NET_WM_PID'], text=True)
        if not re.search(rf'= {pid}\s*$', prop):
            continue
        break
    else:
        raise RuntimeError(f'No Isaac Sim window owned by PID {pid}')

    class Data(C.Union):
        _fields_ = [('b', C.c_char * 20), ('s', C.c_short * 10), ('l', C.c_long * 5)]

    class Client(C.Structure):
        _fields_ = [('type', C.c_int), ('serial', C.c_ulong), ('send_event', C.c_int),
                    ('display', C.c_void_p), ('window', C.c_ulong), ('message_type', C.c_ulong),
                    ('format', C.c_int), ('data', Data)]

    class Event(C.Union):
        _fields_ = [('client', Client), ('padding', C.c_long * 24)]

    x = C.CDLL('libX11.so.6')
    x.XOpenDisplay.argtypes = [C.c_char_p]
    x.XOpenDisplay.restype = C.c_void_p
    x.XInternAtom.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
    x.XInternAtom.restype = C.c_ulong
    x.XSendEvent.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_long, C.POINTER(Event)]
    x.XFlush.argtypes = [C.c_void_p]
    x.XCloseDisplay.argtypes = [C.c_void_p]
    display = x.XOpenDisplay(None)
    if not display:
        raise RuntimeError('Cannot open X11 DISPLAY')
    try:
        event = Event()
        event.client.type = 33  # ClientMessage
        event.client.display = display
        event.client.window = int(window, 16)
        event.client.message_type = x.XInternAtom(display, b'WM_PROTOCOLS', 0)
        event.client.format = 32
        event.client.data.l[0] = x.XInternAtom(display, b'WM_DELETE_WINDOW', 0)
        assert x.XSendEvent(display, event.client.window, 0, 0, C.byref(event))
        x.XFlush(display)
    finally:
        x.XCloseDisplay(display)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('editor', 'physics', 'play'), default='editor')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.case == 'play' and args.checkpoint is None:
        parser.error('--case play requires --checkpoint')
    directory = Path(tempfile.mkdtemp(prefix=f'hora-close-{args.case}-'))
    manifest = ROOT / 'assets/usd/objects/manifest.json'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    command = [sys.executable, '-u']
    if args.case == 'play':
        command += ['train.py', '--task', 'rubber_duck', '--test', '--num_envs', '1',
                    '--physics_hz', '120', '--finger_gait', '--rotation_speed', '.5',
                    '--cache_file', 'revo3_right_grasp_rubber_duck_gait_v2.npy',
                    '--checkpoint', str(args.checkpoint.resolve()),
                    '--output_name', str(directory / 'play'), '--real-time']
        marker = 'RunningMeanStd:  (1,)'
    else:
        command += ['tools/view_init_pose.py', '--task', 'octagonal_prism', '--num_envs', '1',
                    '--edit_pose' if args.case == 'editor' else '--physics']
        marker = '[VIEW] Frozen render mode.' if args.case == 'editor' else '[PHYSICS] Stepping'
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    log_path = directory / 'app.log'
    with log_path.open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while marker not in log_path.read_text(errors='replace'):
                if child.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f'Application did not become ready; see {log_path}')
                time.sleep(.25)
            time.sleep(2)
            started = time.monotonic()
            close_window(child.pid)
            code = child.wait(timeout=20)
            elapsed = time.monotonic() - started
            assert code == 0, f'Exit status {code}; see {log_path}'
            assert 'Traceback (most recent call last)' not in log_path.read_text(errors='replace')
            assert hashlib.sha256(manifest.read_bytes()).hexdigest() == digest, 'Seed manifest changed'
            print(f'PASS {args.case}: WM_DELETE_WINDOW -> exit 0 in {elapsed:.2f}s; '
                  f'seed unchanged; log={log_path}', flush=True)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
