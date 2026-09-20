#!/usr/bin/env python3
"""Reconnect the monitoring SSH master; credentials stay in memory only."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import pexpect

CONTROL = '/tmp/brainco-collect.sock'
AUTH = '/tmp/brainco-monitor-auth.sock'
CHECK = ['ssh', '-S', CONTROL, '-O', 'check', '-p', '22020', 'tanjie@127.0.0.1']


def ready():
    return subprocess.run(CHECK, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          timeout=5).returncode == 0


def main():
    if ready():
        raise SystemExit('A monitoring SSH master already exists; refusing to replace it.')
    Path(AUTH).unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX)
    server.bind(AUTH)
    os.chmod(AUTH, 0o600)
    server.listen(1)
    server.settimeout(120)
    print('Waiting for credentials on private local socket; no passwords saved to disk.', flush=True)
    try:
        conn, _ = server.accept()
        with conn, conn.makefile('r') as stream:
            passwords = json.loads(stream.readline(4096))
        if len(passwords) != 2 or not all(isinstance(p, str) for p in passwords):
            raise ValueError('Expected two credentials')
    finally:
        server.close()
        Path(AUTH).unlink(missing_ok=True)
    # Apply keepalives to both SSH hops.
    proxy = ('ssh -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=15 '
             '-o ServerAliveInterval=15 -o ServerAliveCountMax=3 '
             '-o NumberOfPasswordPrompts=1 -W %h:%p gpu-access@47.115.128.206')
    args = ['-M', '-S', CONTROL, '-o', 'ControlPersist=no', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
            '-o', 'NumberOfPasswordPrompts=1', '-o', 'ProxyCommand=' + proxy,
            '-p', '22020', 'tanjie@127.0.0.1', '-N']
    delay = 5
    while True:
        Path(CONTROL).unlink(missing_ok=True)
        child = None
        try:
            child = pexpect.spawn('ssh', args, encoding='utf-8', timeout=3, echo=False)
            deadline = time.monotonic() + 60
            authenticated = False
            while time.monotonic() < deadline:
                match = child.expect([
                    r"gpu-access@[^\r\n]*[Pp]assword:",
                    r"tanjie@[^\r\n]*[Pp]assword:",
                    'Permission denied', pexpect.EOF, pexpect.TIMEOUT])
                if match < 2:
                    child.sendline(passwords[match])
                elif match == 2:
                    raise SystemExit('SSH authentication rejected; reconnect with valid credentials.')
                elif match == 3:
                    raise RuntimeError('SSH transport exited')
                if ready():
                    authenticated = True
                    break
            if not authenticated:
                raise RuntimeError('SSH connection deadline exceeded')
            print('SSH monitoring connection active; transport failures reconnect automatically.', flush=True)
            delay = 5
            child.expect(pexpect.EOF, timeout=None)
            print('SSH transport disconnected.', flush=True)
        except (OSError, RuntimeError, pexpect.ExceptionPexpect, subprocess.TimeoutExpired) as exc:
            # Never print the PTY authentication buffer.
            print(f'SSH retry: {type(exc).__name__}', flush=True)
        finally:
            if child is not None:
                child.close(force=True)
            Path(CONTROL).unlink(missing_ok=True)
        print(f'Reconnecting in {delay}s.', flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 60)


if __name__ == '__main__':
    def stop(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    main()
