"""Bound remote worker lifetime to the shared scheduler-controller lease."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lease',required=True)
    p.add_argument('--lease-timeout',type=int,default=90)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    path=Path(args.lease)
    previous=path.read_text()
    if not json.loads(previous)['active']:
        raise SystemExit('scheduler controller lease is inactive')
    child=subprocess.Popen(command)
    def stop(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    last=time.monotonic()
    try:
        while child.poll() is None:
            current=path.read_text()
            if not json.loads(current)['active']:
                raise RuntimeError('scheduler controller stopped')
            if current!=previous:
                last=time.monotonic(); previous=current
            if time.monotonic()-last>args.lease_timeout:
                raise RuntimeError('scheduler controller heartbeat expired')
            time.sleep(1)
        raise SystemExit(child.returncode)
    finally:
        if child.poll() is None:
            child.terminate()  # launch_acceptance handles TERM and kills its own worker group
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__=='__main__':
    main()
