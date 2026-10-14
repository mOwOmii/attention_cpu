"""Replay one DP replica's full TP capture group using a single full CPU module."""

import argparse
import json

from attention_cpu.analyze import write_json
from attention_cpu.parallel.replay import replay_parallel_captures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records",nargs="+")
    parser.add_argument("--output",required=True)
    parser.add_argument("--atol",type=float,default=1e-5)
    parser.add_argument("--rtol",type=float,default=1e-4)
    args = parser.parse_args()
    report = replay_parallel_captures(args.records,atol=args.atol,rtol=args.rtol)
    write_json(args.output,report)
    print(json.dumps(report,indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
