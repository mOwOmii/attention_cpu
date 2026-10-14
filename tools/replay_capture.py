"""Replay a full capture's input, weights, dropout RNG, and upstream gradient."""

import argparse
import json

from attention_cpu.replay import replay_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record")
    parser.add_argument("--output", help="optional directory to retain the regenerated full capture")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    args = parser.parse_args()
    report = replay_capture(args.record, output=args.output, atol=args.atol, rtol=args.rtol)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
