"""Run TP/DP tests and report separately whether actual Gloo communication passed."""

import argparse
from pathlib import Path
import unittest

from attention_cpu.analyze import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="runs/parallel_validation/report.json")
    parser.add_argument("--require-gloo",action="store_true",help="exit 2 if actual Gloo tests are skipped")
    parser.add_argument("--offline-capture-root",help="optional new directory for explicitly simulated TP=2/DP=2 captures")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_parallel")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    skips = [{"test":str(test),"reason":reason} for test,reason in result.skipped]
    actual_skipped = any("test_real_gloo" in row["test"] for row in skips)
    failed_real = any("test_real_gloo" in str(test) for test,_ in result.failures+result.errors)
    report = {"tests_run":result.testsRun,"passed_count":result.testsRun-len(result.failures)-len(result.errors)-len(skips),
              "failures":[{"test":str(t),"traceback":trace} for t,trace in result.failures+result.errors],
              "skipped":skips,"tests_successful":result.wasSuccessful(),
              "actual_gloo_status":"blocked_by_environment" if actual_skipped else "failed" if failed_real else "passed",
              "stage4_fully_validated":result.wasSuccessful() and not actual_skipped,
              "offline_transport":"in-process threaded test SUM; not a network backend",
              "actual_topologies":[[2,1],[1,2],[2,2]],
              "coverage":"forward, dInput, parameter gradients, SGD update, unequal valid counts, accumulation, checkpoint resume"}
    if args.offline_capture_root and result.wasSuccessful():
        from tests.test_parallel import run_offline_case
        from attention_cpu.parallel.replay import replay_parallel_captures
        cases = run_offline_case(2,2,capture_root=args.offline_capture_root)
        replay_reports = []
        for dp in range(2):
            paths = [cases[dp*2+tp]["records"][0] for tp in range(2)]
            replay_reports.append(replay_parallel_captures(paths,atol=1e-10,rtol=1e-8))
        report["offline_equivalence"] = {"transport":"in_process_test_simulation","ranks":cases}
        report["offline_replay"] = replay_reports
        if not all(r["passed"] for r in replay_reports):
            report["tests_successful"] = False
            report["stage4_fully_validated"] = False
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,report)
    print(f"Report: {path}; actual Gloo: {report['actual_gloo_status']}")
    if not report["tests_successful"]:
        raise SystemExit(1)
    if args.require_gloo and actual_skipped:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
