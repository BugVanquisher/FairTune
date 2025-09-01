import argparse
import json
import tempfile
import os
from metrics.utility import eval_utility
from metrics.safety import eval_safety
from metrics.fairness import eval_fairness

def compute_delta(baseline, candidate):
    delta = {}
    for key in baseline:
        if key in candidate:
            delta[key] = {}
            for subkey in baseline[key]:
                if (
                    subkey in candidate[key]
                    and isinstance(baseline[key][subkey], (int, float))
                    and isinstance(candidate[key][subkey], (int, float))
                ):
                    delta[key][subkey] = candidate[key][subkey] - baseline[key][subkey]
    return delta

def atomic_write_json(obj, path):
    dir_ = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False) as tf:
        json.dump(obj, tf, indent=2)
        tf.flush()
        os.fsync(tf.fileno())
        tempname = tf.name
    os.replace(tempname, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model id/path")
    parser.add_argument("--baseline", type=str)
    parser.add_argument("--candidate", type=str)
    args = parser.parse_args()

    if args.baseline and args.candidate:
        # Baseline vs candidate comparison
        baseline_utility = eval_utility(args.baseline)
        baseline_safety  = eval_safety(args.baseline)
        baseline_fairness = eval_fairness(args.baseline)
        candidate_utility = eval_utility(args.candidate)
        candidate_safety  = eval_safety(args.candidate)
        candidate_fairness = eval_fairness(args.candidate)
        baseline_report = {"utility": baseline_utility, "safety": baseline_safety, "fairness": baseline_fairness}
        candidate_report = {"utility": candidate_utility, "safety": candidate_safety, "fairness": candidate_fairness}
        delta = compute_delta(baseline_report, candidate_report)
        out = {
            "baseline": baseline_report,
            "candidate": candidate_report,
            "delta": delta
        }
        atomic_write_json(out, "eval_report.json")
        print("✅ Eval complete. Baseline vs candidate report written to eval_report.json")
    else:
        # Backwards compatible: single model
        model = args.model or args.candidate or args.baseline
        if not model:
            raise ValueError("Must provide --model or both --baseline and --candidate")
        utility = eval_utility(model)
        safety  = eval_safety(model)
        fairness = eval_fairness(model)
        report = {"utility": utility, "safety": safety, "fairness": fairness}
        atomic_write_json(report, "eval_report.json")
        print("✅ Eval complete. Report written to eval_report.json")

if __name__ == "__main__":
    main()