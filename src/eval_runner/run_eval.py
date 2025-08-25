import argparse
import json
from metrics.utility import eval_utility
from metrics.safety import eval_safety
from metrics.fairness import eval_fairness

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model id/path")
    parser.add_argument("--baseline", type=str)
    parser.add_argument("--candidate", type=str)
    args = parser.parse_args()

    # Run eval suites
    utility = eval_utility(args.model)
    safety  = eval_safety(args.model)
    fairness = eval_fairness(args.model)

    report = {"utility": utility, "safety": safety, "fairness": fairness}
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Eval complete. Report written to eval_report.json")

if __name__ == "__main__":
    main()