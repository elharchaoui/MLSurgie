import argparse
import csv
import json
from pathlib import Path

from .artifacts import load, read_json, replay
from .contracts import Budget, Feature, Goal
from .data import Table
from .demo import demo
from .editors import OllamaEditor, ScriptedEditor
from .runner import build


def main():
    parser = argparse.ArgumentParser(description="Construct networks through direct weight programs")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Run the clearly labeled offline known-rule fixture")
    d.add_argument("--output", required=True)
    b = sub.add_parser("build", help="Build from labeled CSV using an explicit JSON configuration")
    b.add_argument("--config", required=True)
    b.add_argument("--dataset", required=True)
    b.add_argument("--output", required=True)
    editors = b.add_mutually_exclusive_group(required=True)
    editors.add_argument("--ollama-model")
    editors.add_argument("--programs", help="JSON array of scripted proposals, not a live LLM")
    b.add_argument("--ollama-url", default="http://localhost:11434")
    r = sub.add_parser("replay", help="Reconstruct and verify the exported network from programs")
    r.add_argument("run")
    p = sub.add_parser("predict", help="Predict using a saved network; no editor required")
    p.add_argument("run")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            result = demo(args.output)
        elif args.command == "build":
            config = read_json(args.config)
            allowed = {"goal", "features", "budget", "seed", "group_column"}
            if set(config) - allowed:
                raise ValueError(f"Unknown config fields: {set(config) - allowed}")
            goal = Goal(**config["goal"])
            features = [Feature(**f) for f in config["features"]]
            table = Table.from_csv(args.dataset, features, goal, config.get("group_column"))
            editor = OllamaEditor(args.ollama_model, args.ollama_url) if args.ollama_model else ScriptedEditor(read_json(args.programs))
            result = build(table=table, editor=editor, output=args.output,
                           budget=Budget(**config.get("budget", {})), seed=config.get("seed", 42))
        elif args.command == "replay":
            print(json.dumps({"verified_state_hash": replay(args.run).fingerprint()}))
            return
        else:
            predictor = load(args.run)
            with Path(args.dataset).open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            probabilities = predictor.predict_proba(rows)
            labels = [predictor.classes[int(p >= predictor.threshold)] for p in probabilities[:, 1]]
            with Path(args.output).open("x", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["prediction", "negative_probability", "positive_probability"])
                writer.writerows((label, *pair) for label, pair in zip(labels, probabilities))
            print(f"Wrote {len(rows)} predictions to {args.output}")
            return
        print(json.dumps({"run": str(result.path), "status": result.report["status"],
                          "selected_iteration": result.report["selected_iteration"],
                          "final_test": result.report["final_test"]}, indent=2))
        if result.model is None:
            raise SystemExit(2)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
