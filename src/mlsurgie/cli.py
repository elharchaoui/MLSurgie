import argparse
import csv
import json
from pathlib import Path

from .artifacts import load, read_json, replay
from .contracts import Budget, Feature, Goal
from .data import Table
from .demo import demo
from .editors import OllamaEditor, OpenRouterEditor, ScriptedEditor
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
    editors.add_argument("--openrouter-model", help="Explicit OpenRouter model ID; key read from environment")
    editors.add_argument("--programs", help="JSON array of scripted proposals, not a live LLM")
    b.add_argument("--ollama-url", default="http://localhost:11434")
    b.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY",
                   help="Environment variable holding the OpenRouter key (never pass the key itself)")
    a = sub.add_parser("agent", help="Use the optional OpenHands coding harness with Docker construction")
    a.add_argument("--config", required=True)
    a.add_argument("--dataset", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--model", required=True, help="LiteLLM model ID, e.g. openrouter/vendor/model")
    a.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    a.add_argument("--base-url")
    a.add_argument("--image", default="mlsurgie-constructor:local")
    a.add_argument("--agent-steps", type=int, default=12)
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
        elif args.command in {"build", "agent"}:
            config = read_json(args.config)
            allowed = {"goal", "features", "budget", "seed", "group_column"}
            if set(config) - allowed:
                raise ValueError(f"Unknown config fields: {set(config) - allowed}")
            goal = Goal(**config["goal"])
            features = [Feature(**f) for f in config["features"]]
            table = Table.from_csv(args.dataset, features, goal, config.get("group_column"))
            if args.command == "agent":
                from .harness import OpenHandsEditor
                editor = OpenHandsEditor(args.model, args.output, args.api_key_env,
                                         args.image, args.base_url, args.agent_steps)
                editor.check()
                limits = dict(config.get("budget", {}))
                limits.setdefault("support_examples", len(table.rows))
                try:
                    result = build(table=table, editor=editor, output=args.output,
                                   budget=Budget(**limits), seed=config.get("seed", 42),
                                   dynamic_architecture=True, isolated_evaluation=True)
                finally:
                    editor.close()
            elif args.ollama_model:
                editor = OllamaEditor(args.ollama_model, args.ollama_url)
            elif args.openrouter_model:
                editor = OpenRouterEditor(args.openrouter_model, args.openrouter_api_key_env)
            else:
                editor = ScriptedEditor(read_json(args.programs))
            if args.command == "build":
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
        if result.model is None or result.report["status"] == "final_evaluation_error":
            raise SystemExit(2)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
