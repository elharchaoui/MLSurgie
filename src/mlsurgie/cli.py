import argparse
import csv
import json
from pathlib import Path

from .artifacts import load, read_json, replay
from .config import load_task
from .demo import demo
from .editors import OllamaEditor, OpenRouterEditor, RandomSearchEditor, ScriptedEditor
from .protocol import Track, build


def add_task_arguments(parser):
    parser.add_argument("--config", required=True, help="Task configuration JSON")
    parser.add_argument("--dataset", help="CSV path; defaults to the config's `dataset`")
    parser.add_argument("--names", choices=["semantic", "anonymized"], default="semantic",
                        help="Track: real feature names, or anonymized f01... names (tests reliance on priors)")
    parser.add_argument("--policy", choices=["strict", "open"], default="strict",
                        help="Track: strict rejects code flagged by the fitting audit; open only records flags")


def add_agent_arguments(parser, required=True):
    parser.add_argument("--model", required=required, help="LiteLLM model ID, e.g. openrouter/vendor/model")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--base-url")
    parser.add_argument("--image", default="mlsurgie-constructor:local")
    parser.add_argument("--agent-steps", type=int, default=24, help="Maximum tool calls per submitted candidate")


def make_agent(args, output):
    from .harness import OpenHandsEditor
    editor = OpenHandsEditor(args.model, output, args.api_key_env, args.image, args.base_url, args.agent_steps)
    editor.check()
    return editor


def main():
    parser = argparse.ArgumentParser(description="Weight-space programming: construct networks by writing their weights")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Run the clearly labeled offline known-rule fixture")
    d.add_argument("--output", required=True)

    b = sub.add_parser("build", help="JSON-only editor loop (Ollama, OpenRouter, scripted, or random-search control)")
    add_task_arguments(b)
    b.add_argument("--output", required=True)
    editors = b.add_mutually_exclusive_group(required=True)
    editors.add_argument("--ollama-model")
    editors.add_argument("--openrouter-model", help="Explicit OpenRouter model ID; key read from environment")
    editors.add_argument("--programs", help="JSON array of scripted proposals, not a live LLM")
    editors.add_argument("--random-search", type=int, metavar="SEED", help="Non-LLM search control with this seed")
    b.add_argument("--ollama-url", default="http://localhost:11434")
    b.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY",
                   help="Environment variable holding the OpenRouter key (never pass the key itself)")

    a = sub.add_parser("agent", help="Free coding agent (OpenHands) with data files, workspace, skills, and Docker sandbox")
    add_task_arguments(a)
    a.add_argument("--output", required=True)
    add_agent_arguments(a)

    s = sub.add_parser("bench", help="Run a benchmark suite (tasks x seeds) against trained baselines")
    s.add_argument("--suite", default="benchmarks/suite.json")
    s.add_argument("--output", required=True)
    s.add_argument("--editor", choices=["none", "random-search", "agent", "openrouter", "ollama"], default="none",
                   help="none = baselines only")
    s.add_argument("--seeds", type=int, nargs="+", help="Override the suite's seeds")
    s.add_argument("--iterations", type=int, help="Override each task's iteration budget")
    s.add_argument("--names", choices=["semantic", "anonymized"], default="semantic")
    s.add_argument("--policy", choices=["strict", "open"], default="strict")
    s.add_argument("--ollama-model")
    s.add_argument("--ollama-url", default="http://localhost:11434")
    add_agent_arguments(s, required=False)

    bl = sub.add_parser("baselines", help="Fit trained reference models on the protocol split of one task")
    add_task_arguments(bl)
    bl.add_argument("--seed", type=int)

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
            track = Track(args.names, args.policy)
            table, options, _ = load_task(args.config, args.dataset, track)
            if args.command == "agent":
                editor = make_agent(args, args.output)
                try:
                    result = build(table=table, editor=editor, output=args.output,
                                   dynamic_architecture=True, isolated_evaluation=True, **options)
                finally:
                    editor.close()
            else:
                if args.ollama_model:
                    editor = OllamaEditor(args.ollama_model, args.ollama_url)
                elif args.openrouter_model:
                    editor = OpenRouterEditor(args.openrouter_model, args.openrouter_api_key_env)
                elif args.random_search is not None:
                    editor = RandomSearchEditor(args.random_search)
                else:
                    editor = ScriptedEditor(read_json(args.programs))
                result = build(table=table, editor=editor, output=args.output, **options)
        elif args.command == "bench":
            bench(args)
            return
        elif args.command == "baselines":
            from .baselines import run_baselines
            table, options, _ = load_task(args.config, args.dataset, Track(args.names, args.policy))
            results = run_baselines(table, options["seed"] if args.seed is None else args.seed)
            metric = table.goal.primary_metric
            print(json.dumps({name: {"final_test_" + metric: r["final_test"][metric],
                                     "hyperparameters": r["hyperparameters"], "size": r["size"]}
                              for name, r in results.items()}, indent=2))
            return
        elif args.command == "replay":
            print(json.dumps({"verified_state_hash": replay(args.run).fingerprint()}))
            return
        else:
            predict(args)
            return
        print(json.dumps({"run": str(result.path), "status": result.report["status"],
                          "selected_iteration": result.report["selected_iteration"],
                          "final_test": result.report["final_test"]}, indent=2))
        if result.model is None or result.report["status"] == "final_evaluation_error":
            raise SystemExit(2)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


def bench(args):
    from .suite import run_suite
    track = Track(args.names, args.policy)
    options = {}
    if args.iterations:
        options["budget_overrides"] = {"iterations": args.iterations}
    if args.editor == "none":
        make = None
    elif args.editor == "random-search":
        make = lambda task, seed, out: RandomSearchEditor(seed)
    elif args.editor == "openrouter":
        if not args.model:
            raise ValueError("--model is required for the openrouter editor")
        make = lambda task, seed, out: OpenRouterEditor(args.model, args.api_key_env)
    elif args.editor == "ollama":
        if not args.ollama_model:
            raise ValueError("--ollama-model is required for the ollama editor")
        make = lambda task, seed, out: OllamaEditor(args.ollama_model, args.ollama_url)
    else:
        if not args.model:
            raise ValueError("--model is required for the agent editor")
        # Fail on a missing key, SDK, or Docker image before any output or baseline work.
        make_agent(args, args.output)
        make = lambda task, seed, out: make_agent(args, out)
        options.update(dynamic_architecture=True, isolated_evaluation=True)
    run_suite(args.suite, make, args.output, seeds=args.seeds, track=track, editor_options=options)
    print(Path(args.output, "results.md").read_text())


def predict(args):
    predictor = load(args.run)
    with Path(args.dataset).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    with Path(args.output).open("x", newline="") as handle:
        writer = csv.writer(handle)
        if predictor.task in {"binary", "multiclass"}:
            probabilities = predictor.predict_proba(rows)
            labels = predictor.predict(rows)
            writer.writerow(["prediction", *[f"probability_{c}" for c in predictor.classes]]
                            if predictor.task == "multiclass" else ["prediction", "negative_probability", "positive_probability"])
            writer.writerows((label, *map(float, pair)) for label, pair in zip(labels, probabilities))
        else:
            writer.writerow(["prediction" if predictor.task == "regression" else "cluster"])
            writer.writerows([v] for v in predictor.predict(rows).tolist())
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
