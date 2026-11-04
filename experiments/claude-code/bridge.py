"""File bridge: an interactive agent (here, Claude Code) drives the standard agent harness.

The protocol is unchanged: support data as files, persistent workspace, Docker sandbox,
fitting audit, independent development evaluation, sealed final test. Only the "brain"
differs: instead of an OpenHands LLM loop, the harness waits for code dropped into a
bridge directory and returns the execution output there.

Server (background):  python experiments/claude-code/bridge.py serve OUTPUT
Client:               python experiments/claude-code/bridge.py exec OUTPUT FILE.py|FILE.sh
                      python experiments/claude-code/bridge.py end OUTPUT      # submit, wait for feedback
                      python experiments/claude-code/bridge.py finish OUTPUT   # end the task's experiment
                      python experiments/claude-code/bridge.py status OUTPUT

The client only ever shows what the sandboxed agent sees: execution output and the
harness brief. Development and test rows are never printed.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SUITE = ["fraud", "breast_cancer", "wine", "diabetes", "segments", "segments_repair"]
POLL = 0.5


def bridge_root(output):
    return Path(output).resolve() / "bridge"


def serve(output):
    from mlsurgie.harness import OpenHandsEditor
    from mlsurgie.suite import run_suite

    os.environ.setdefault("MLSURGIE_BRIDGE_KEY", "interactive-session-no-provider")
    root = bridge_root(output)

    def runtime(editor, context, timeout):
        turn = root / editor.task / f"turn-{context['iteration']:03d}"
        turn.mkdir(parents=True)
        (turn / "brief.txt").write_text(editor._brief(context))
        (root / "current").write_text(str(turn))
        deadline = time.monotonic() + timeout
        n = 0
        while time.monotonic() < deadline:
            for marker in ("END", "FINISH"):
                if (turn / marker).exists():
                    if marker == "FINISH":
                        editor.stop_requested = True
                    return {"input_tokens": 0, "output_tokens": 0,
                            "measurement": "unavailable: interactive Claude Code session"}
            for language, suffix in (("python", ".py"), ("shell", ".sh")):
                request = turn / f"exec-{n + 1:03d}{suffix}"
                if request.exists():
                    n += 1
                    result = editor.execute(request.read_text(), language)
                    (turn / f"exec-{n:03d}.out").write_text(result)
            time.sleep(POLL)
        raise TimeoutError("Interactive turn exceeded its hang guard")

    def make_editor(task, seed, run_directory):
        editor = OpenHandsEditor("anthropic/claude-opus-5-5 (Claude Code, interactive)", run_directory,
                                 "MLSURGIE_BRIDGE_KEY", runtime=runtime, max_agent_steps=60)
        editor.task = task
        editor.identity.update(kind="interactive_llm", model="claude-opus-5-5",
                               interface="Claude Code session via file bridge; Docker sandbox; strict audit",
                               token_accounting="not available from the host session")
        editor.check()
        return editor

    suite = root.parent.parent / f"{Path(output).name}-suite.json"
    suite.write_text(json.dumps({"name": "claude-code-interactive", "seeds": [0],
                                 "tasks": [str(ROOT / "benchmarks" / "tasks" / f"{t}.json") for t in SUITE]}))
    run_suite(suite, make_editor, output, editor_options={"dynamic_architecture": True, "isolated_evaluation": True})
    (bridge_root(output) / "DONE").write_text("suite complete")


def current(output):
    root = bridge_root(output)
    while not (root / "current").exists():
        time.sleep(POLL)
    return Path((root / "current").read_text())


def wait_for_next(output, previous):
    root = bridge_root(output)
    while True:
        if (root / "DONE").exists():
            return None
        if (root / "current").exists():
            turn = Path((root / "current").read_text())
            if turn != previous and (turn / "brief.txt").exists():
                return turn
        time.sleep(POLL)


def client(command, output, path=None):
    turn = current(output)
    if command == "status":
        print(f"[{turn.parent.name} / {turn.name}]\n" + (turn / "brief.txt").read_text())
        return
    if command == "exec":
        path = Path(path)
        n = len(list(turn.glob("exec-*.out"))) + 1
        target = turn / f"exec-{n:03d}{'.sh' if path.suffix == '.sh' else '.py'}"
        target.write_text(path.read_text())
        out = turn / f"exec-{n:03d}.out"
        while not out.exists():
            time.sleep(POLL)
        print(out.read_text())
        return
    (turn / ("FINISH" if command == "finish" else "END")).write_text("")
    nxt = wait_for_next(output, turn)
    if nxt is None:
        print("Suite complete. Final results: results.md")
    else:
        print(f"[{nxt.parent.name} / {nxt.name}]\n" + (nxt / "brief.txt").read_text())


if __name__ == "__main__":
    command, output, *rest = sys.argv[1:]
    serve(output) if command == "serve" else client(command, output, *rest)
