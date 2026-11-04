"""Container-only execution of agent-authored code, with a persistent agent workspace.

Mounts:  /inputs (read-only: task brief, support data, current state)
         /skills (read-only: weight-space programming knowledge, no code)
         /workspace (read-write, persists across executions: the agent's own tools)
         /exec (read-only: the source being executed)
No network, no credentials, no development or final-test rows.
"""
import os
import subprocess
import uuid
from pathlib import Path

LANGUAGES = {"python": ("source.py", "python", ["-B", "/exec/source.py"]),
             "shell": ("source.sh", "sh", ["/exec/source.sh"])}


class DockerSandbox:
    def __init__(self, image="mlsurgie-constructor:local"):
        if not image or image.startswith("-"):
            raise ValueError("A local construction image is required")
        self.image = image

    def check(self):
        try:
            subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=True)
            subprocess.run(["docker", "image", "inspect", self.image],
                           capture_output=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            raise RuntimeError("Start Docker and build the construction image; see docs/agent-harness.md") from None

    def command(self, name, language, inputs, skills, workspace, record):
        filename, entrypoint, arguments = LANGUAGES[language]
        command = ["docker", "run", "--rm", "--pull", "never", "--name", name,
                   "--network", "none", "--read-only", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--pids-limit", "128",
                   "--memory", "2g", "--cpus", "2", "--tmpfs", "/tmp:rw,size=256m",
                   "--env", "HOME=/tmp", "--env", "MPLCONFIGDIR=/tmp", "--env", "PYTHONPATH=/workspace"]
        if hasattr(os, "getuid"):  # POSIX hosts: never run agent code as root
            command += ["--user", f"{os.getuid()}:{os.getgid()}"]
        for source, target, mode in ((inputs, "/inputs", ",readonly"), (skills, "/skills", ",readonly"),
                                     (record, "/exec", ",readonly"), (workspace, "/workspace", "")):
            command += ["--mount", f"type=bind,src={source},dst={target}{mode}"]
        return command + ["--workdir", "/workspace", "--entrypoint", entrypoint, self.image, *arguments]

    def execute(self, source, *, inputs, skills, workspace, record, timeout, language="python"):
        """Run source; returns (exit code, combined output). Raises ValueError on timeout."""
        if language not in LANGUAGES:
            raise ValueError("language must be python or shell")
        record = Path(record).resolve()
        record.mkdir(parents=True)
        (record / LANGUAGES[language][0]).write_text(source)
        name = "mlsurgie-" + uuid.uuid4().hex
        command = self.command(name, language, *(Path(p).resolve() for p in (inputs, skills, workspace)), record)
        log_path = record / "execution.log"
        try:
            with log_path.open("wb") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            return result.returncode, log_path.read_bytes()[-12_000:].decode(errors="replace")
        except subprocess.TimeoutExpired:
            raise ValueError("Execution exceeded its deadline") from None
        finally:
            # --rm alone does not remove a container when its client is killed.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
