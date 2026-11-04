"""Container-only execution of agent-authored construction code."""
import json
import os
from pathlib import Path
import subprocess
import uuid


class DockerSandbox:
    def __init__(self, image='mlsurgie-constructor:local'):
        if not image or image.startswith('-'):
            raise ValueError('A local construction image is required')
        self.image = image

    def check(self):
        try:
            subprocess.run(['docker', 'info'], capture_output=True, timeout=10, check=True)
            subprocess.run(['docker', 'image', 'inspect', self.image],
                           capture_output=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            raise RuntimeError('Start Docker and build the construction image; see docs/agent-harness.md') from None

    def execute(self, source, context, directory, timeout):
        directory = Path(directory).resolve()
        inputs, work = directory / 'inputs', directory / 'work'
        inputs.mkdir(parents=True)
        work.mkdir()
        (inputs / 'source.py').write_text(source)
        (inputs / 'task.json').write_text(json.dumps(context, allow_nan=False))
        name = 'mlsurgie-' + uuid.uuid4().hex
        command = ['docker', 'run', '--rm', '--pull', 'never', '--name', name,
                   '--network', 'none', '--read-only', '--cap-drop', 'ALL',
                   '--security-opt', 'no-new-privileges', '--pids-limit', '128',
                   '--memory', '1g', '--cpus', '1', '--tmpfs', '/tmp:rw,size=64m',
                   '--user', f'{os.getuid()}:{os.getgid()}',
                   '--mount', f'type=bind,src={inputs},dst=/inputs,readonly',
                   '--mount', f'type=bind,src={work},dst=/work', '--workdir', '/work',
                   '--entrypoint', 'python', self.image, '-B', '/inputs/source.py']
        try:
            with (directory / 'execution.log').open('wb') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            if result.returncode:
                raise ValueError('Construction code failed: ' + (directory / 'execution.log').read_bytes()[-8192:].decode(errors='replace'))
            path = work / 'candidate.json'
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
                raise ValueError('Write a regular candidate.json file smaller than 2 MB')
            return json.loads(path.read_text())
        except subprocess.TimeoutExpired:
            raise ValueError('Construction code exceeded its execution deadline') from None
        finally:
            # --rm alone does not remove a container when its client is killed.
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=10)
