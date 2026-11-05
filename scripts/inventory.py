"""Write an explicit release inventory of local Python dependencies (no data or credentials)."""

import ast
import json
from pathlib import Path

from common import sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ASR = ROOT / "asr"
SEARCH = [ASR, ASR / "zipformer", ROOT, SCRIPTS]


def module_file(path):
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    return None


def closure(seeds):
    pending = list(seeds)
    visited = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            requests = []
            if isinstance(node, ast.Import):
                requests = [(SEARCH, alias.name) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                bases = SEARCH
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    bases = [base]
                name = node.module or ""
                requests.append((bases, name))
                requests.extend((bases, ".".join(filter(None, [name, alias.name]))) for alias in node.names)
            for bases, name in requests:
                if not name:
                    continue
                for base in bases:
                    target = module_file(base.joinpath(*name.split(".")))
                    if target and target.resolve().is_relative_to(ROOT):
                        pending.append(target)
                        parent = target.parent
                        while parent != ROOT and parent.is_relative_to(ROOT):
                            if (parent / "__init__.py").is_file():
                                pending.append(parent / "__init__.py")
                            parent = parent.parent
                        break
    return visited


def main():
    core_seeds = [ASR / name for name in (
        "train.py", "decode.py", "pipeline_decode.py")]
    core_seeds += [p for p in SCRIPTS.glob("*.py") if not p.name.startswith("test_")]
    core_seeds += [ASR / "textnorm" / f"{code}.py" for code in ("DZA", "EGY", "IRQ", "SAU", "SYR")]
    core = closure(core_seeds)

    def entries(paths):
        return [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)}
                for p in sorted(paths)]

    report = {
        "description": "Static local import closure; not an installation lockfile or a secret scan.",
        "core_pipeline": entries(core),
        "documentation_and_config": [
            "README.md", "INSTALL.md", "NOTICE.md", "requirements.txt",
            "requirements-data.txt", "configs/dialect-specific.example.json",
            "configs/sau.example.json",
        ],
        "excluded": ["model weights", "audio", "Fbank caches", "experiment logs",
                     "GigaSpeechBench checkout", "gigaspeech2_method API clients"],
    }
    target = ROOT / "release_files.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Release inventory: {len(core)} Python files")


if __name__ == "__main__":
    main()
