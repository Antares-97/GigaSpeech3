import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

DIALECTS = ("DZA", "EGY", "IRQ", "SAU", "SYR")


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(value):
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError(f"Unsafe archive path: {value!r}")
    return path


def shard_path(dialect, shard):
    path = safe_relative(shard)
    if len(path.parts) == 1:
        return f"{dialect}/audio/{path}"
    if str(path).startswith("audio/"):
        return f"{dialect}/{path}"
    if str(path).startswith(f"{dialect}/audio/"):
        return str(path)
    raise ValueError(f"Unexpected shard reference: {shard}")
