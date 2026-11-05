"""Prepare the released training corpus, with an optional reference converter.

``hf`` is the training-data stage used by ``run.py``. ``references`` is an
optional evaluation utility for converting human-reference JSONL to Lhotse
cuts; it is not part of the training-data pipeline.
"""

import argparse
import json
import math
import shutil
import sqlite3
import tarfile
import tempfile
from pathlib import Path

from common import DIALECTS, read_jsonl, safe_relative, sha256, shard_path


def provenance_path(output):
    """Uniform provenance filename next to a .jsonl.gz manifest."""
    name = Path(output).name
    if name.endswith(".jsonl.gz"):
        return Path(output).with_name(name[:-len(".jsonl.gz")] + ".provenance.json")
    return Path(output).with_suffix(".provenance.json")


def extract_selected(archive, destination, members):
    """Stream a shard once; copy only indexed regular members, without tar.extractall."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tarfile.open(archive, "r|*") as stream:
        for item in stream:
            name = str(safe_relative(item.name))
            if not members(name):
                continue
            if name in seen or not item.isfile():
                raise ValueError(f"Duplicate or non-regular selected audio member: {name}")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive path escapes destination: {name}")
            seen.add(name)
            if target.exists():
                if target.stat().st_size != item.size:
                    raise ValueError(f"Existing audio has wrong size: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with stream.extractfile(item) as source, partial.open("wb") as out:
                shutil.copyfileobj(source, out)
            if partial.stat().st_size != item.size:
                raise ValueError(f"Incomplete audio member: {name}")
            partial.replace(target)
    return seen


def make_cut(uid, audio, text, dialect, start=0.0, duration=None, custom=None):
    from lhotse import MonoCut, Recording, SupervisionSegment

    audio = Path(audio).resolve()
    recording = Recording.from_file(audio, recording_id=str(audio))
    if recording.num_channels != 1:
        raise ValueError(f"Expected mono audio: {audio}")
    start = float(start)
    duration = recording.duration - start if duration is None else float(duration)
    if not all(math.isfinite(v) for v in [start, duration]) or start < 0 or duration <= 0:
        raise ValueError(f"Invalid segment bounds: {uid}")
    if start + duration > recording.duration + 0.03:
        raise ValueError(f"Segment exceeds recording: {uid}")
    # Virtual resampling is stored in the manifest; no corpus-wide WAV rewrite.
    recording = recording.resample(16000)
    code = dialect.upper()
    # Cut.start is relative to the recording; supervision.start is relative to
    # the cut. The supervision covers the whole cut, so its offset is always 0.
    supervision = SupervisionSegment(
        id=uid, recording_id=recording.id, start=0.0, duration=duration,
        channel=0, text=text,
    )
    return MonoCut(id=uid, start=start, duration=duration, channel=0,
                   recording=recording, supervisions=[supervision],
                   custom={"dialect": code, **(custom or {})})


def prepare_hf(args):
    import pyarrow.parquet as pq
    from lhotse import CutSet

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Choose a new output or explicitly remove the old manifest: {args.output}")
    local = args.local_dataset.resolve() if args.local_dataset else None
    revision = args.revision
    if local is None:
        from huggingface_hub import HfApi, hf_hub_download
        revision = HfApi().dataset_info(args.repo_id, revision=revision).sha

    def obtain(path):
        if local:
            result = local / safe_relative(path)
            if not result.is_file():
                raise FileNotFoundError(result)
            return result
        return Path(hf_hub_download(args.repo_id, path, repo_type="dataset", revision=revision))

    report = {"repo_id": args.repo_id, "revision": revision, "local_dataset": str(local) if local else None,
              "max_shards_per_dialect": args.max_shards, "dialects": {}, "sampling_rate": 16000}
    partial = args.output.with_name(args.output.stem + ".partial.jsonl.gz")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        db = sqlite3.connect(str(Path(tmp) / "index.sqlite3"))
        db.execute("CREATE TABLE samples (id TEXT PRIMARY KEY, dialect TEXT, shard TEXT, member TEXT, duration REAL, text TEXT, UNIQUE(dialect, shard, member))")
        db.execute("CREATE INDEX member_lookup ON samples(dialect, shard, member)")
        for code in args.dialects:
            metadata = obtain(f"{code}/metadata/{code}.parquet")
            pf = pq.ParquetFile(metadata)
            required = {"id", "duration", "text", "dialect", "shard", "member"}
            if not required.issubset(pf.schema_arrow.names):
                raise ValueError(f"Metadata schema mismatch: {metadata}")
            for batch in pf.iter_batches(batch_size=8192, columns=sorted(required)):
                values = []
                for row in batch.to_pylist():
                    if row["dialect"] != code or not isinstance(row["id"], str) or not row["id"]:
                        raise ValueError(f"Invalid ID or dialect in {metadata}")
                    if not isinstance(row["text"], str) or not row["text"].strip():
                        raise ValueError(f"Empty or invalid transcript: {row['id']}")
                    duration = float(row["duration"])
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError(f"Invalid duration: {row['id']}")
                    values.append((row["id"], code, shard_path(code, row["shard"]),
                                   str(safe_relative(row["member"])), duration, row["text"]))
                db.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?)", values)
            db.commit()
            report["dialects"][code] = {"metadata_sha256": sha256(metadata), "rows": 0, "hours": 0.0, "shards": []}
        with CutSet.open_writer(partial) as writer:
            for code in args.dialects:
                shards = [r[0] for r in db.execute("SELECT DISTINCT shard FROM samples WHERE dialect=? ORDER BY shard", (code,))]
                if args.max_shards:
                    shards = shards[:args.max_shards]
                for shard in shards:
                    print(f"Preparing {shard}", flush=True)
                    destination = args.audio_dir.resolve() / Path(shard).stem
                    def selected(name):
                        return db.execute("SELECT 1 FROM samples WHERE dialect=? AND shard=? AND member=?", (code, shard, name)).fetchone() is not None
                    seen = extract_selected(obtain(shard), destination, selected)
                    records = db.execute("SELECT id, member, duration, text FROM samples WHERE dialect=? AND shard=? ORDER BY id", (code, shard))
                    for uid, member, duration, text in records:
                        if member not in seen:
                            raise FileNotFoundError(f"Indexed member absent from archive: {shard}:{member}")
                        cut = make_cut(uid, destination / member, text, code,
                                       custom={"hf_repo": args.repo_id, "hf_revision": revision,
                                               "shard": shard, "member": member, "label_type": "weak"})
                        if abs(cut.recording.duration - duration) > 0.05:
                            raise ValueError(
                                f"Metadata/audio duration mismatch: {uid} "
                                f"(metadata={duration:.3f}s, audio={cut.recording.duration:.3f}s)")
                        writer.write(cut)
                        stats = report["dialects"][code]
                        stats["rows"] += 1
                        stats["hours"] += cut.duration / 3600
                    report["dialects"][code]["shards"].append(shard)
        db.close()
    partial.replace(args.output)
    provenance_path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


def prepare_references(args):
    from lhotse import CutSet

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ids = set()
    partial = args.output.with_name(args.output.stem + ".partial.jsonl.gz")
    with CutSet.open_writer(partial) as writer:
        for row in read_jsonl(args.input):
            uid = row["id"]
            if uid in ids:
                raise ValueError(f"Duplicate reference ID: {uid}")
            ids.add(uid)
            if row["dialect"] not in args.dialects:
                raise ValueError(f"Reference outside selected dialects: {uid}")
            if row.get("label_type") != "human":
                raise ValueError(f"Reference must explicitly declare label_type=human: {uid}")
            if not isinstance(row["text"], str):
                raise ValueError(f"Invalid reference text: {uid}")
            audio = Path(row["audio_path"])
            if not audio.is_absolute():
                audio = args.input.resolve().parent / audio
            cut = make_cut(uid, audio, row["text"], row["dialect"],
                           start=row.get("start", 0), duration=row["duration"],
                           custom={"label_type": "human", "source_reference": str(args.input.resolve())})
            writer.write(cut)
    if not ids:
        raise ValueError("Reference manifest is empty")
    partial.replace(args.output)
    provenance_path(args.output).write_text(json.dumps({
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "references": len(ids),
        "dialects": sorted({row["dialect"] for row in read_jsonl(args.input)}),
        "label_type": "human",
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    hf = sub.add_parser("hf")
    hf.add_argument("--repo-id", default="speechcolab/gigaspeech3")
    hf.add_argument("--revision", default="main")
    hf.add_argument("--local-dataset", type=Path)
    hf.add_argument("--audio-dir", type=Path, required=True)
    hf.add_argument("--max-shards", type=int, default=0)
    hf.set_defaults(func=prepare_hf)
    refs = sub.add_parser("references")
    refs.add_argument("--input", type=Path, required=True)
    refs.set_defaults(func=prepare_references)
    for target in (hf, refs):
        target.add_argument("--dialects", nargs="+", choices=DIALECTS, required=True)
        target.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
