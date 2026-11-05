"""Convert native GigaSpeechBench audios/segments metadata to human-reference JSONL."""

import argparse
import json
import math
from pathlib import Path, PurePosixPath

from common import DIALECTS, sha256
from prepare import extract_selected


def reference_rows(metadata, audio_paths, dialect):
    seen = set()
    for audio in metadata["audios"]:
        aid = audio["aid"]
        if aid not in audio_paths:
            raise FileNotFoundError(f"Benchmark recording absent from tar: {aid}")
        for segment in audio["segments"]:
            if segment.get("status") == "invalid":
                continue
            uid = segment["sid"]
            if uid in seen:
                raise ValueError(f"Duplicate benchmark segment: {uid}")
            seen.add(uid)
            begin, end = float(segment["begin_time"]), float(segment["end_time"])
            if not math.isfinite(begin) or not math.isfinite(end) or begin < 0 or end <= begin:
                raise ValueError(f"Invalid benchmark time bounds: {uid}")
            if not isinstance(segment["text"], str):
                raise ValueError(f"Invalid benchmark reference text: {uid}")
            yield {"id": uid, "audio_path": str(audio_paths[aid]), "start": begin,
                   "duration": end - begin, "text": segment["text"], "dialect": dialect,
                   "label_type": "human", "audio_id": aid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialect", choices=DIALECTS, required=True)
    parser.add_argument("--repo-id", default="speechcolab/GigaSpeechBench")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dataset", type=Path)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = args.revision
    if args.local_dataset is None:
        from huggingface_hub import HfApi, hf_hub_download
        revision = HfApi().dataset_info(args.repo_id, revision=revision).sha

    prefix = f"Low-Resource-Languages/data/{args.dialect}"
    def obtain(name):
        path = f"{prefix}/{name}"
        if args.local_dataset:
            return args.local_dataset.resolve() / path
        return Path(hf_hub_download(args.repo_id, path, repo_type="dataset", revision=revision))

    metadata_path = obtain("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    aids = {audio["aid"] for audio in metadata["audios"]}
    if len(aids) != len(metadata["audios"]):
        raise ValueError("Duplicate benchmark audio IDs")
    destination = args.audio_dir.resolve() / args.dialect
    seen = extract_selected(obtain("audio.tar.gz"), destination,
                            lambda name: PurePosixPath(name).stem in aids and name.lower().endswith(".wav"))
    audio_paths = {}
    for member in seen:
        aid = PurePosixPath(member).stem
        if aid in audio_paths:
            raise ValueError(f"Ambiguous benchmark recording path: {aid}")
        audio_paths[aid] = destination / member
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    count = 0
    with partial.open("w", encoding="utf-8") as out:
        for row in reference_rows(metadata, audio_paths, args.dialect):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    if not count:
        raise ValueError("Benchmark metadata contains no valid segments")
    partial.replace(args.output)
    args.output.with_suffix(".provenance.json").write_text(json.dumps({
        "repo_id": args.repo_id, "revision": revision, "metadata_sha256": sha256(metadata_path),
        "output_sha256": sha256(args.output), "segments": count,
        "policy": "preserve original sid, begin/end, and raw Arabic reference text; skip status=invalid",
    }, indent=2), encoding="utf-8")
    print(f"Imported {count} human-reference test segments at revision {revision}")


if __name__ == "__main__":
    main()
