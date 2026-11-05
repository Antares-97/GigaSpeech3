"""Score raw hypotheses with a pinned GigaSpeechBench checkout, without torch/k2."""

import argparse
import importlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from common import DIALECTS, read_jsonl, sha256


def word_errors(reference, hypothesis):
    """Exact Levenshtein alignment. Cells contain (errors, substitutions, deletions, insertions)."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                current.append(previous[j - 1])
                continue
            cost, sub, delete, insert = previous[j - 1]
            substitution = (cost + 1, sub + 1, delete, insert)
            cost, sub, delete, insert = previous[j]
            deletion = (cost + 1, sub, delete + 1, insert)
            cost, sub, delete, insert = current[j - 1]
            insertion = (cost + 1, sub, delete, insert + 1)
            current.append(min((substitution, deletion, insertion), key=lambda c: c[0]))
        previous = current
    return previous[-1]


def load_normalizers(checkout, revision):
    checkout = Path(checkout).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise ValueError(f"Benchmark revision mismatch: {actual} != {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--", "text_norm"], text=True
    ).strip()
    if dirty:
        raise ValueError("Benchmark text_norm has uncommitted changes")
    if "text_norm" in sys.modules:
        raise RuntimeError("Run scoring in a fresh process to avoid another text_norm package")
    sys.path.insert(0, str(checkout))
    functions = {}
    for code in DIALECTS:
        # Import the concrete module so upstream's fallback cannot silently hide missing dependencies.
        module = importlib.import_module(f"text_norm.{code}")
        if not Path(module.__file__).resolve().is_relative_to(checkout):
            raise RuntimeError(f"Wrong normalizer package loaded for {code}")
        functions[code] = module.normalize
    hashes = {str(p.relative_to(checkout)): sha256(p)
              for p in sorted((checkout / "text_norm").rglob("*.py"))}
    return functions, {"revision": actual, "normalizer_sha256": hashes}


def score(references, hypotheses, normalizers, min_duration=0.5):
    refs = {}
    for row in references:
        if row["id"] in refs:
            raise ValueError(f"Duplicate reference ID: {row['id']}")
        if row["dialect"] not in DIALECTS:
            raise ValueError(f"Unknown reference dialect: {row['dialect']}")
        if "text" not in row or "duration" not in row:
            raise ValueError("Reference needs text and duration")
        refs[row["id"]] = row
    hyps = {}
    for row in hypotheses:
        if row["id"] in hyps:
            raise ValueError(f"Duplicate hypothesis ID: {row['id']}")
        if not isinstance(row.get("raw_text"), str):
            raise ValueError("Hypothesis needs string field raw_text (pre-textnorm output)")
        hyps[row["id"]] = row["raw_text"]
    missing, extra = refs.keys() - hyps.keys(), hyps.keys() - refs.keys()
    if missing or extra:
        raise ValueError(f"ID coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    totals = defaultdict(lambda: {"segments": 0, "reference_words": 0,
                                  "substitutions": 0, "deletions": 0, "insertions": 0})
    skipped_duration = skipped_empty = 0
    for uid, row in refs.items():
        if float(row["duration"]) <= min_duration:
            skipped_duration += 1
            continue
        norm = normalizers[row["dialect"]]
        ref = norm(row["text"]).split()
        hyp = norm(hyps[uid]).split()
        if not ref:
            skipped_empty += 1
            continue
        _, sub, delete, insert = word_errors(ref, hyp)
        target = totals[row["dialect"]]
        for key, value in zip(target, [1, len(ref), sub, delete, insert]):
            target[key] += value
    if not totals:
        raise ValueError("No scoreable references")
    pooled_words = pooled_errors = 0
    for target in totals.values():
        errors = sum(target[k] for k in ("substitutions", "deletions", "insertions"))
        target["wer_percent"] = errors * 100 / target["reference_words"]
        pooled_words += target["reference_words"]
        pooled_errors += errors
    return {
        "per_dialect": dict(totals),
        "macro_wer_percent": sum(t["wer_percent"] for t in totals.values()) / len(totals),
        "pooled_wer_percent": pooled_errors * 100 / pooled_words,
        "covered_dialects": sorted(totals),
        "is_five_dialect_macro": set(totals) == set(DIALECTS),
        "input_segments": len(refs), "skipped_duration": skipped_duration,
        "skipped_empty_normalized_reference": skipped_empty,
        "min_duration_exclusive": min_duration,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--benchmark-revision", required=True)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    norms, provenance = load_normalizers(args.benchmark_dir, args.benchmark_revision)
    result = score(read_jsonl(args.references), read_jsonl(args.hypotheses), norms, args.min_duration)
    result.update(provenance)
    result["reference_sha256"] = sha256(args.references)
    result["hypothesis_sha256"] = sha256(args.hypotheses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
