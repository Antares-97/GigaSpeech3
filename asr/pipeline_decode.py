"""Raw and normalized transcript export for independent scoring."""

import json


def add_pipeline_arguments(parser):
    parser.add_argument(
        "--raw-transcripts", action="store_true",
        help="Retained for compatibility; decoded raw text is always exported."
    )
    parser.add_argument(
        "--keep-eval-cuts", action="store_true",
        help="Decode all cuts in the explicitly supplied evaluation manifest; do not filter by text or duration."
    )


def export_raw_hypotheses(params, test_set_name, key, results, normalized_results=None):
    if not (getattr(params, "raw_transcripts", False)
            or getattr(params, "keep_raw_text", False)):
        return
    normalized_source = results if normalized_results is None else normalized_results
    normalized = {cut_id: words for cut_id, _reference, words in normalized_source}
    path = params.res_dir / f"hyp-{test_set_name}-{key}-{params.suffix}.jsonl"
    with path.open("w", encoding="utf-8") as out:
        for cut_id, _reference_words, hyp_words in results:
            row = {
                "id": cut_id,
                "raw_text": " ".join(hyp_words),
                "text": " ".join(normalized[cut_id]),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
