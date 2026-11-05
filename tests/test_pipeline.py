"""CPU/offline contract tests; no model training or external services."""

import argparse
import ast
import io
import json
import logging
import random
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ASR = ROOT / "asr"
for _path in (SCRIPTS, ASR / "zipformer", ASR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import DIALECTS, shard_path
from benchmark import reference_rows
from build_tokenizer import build_tokenizer, write_training_text
from prepare import extract_selected, make_cut
from features import compute_features
from recipe_textnorm import get_recipe_normalizer_with_word_level
from score import score, word_errors
from run import load_config
from pipeline_decode import add_pipeline_arguments, export_raw_hypotheses


class PipelineTests(unittest.TestCase):
    def test_manifest_conversion_does_not_require_language_or_scores(self):
        record = types.SimpleNamespace(num_channels=1, duration=2.0, id="audio")
        record.resample = lambda _: record
        fake = types.ModuleType("lhotse")
        fake.Recording = types.SimpleNamespace(from_file=lambda *a, **k: record)
        fake.SupervisionSegment = lambda **kw: types.SimpleNamespace(**kw)
        fake.MonoCut = lambda **kw: types.SimpleNamespace(**kw)
        with patch.dict(sys.modules, {"lhotse": fake}):
            cut = make_cut("sample", "audio.wav", "text", "DZA")
        self.assertFalse(hasattr(cut.supervisions[0], "language"))
        self.assertEqual(cut.custom, {"dialect": "DZA"})
        self.assertEqual(cut.supervisions[0].text, "text")

    def test_reference_cut_preserves_recording_offset(self):
        record = types.SimpleNamespace(num_channels=1, duration=10.0, id="audio")
        record.resample = lambda _: record
        fake = types.ModuleType("lhotse")
        fake.Recording = types.SimpleNamespace(from_file=lambda *a, **k: record)
        fake.SupervisionSegment = lambda **kw: types.SimpleNamespace(**kw)
        fake.MonoCut = lambda **kw: types.SimpleNamespace(**kw)
        with patch.dict(sys.modules, {"lhotse": fake}):
            cut = make_cut("sample", "audio.wav", "text", "DZA", start=2.0, duration=3.0)
        self.assertEqual(cut.start, 2.0)
        self.assertEqual(cut.duration, 3.0)
        # Supervision bounds are relative to the cut, not the recording.
        self.assertEqual(cut.supervisions[0].start, 0.0)
        self.assertEqual(cut.supervisions[0].duration, 3.0)

    def test_reference_cut_uses_real_lhotse_timeline(self):
        try:
            import lhotse  # noqa: F401
            import numpy as np
            import soundfile
        except ImportError:
            self.skipTest("lhotse/soundfile/numpy not installed")
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "rec.wav"
            soundfile.write(audio, np.zeros(16000 * 5, dtype="float32"), 16000)
            cut = make_cut("sample", audio, "text", "DZA", start=2.0, duration=3.0)
            self.assertEqual(cut.start, 2.0)
            self.assertEqual(cut.duration, 3.0)
            self.assertEqual(cut.supervisions[0].start, 0.0)
            self.assertEqual(cut.supervisions[0].duration, 3.0)
            self.assertAlmostEqual(cut.load_audio().shape[-1] / 16000, 3.0, places=2)

    def test_missing_raw_cuts_fail_before_feature_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                compute_features("DZA", "train", Path(tmp) / "missing", Path(tmp) / "out")

    def test_decode_loads_current_textnorm_modules(self):
        for code in DIALECTS:
            normalizer, _ = get_recipe_normalizer_with_word_level(code)
            self.assertIsNotNone(normalizer, code)
            self.assertEqual(
                Path(normalizer.__code__.co_filename).resolve(),
                (ASR / "textnorm" / f"{code}.py").resolve(),
            )

    def test_explicit_cuts_path_has_priority_over_named_split(self):
        try:
            import decode
        except ImportError:
            self.skipTest("decoding dependencies are not installed")
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom.jsonl.gz"
            custom.write_bytes(b"")
            args = types.SimpleNamespace(
                cuts_name="test", cuts_path=str(custom), manifest_dir=Path(tmp)
            )
            class DataModule:
                def test_cuts(self):
                    raise AssertionError("named split must not be consulted")
                def dev_cuts(self):
                    raise AssertionError("named split must not be consulted")
            with patch("decode.lhotse.load_manifest_lazy", return_value="CUSTOM") as loader:
                result = decode.resolve_test_sets(args, DataModule())
            self.assertEqual(result, {"test": "CUSTOM"})
            loader.assert_called_once_with(custom)
            missing = types.SimpleNamespace(
                cuts_name="test", cuts_path=str(Path(tmp) / "missing.jsonl.gz"),
                manifest_dir=Path(tmp),
            )
            with self.assertRaises(FileNotFoundError):
                decode.resolve_test_sets(missing, DataModule())

    def test_missing_dev_cuts_fail_without_train_fallback(self):
        try:
            from datamodule import ArabicAsrDataModule
        except ImportError:
            self.skipTest("training dependencies are not installed")
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(manifest_dir=Path(tmp), dialect="DZA")
            module = ArabicAsrDataModule(args)
            with self.assertRaises(FileNotFoundError) as context:
                module.dev_cuts()
            self.assertIn("Independent dev cuts", str(context.exception))
            with self.assertRaises(FileNotFoundError):
                module.test_cuts()

    def test_release_root_has_no_legacy_python_entrypoints(self):
        self.assertEqual(sorted(path.name for path in ROOT.glob("*.py")), [])

    def test_data_recipe_excludes_training_and_evaluation_fields(self):
        template = json.loads((ROOT / "configs/dialect-specific.example.json").read_text())
        self.assertNotIn("train_options", template)
        self.assertNotIn("dev_references", template)
        self.assertNotIn("test_references", template)
        self.assertNotIn("run_name", template)
        self.assertNotIn("exp_root", template)
        self.assertEqual(template["tokenizer_vocab_size"], 500)

    def test_shared_recipe_isolates_all_five_dialects(self):
        path = ROOT / "configs/dialect-specific.example.json"
        original = path.read_bytes()
        for code in DIALECTS:
            config = load_config(path, {"dialect": code.lower()})
            self.assertEqual(config["dialect"], code)
            self.assertEqual(config["data_root"], "data")
            self.assertEqual(config["tokenizer_vocab_size"], 500)
            output = subprocess.check_output(
                [sys.executable, str(SCRIPTS / "run.py"), "--dialect", code,
                 "--stage", "prepare", "--dry-run"],
                cwd=ROOT, text=True, encoding="utf-8")
            self.assertIn(f"--dialects {code}", output)
            self.assertIn("[prepare]", output)
            self.assertNotIn("[features]", output)
            self.assertNotIn("[train]", output)
            self.assertNotIn("[decode]", output)
            tokenizer_output = subprocess.check_output(
                [sys.executable, str(SCRIPTS / "run.py"), "--dialect", code,
                 "--stage", "tokenizer", "--dry-run"],
                cwd=ROOT, text=True, encoding="utf-8")
            self.assertIn("build_tokenizer.py", tokenizer_output)
            self.assertNotIn("bpe.py", tokenizer_output)
        self.assertEqual(path.read_bytes(), original)

    def test_data_stages_only_cover_the_training_split(self):
        for stage, marker in (("prepare", "[prepare]"), ("features", "[features]"),
                              ("tokenizer", "[tokenizer]")):
            output = subprocess.check_output(
                [sys.executable, str(SCRIPTS / "run.py"), "--dialect", "DZA",
                 "--stage", stage, "--dry-run"], cwd=ROOT, text=True, encoding="utf-8")
            self.assertIn(marker, output)
            self.assertNotIn("dev", output)
            self.assertNotIn("test", output)
            self.assertNotIn("asr/train.py", output)
            self.assertNotIn("asr/decode.py", output)
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "run.py"), "--dialect", "DZA",
             "--stage", "all", "--dry-run"], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        with self.assertRaises(ValueError):
            load_config(ROOT / "configs/dialect-specific.example.json", {"max_shards": -1})

    def test_native_benchmark_segments_preserve_reference_boundaries(self):
        metadata = {"audios": [{"aid": "SAU#recording", "segments": [
            {"sid": "SAU#recording#2.000#5.200", "begin_time": "2.000", "end_time": "5.200", "text": "النص الأصلي"},
            {"sid": "invalid", "status": "invalid"},
        ]}]}
        rows = list(reference_rows(metadata, {"SAU#recording": Path("recording.wav")}, "SAU"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "SAU#recording#2.000#5.200")
        self.assertEqual(rows[0]["text"], "النص الأصلي")
        self.assertEqual(rows[0]["start"], 2)
        self.assertAlmostEqual(rows[0]["duration"], 3.2)
        self.assertEqual(rows[0]["label_type"], "human")
        with self.assertRaises(FileNotFoundError):
            list(reference_rows(metadata, {}, "SAU"))

    def test_word_alignment(self):
        self.assertEqual(word_errors("a b c".split(), "a x c d".split()), (2, 1, 0, 1))
        self.assertEqual(word_errors("a b".split(), []), (2, 0, 2, 0))
        self.assertEqual(word_errors([], ["a"]), (1, 0, 0, 1))
        self.assertEqual(word_errors(["مرحبا"], ["مرحبا"]), (0, 0, 0, 0))

    def test_alignment_matches_independent_distance(self):
        def oracle(a, b):
            grid = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
            for i in range(len(a) + 1): grid[i][0] = i
            for j in range(len(b) + 1): grid[0][j] = j
            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    grid[i][j] = min(grid[i - 1][j] + 1, grid[i][j - 1] + 1,
                                     grid[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            return grid[-1][-1]
        rng = random.Random(97)
        for _ in range(200):
            a, b = [[rng.choice("abc") for _ in range(rng.randrange(12))] for _ in range(2)]
            error, sub, delete, insert = word_errors(a, b)
            self.assertEqual(error, oracle(a, b))
            self.assertEqual(error, sub + delete + insert)

    def test_five_dialect_macro_and_empty_hypothesis(self):
        refs = [{"id": d, "dialect": d, "duration": 1, "text": "a b"} for d in DIALECTS]
        hyps = [{"id": d, "raw_text": "" if d == "SAU" else "a b"}
                for d in DIALECTS]
        result = score(refs, hyps, dict.fromkeys(DIALECTS, str.strip))
        self.assertEqual(result["macro_wer_percent"], 20)
        self.assertEqual(result["per_dialect"]["SAU"]["deletions"], 2)
        self.assertTrue(result["is_five_dialect_macro"])

    def test_missing_duplicate_and_invalid_hypotheses_fail(self):
        refs = [{"id": "x", "dialect": "SAU", "duration": 1, "text": "a"}]
        valid = {"id": "x", "raw_text": "a", "text": "a"}
        for rows in ([], [valid, valid], [{"id": "x", "text": "a"}]):
            with self.assertRaises(ValueError):
                score(refs, rows, {"SAU": str.strip})

    def test_duration_policy_and_pooled_vs_macro(self):
        refs = [{"id": "x", "dialect": "SAU", "duration": 2, "text": "a b c"},
                {"id": "y", "dialect": "IRQ", "duration": 1, "text": "a"},
                {"id": "z", "dialect": "IRQ", "duration": 0.5, "text": "a"}]
        hyps = [{"id": r["id"], "raw_text": "a b c" if r["id"] == "x" else ""}
                for r in refs]
        result = score(refs, hyps, dict.fromkeys(DIALECTS, str.strip))
        self.assertEqual(result["macro_wer_percent"], 50)
        self.assertEqual(result["pooled_wer_percent"], 25)
        self.assertEqual(result["skipped_duration"], 1)
        self.assertFalse(result["is_five_dialect_macro"])

    def test_safe_selective_tar_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with tarfile.open(root / "audio.tar", "w") as out:
                for name in ["001/keep.wav", "001/unused.wav"]:
                    item = tarfile.TarInfo(name)
                    item.size = 3
                    out.addfile(item, io.BytesIO(b"abc"))
            seen = extract_selected(root / "audio.tar", root / "audio", lambda name: "keep" in name)
            self.assertEqual(seen, {"001/keep.wav"})
            self.assertEqual((root / "audio/001/keep.wav").read_bytes(), b"abc")
            self.assertFalse((root / "audio/001/unused.wav").exists())

    def test_tar_traversal_and_links_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, member_type in [("../escape.wav", tarfile.REGTYPE), ("link.wav", tarfile.SYMTYPE)]:
                with tarfile.open(root / "audio.tar", "w") as out:
                    item = tarfile.TarInfo(name)
                    item.type = member_type
                    item.linkname = "/outside"
                    out.addfile(item)
                with self.assertRaises(ValueError):
                    extract_selected(root / "audio.tar", root / "audio", lambda _: True)

    def test_dza_boundary_and_shard_paths(self):
        for name in ["001.tar", "audio/001.tar", "DZA/audio/001.tar"]:
            self.assertEqual(shard_path("DZA", name), "DZA/audio/001.tar")
        with self.assertRaises(ValueError): shard_path("DZA", "../001.tar")
        with self.assertRaises(ValueError): shard_path("DZA", "SAU/audio/001.tar")

    def test_raw_export_keeps_internal_wer_on_current_textnorm(self):
        try:
            from decode import _wer_results
        except ImportError:
            self.skipTest("decoding dependencies are not installed")
        params = types.SimpleNamespace(dialect="SAU", raw_transcripts=True)
        raw = [("x", ["هذه", "المدينة"], ["هذي", "المدينه"])]
        normalized = _wer_results(params, raw)
        self.assertEqual(raw[0][1], ["هذه", "المدينة"])
        self.assertEqual(normalized[0][1], ["هذي", "المدينه"])
        self.assertEqual(normalized[0][2], ["هذي", "المدينه"])

    def test_raw_export_and_scoring_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = types.SimpleNamespace(
                raw_transcripts=False, keep_raw_text=True,
                res_dir=Path(tmp), suffix="epoch-1",
            )
            export_raw_hypotheses(
                p, "test", "greedy",
                [("a", ["مرحبا"], ["مرحبا"])],
                [("a", ["مرحبا"], ["مرحبا"])],
            )
            path = next(Path(tmp).glob("hyp-*.jsonl"))
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["raw_text"], "مرحبا")
            self.assertEqual(rows[0]["text"], "مرحبا")
            result = score([{"id": "a", "text": "مرحبا", "duration": 1, "dialect": "SAU"}],
                           rows, {"SAU": str.strip})
            self.assertEqual(result["macro_wer_percent"], 0)

    def test_raw_hook_preserves_legacy_defaults(self):
        for name in ["decode.py"]:
            tree = ast.parse((ASR / name).read_text(encoding="utf-8"))
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_normalize_for_wer")
            namespace = {"get_recipe_normalizer_with_word_level": lambda _: (lambda text: "normalized", None),
                         "_normalizer_logged": {"SAU"}, "logging": logging}
            exec(compile(ast.Module(body=[function], type_ignores=[]), name, "exec"), namespace)
            fn = namespace["_normalize_for_wer"]
            self.assertEqual(fn(types.SimpleNamespace(dialect="SAU"), "raw text"), ["normalized"])
            self.assertEqual(fn(types.SimpleNamespace(dialect="SAU", raw_transcripts=True), "raw text"), ["raw", "text"])
        parser = argparse.ArgumentParser()
        add_pipeline_arguments(parser)
        self.assertFalse(parser.parse_args([]).keep_eval_cuts)
        self.assertFalse(parser.parse_args([]).raw_transcripts)

    def test_tokenizer_reuse_requires_matching_metadata(self):
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            self.skipTest("sentencepiece is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = root / "train.txt"
            text.write_text(
                "\n".join([
                    "مرحبا بالعالم هذا اختبار",
                    "الجملة الثانية من النص",
                    "نحن نختبر بناء التوكنيزر",
                    "اللغة العربية جميلة جدا",
                    "هذا نص تدريبي قصير",
                    "المزيد من الكلمات العربية",
                    "الطقس اليوم جميل",
                    "أنا أحب تعلم اللغات",
                    "الجامعة قريبة من هنا",
                    "الكتاب على الطاولة",
                    "نحن نعمل على مشروع جديد",
                    "شكرا جزيلا لك",
                ]) + "\n",
                encoding="utf-8",
            )
            lang = root / "lang"
            build_tokenizer(text, lang, 40)
            build_tokenizer(text, lang, 40)  # matching metadata: reuse
            changed = root / "changed.txt"
            changed.write_text(text.read_text(encoding="utf-8") + "سطر إضافي\n",
                               encoding="utf-8")
            with self.assertRaises(ValueError):
                build_tokenizer(changed, lang, 40)
            with self.assertRaises(ValueError):
                build_tokenizer(text, lang, 41)
            build_tokenizer(changed, lang, 40, overwrite=True)
            meta = json.loads((lang / "tokenizer.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["model_type"], "unigram")
            self.assertEqual(meta["requested_vocab_size"], 40)

    def test_training_text_is_normalized_from_training_cuts(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.jsonl"
            target = Path(tmp) / "train.txt"
            source.write_text(
                json.dumps({"supervisions": [{"text": "  مرحبا   بالعالم  "}]}) + "\n"
                + json.dumps({"supervisions": [{"text": "   "}]}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(write_training_text(source, target), 1)
            self.assertEqual(target.read_text(encoding="utf-8"), "مرحبا بالعالم\n")

    def test_dry_run_without_ml_dependencies(self):
        output = subprocess.check_output([sys.executable, str(SCRIPTS / "run.py"), "--config",
                                          "configs/sau.example.json", "--dry-run"], cwd=ROOT, text=True)
        self.assertIn("[prepare]", output)
        self.assertIn("--dialects SAU", output)
        self.assertNotIn("asr/train.py", output)
        self.assertNotIn("asr/decode.py", output)
        self.assertNotIn("torchrun", output)


if __name__ == "__main__":
    unittest.main()
