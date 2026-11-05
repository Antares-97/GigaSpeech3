<div align="center">

# 🎙️ GigaSpeech 3

### Scaling Multi-Dialect Arabic Speech Recognition with Weakly Labeled Data

[![Hugging Face Dataset](https://img.shields.io/badge/🤗_Hugging_Face-Dataset-FFD21E?style=flat-square)](https://huggingface.co/datasets/speechcolab/gigaspeech3)
[![arXiv](https://img.shields.io/badge/arXiv-2611.02476-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2611.02476)
[![Online Demo](https://img.shields.io/badge/🎧_Online_Demo-Try_ASR-E11D48?style=flat-square)](https://chenxie95-gigaspeech3-arabic-asr.hf.space)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/Antares-97/GigaSpeech3)
[![Benchmark](https://img.shields.io/badge/📊_GigaSpeechBench-Evaluation-2563EB?style=flat-square)](https://github.com/SpeechColab/GigaSpeechBench)

![Segments](https://img.shields.io/badge/Segments-~10M-0D9488?style=flat-square)
![Dialects](https://img.shields.io/badge/Arabic_Dialects-5-7C3AED?style=flat-square)
![Supervision](https://img.shields.io/badge/Supervision-Weak_Labels-EA580C?style=flat-square)

**Algerian · Egyptian · Iraqi · Saudi · Syrian**

</div>

---

GigaSpeech 3 is a large-scale, weakly labeled Arabic speech corpus containing approximately **10 million speech segments** across **Algerian (DZA), Egyptian (EGY), Iraqi (IRQ), Saudi (SAU), and Syrian (SYR) Arabic**. It supports dialect-specific and multi-dialect automatic speech recognition (ASR), weak supervision, and low-resource speech research.

The accompanying manuscript reports approximately **39,000 hours** for its study corpus. Starting from only 51–63 hours of manual transcription per dialect, GigaSpeech 3 combines ASR-biased self-supervised learning, automatic transcription, data selection, and noisy-student refinement to build reusable audio–text supervision.

## 📥 Download and Demo

- 📄 **Paper:** [arXiv](https://arxiv.org/abs/2611.02476) · [PDF](https://arxiv.org/pdf/2611.02476)
- 🤗 **Training data:** [GigaSpeech 3 on Hugging Face](https://huggingface.co/datasets/speechcolab/gigaspeech3)
- 🎧 **Online demo:** [Try GigaSpeech 3 Arabic ASR](https://chenxie95-gigaspeech3-arabic-asr.hf.space)
- 🧪 **Evaluation data:** [GigaSpeechBench on Hugging Face](https://huggingface.co/datasets/speechcolab/GigaSpeechBench)
- 🏆 **Evaluation toolkit and leaderboard:** [SpeechColab/GigaSpeechBench](https://github.com/SpeechColab/GigaSpeechBench)

See the [dataset card](https://huggingface.co/datasets/speechcolab/gigaspeech3/blob/main/README.md) for file descriptions and terms of use. The dialect-specific training recipe is being prepared for release. Model checkpoint downloads will be listed here when available; the online demo can already be used for inference.

## 🛠️ Training and Evaluation Resources

The release recipe follows four steps for each dialect (`DZA`, `EGY`, `IRQ`, `SAU`, or `SYR`): **HF training data → Lhotse manifests and Fbank → dialect-specific BPE → Zipformer training → decoding**.

| Resource | Links |
|:---|:---|
| GigaSpeech 3 | [Paper](https://arxiv.org/abs/2611.02476) · [Training data](https://huggingface.co/datasets/speechcolab/gigaspeech3) · [Online demo](https://chenxie95-gigaspeech3-arabic-asr.hf.space) |
| Training framework | [VietASR code](https://github.com/zzasdf/VietASR) · [VietASR paper](https://www.isca-archive.org/interspeech_2025/zhuo25_interspeech.html) · [icefall](https://github.com/k2-fsa/icefall) · [Lhotse](https://github.com/lhotse-speech/lhotse) |
| Evaluation | [GigaSpeechBench data](https://huggingface.co/datasets/speechcolab/GigaSpeechBench) · [Evaluation code](https://github.com/SpeechColab/GigaSpeechBench) · [Paper](https://arxiv.org/abs/2606.28884) |
| Noisy-student refinement (NST) | [GigaSpeech 2 code](https://github.com/SpeechColab/GigaSpeech2) · [Pipeline](https://github.com/SpeechColab/GigaSpeech2/tree/main/pipeline) · [Data](https://huggingface.co/datasets/speechcolab/gigaspeech2) · [Paper](https://aclanthology.org/2025.acl-long.135/) |

**Evaluation inputs:** users prepare separate human-transcribed development and test manifests, preserving the benchmark's segment IDs, boundaries, and references. Use a development set that is independent of the test set; do not use benchmark test data for model selection. Record the text-normalization version and scoring settings used for WER evaluation.

**NST scope:** the released SAU/SYR partitions already contain the final r4-filtered weak labels. GigaSpeech 2 provides the upstream refinement method and implementation; its dataset is separate from our Arabic data and does not contain our intermediate NST rounds. Training on the released final partitions does not rerun the four-round construction process.

The initial code release focuses on dialect-specific Zipformer training and decoding. Multi-dialect joint LID–ASR training and Qwen adaptation are separate experiments described in the paper.

## 🌍 Dataset

### 🗣️ Dialects

| Code | Dialect | Training hours |
|:---:|:---|---:|
| DZA | Algerian Arabic | 11,461 |
| EGY | Egyptian Arabic | 8,533 |
| IRQ | Iraqi Arabic | 10,162 |
| SAU | Saudi Arabic | 4,512 |
| SYR | Syrian Arabic | 4,361 |
| **Total** | **Five dialects** | **39,028** |

*Training durations are reported in the [GigaSpeech 3 paper](https://arxiv.org/abs/2611.02476) and rounded to whole hours; the total is reported separately in the paper.*

Each dialect is released as a weakly labeled training partition. The manual seed data and MSA data used in the manuscript are separate from these five partitions. Evaluation uses independently transcribed data from GigaSpeechBench and external Arabic corpora.

### 📦 Data Format

Audio is stored in tar shards, with a Parquet metadata index for each dialect:

```text
GigaSpeech3/
├── DZA/
│   ├── audio/001.tar, 002.tar, ...
│   └── metadata/DZA.parquet
├── EGY/
├── IRQ/
├── SAU/
└── SYR/
```

All dialect directories use the same layout. Each Parquet row describes one audio segment:

| Field | Type | Description |
|:---|:---|:---|
| `id` | string | Segment identifier; retain it when preparing manifests and decoding outputs. |
| `duration` | float64 | Segment duration in **seconds**. |
| `text` | string | Automatically generated, weakly supervised transcript. |
| `dialect` | string | One of `DZA`, `EGY`, `IRQ`, `SAU`, or `SYR`. |
| `shard` | string | Audio tar filename, for example `001.tar`. |
| `member` | string | Exact path of the segment audio inside its tar archive. |

Locate an audio segment using `<dialect>/audio/<shard>`, then read the archive member named by `member`. Each audio file is already segmented; a Lhotse cut for that file starts at 0 seconds. The six-field index contains weak training transcripts, not human-transcribed development/test references or confidence scores.

### 🚀 Quick Start

```bash
pip install -U datasets huggingface_hub
```

Read metadata without downloading the audio archives:

```python
from datasets import load_dataset

repo_id = "speechcolab/gigaspeech3"
metadata = load_dataset(repo_id, "SAU", split="train", streaming=True)
sample = next(iter(metadata))
print(sample)
```

Download the archive containing the selected segment:

```python
from huggingface_hub import hf_hub_download

archive_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=f"{sample['dialect']}/audio/{sample['shard']}",
)
```

The audio is located at `sample["member"]` inside this archive. Metadata loading does not automatically download or decode audio; individual shards can be several gigabytes.

## 🔄 Corpus Construction

```text
Dialect-matched web speech + small manually transcribed seed sets
    → VietASR-style ASR-biased self-supervised labeling models
    → automatic transcription and data selection
    → four-round noisy-student refinement for SAU and SYR
    → weakly labeled corpus for downstream ASR training
```

The manuscript keeps manual, unlabeled, and evaluation channels disjoint. DZA, EGY, and IRQ use selected weak labels; SAU and SYR use the fourth-round-filtered inventories. Downstream Zipformers are trained from scratch to evaluate the value of the resulting audio–text pairs independently of the labeling-model weights.

## 📊 Results

### 🏆 Comparison with Commercial APIs and Other ASR Systems

**WER (%) on five GigaSpeechBench v3 Arabic dialects; lower is better.** Third-party system results are reproduced from the [GigaSpeechBench paper](https://arxiv.org/abs/2606.28884); results marked “ours” are from our GigaSpeech 3 experiments. All systems use the released test sets and a common Arabic text normalizer. AVG is the unweighted mean over the five dialects.

| System | DZA | IRQ | SAU | SYR | EGY | AVG |
|:---|---:|---:|---:|---:|---:|---:|
| Dolphin Base | 78.26 | 65.20 | 52.35 | 38.12 | 85.31 | 63.85 |
| Dolphin Small | 72.44 | 62.05 | 50.91 | 30.03 | 74.70 | 58.03 |
| GPT-4o Transcribe | 63.14 | 54.53 | 42.38 | 31.67 | 64.23 | 51.19 |
| FunASR-Realtime | 66.30 | 53.44 | 37.67 | 24.24 | 63.33 | 49.00 |
| Whisper Large v3 | 72.02 | 51.04 | 32.79 | 19.12 | 69.78 | 48.95 |
| Deepgram Nova 3 | 57.90 | 47.54 | 25.02 | 30.61 | 52.77 | 42.77 |
| NVIDIA NeMo | 62.66 | 43.22 | 29.28 | 20.13 | 54.83 | 42.02 |
| Qwen3-ASR-1.7B | 63.43 | 41.27 | 25.85 | 18.50 | 59.23 | 41.66 |
| Meta OmniASR 3B | 57.68 | 38.80 | 25.31 | 17.86 | 52.37 | 38.40 |
| ElevenLabs Scribe v2 | 50.43 | 38.67 | 33.33 | 14.73 | 44.44 | 36.32 |
| Qwen3-ASR-Flash | 57.18 | 33.21 | 19.21 | 14.41 | 48.78 | 34.56 |
| Chirp 3 | 53.11 | 35.71 | 16.76 | 24.13 | 42.71 | 34.48 |
| Azure | 51.22 | 34.61 | 20.09 | 17.74 | 47.65 | 34.26 |
| Gemini 3.0 Flash | <em>44.22</em> | 36.55 | 20.10 | 14.40 | 41.22 | 31.30 |
| Qwen3.5-Omni-Plus | 47.11 | <ins><em>28.54</em></ins> | <em>16.56</em> | <ins><em>13.76</em></ins> | <em>37.12</em> | <em>28.62</em> |
| Dialect-Specific (ours) | <strong>33.64</strong> | 29.99 | <strong>12.12</strong> | 14.25 | <ins>29.32</ins> | <ins>23.86</ins> |
| Multi-Dialect w/o LID (ours) | 36.78 | 28.85 | 12.81 | 16.33 | 32.80 | 25.51 |
| Multi-Dialect w/ LID (ours) | <ins>35.29</ins> | <strong>27.84</strong> | <ins>12.62</ins> | <strong>13.18</strong> | <strong>29.16</strong> | <strong>23.62</strong> |

*Italic*: best third-party result. **Bold**: best overall. <ins>Underlined</ins>: second-best overall.

**LID conditions:** “w/o LID” uses automatically predicted dialect IDs; “w/ LID” uses ground-truth dialect IDs as a diagnostic. Dialect-specific SAU/SYR models include four rounds of noisy-student refinement.

The final dialect-specific models improve over their labeling models on all five dialects. SAU/SYR use the NST r4 recognizers. The multi-dialect model additionally uses separately labeled MSA speech and achieves 10.83% WER on MSA, which is excluded from the averages above.

The manuscript also evaluates eight conditions from **MASC, Casablanca, and SADA**, and reports gains from fine-tuning **Qwen3-ASR-1.7B** with GigaSpeech 3 supervision.

## 📜 Usage and License

The corpus, derived transcripts, metadata, and model checkpoints described in the manuscript are intended **for scientific research and evaluation only**. This release does not grant commercial-use permission. Underlying recordings remain subject to their respective rights holders and source-platform terms; software recipes are licensed separately. Refer to the [dataset terms](https://huggingface.co/datasets/speechcolab/gigaspeech3/blob/main/README.md#terms-of-use).

Transcripts are machine-generated and may contain errors. Country-level dialect labels are coarse, and web-source coverage, code-switching, and numerical expressions remain limitations. Use human-transcribed references for evaluation.

## 📝 Citation

Please cite our [paper](https://arxiv.org/abs/2611.02476) if you find this work useful.

```bibtex
@misc{liu2026gigaspeech3,
  title = {GigaSpeech 3: Scaling Multi-Dialect Arabic Speech Recognition with Weakly Labeled Data},
  author = {Liu, Junzhe and Tu, Yujie and Yang, Yifan and Wang, Peng and Wang, Haoran and Lin, Guodong and Liao, Leyan and Zhang, Chengquan and Gao, Zhifu and Li, Xiangang and Cheng, Qinyuan and Qiu, Xipeng and Zhang, Weiqiang and Chen, Guoguo and Yu, Kai and Chen, Xie},
  year = {2026},
  eprint = {2611.02476},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2611.02476}
}
```

## 💬 Contact

For questions, contributions, or removal requests, open a [GitHub issue](https://github.com/Antares-97/GigaSpeech3/issues) or [dataset discussion](https://huggingface.co/datasets/speechcolab/gigaspeech3/discussions). Avoid posting sensitive personal information publicly.

## 🔗 Related Projects

- [GigaSpeech](https://github.com/SpeechColab/GigaSpeech): large-scale English ASR corpus.
- [GigaSpeech 2](https://github.com/SpeechColab/GigaSpeech2): multilingual ASR corpus for low-resource languages.
- [GigaSpeechBench](https://github.com/SpeechColab/GigaSpeechBench): real-world multilingual speech-to-text benchmark.
