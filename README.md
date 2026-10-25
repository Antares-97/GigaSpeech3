<div align="center">

# 🎙️ GigaSpeech 3

### Scaling Multi-Dialect Arabic Speech Recognition with Weakly Labeled Data

[![Hugging Face Dataset](https://img.shields.io/badge/🤗_Hugging_Face-Dataset-FFD21E?style=flat-square)](https://huggingface.co/datasets/Antares97/GigaSpeech3)
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

## 📥 Download

- 🤗 **Training data:** [GigaSpeech 3 on Hugging Face](https://huggingface.co/datasets/Antares97/GigaSpeech3)
- 🧪 **Evaluation data:** [GigaSpeechBench on Hugging Face](https://huggingface.co/datasets/speechcolab/GigaSpeechBench)
- 🏆 **Evaluation toolkit and leaderboard:** [SpeechColab/GigaSpeechBench](https://github.com/SpeechColab/GigaSpeechBench)

See the [dataset card](https://huggingface.co/datasets/Antares97/GigaSpeech3/blob/main/README.md) for file descriptions and terms of use. The current GitHub repository provides the project overview; training recipes and model checkpoint links will be added as they become available.

## 🌍 Dataset

### 🗣️ Dialects

| Code | Dialect |
|:---:|:---|
| DZA | Algerian Arabic |
| EGY | Egyptian Arabic |
| IRQ | Iraqi Arabic |
| SAU | Saudi Arabic |
| SYR | Syrian Arabic |

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

All dialect directories use the same layout. Metadata fields are `id`, `duration` (seconds), `text` (weak transcript), `dialect`, `shard` (tar filename), and `member` (audio path inside the tar).

### 🚀 Quick Start

```bash
pip install -U datasets huggingface_hub
```

Read metadata without downloading the audio archives:

```python
from datasets import load_dataset

repo_id = "Antares97/GigaSpeech3"
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

**WER (%) on the five target Arabic dialects of GigaSpeechBench; lower is better.** These results are reported in the manuscript using a common Arabic text normalizer. AVG is the unweighted average over the five dialects.

| System | DZA | EGY | IRQ | SAU | SYR | AVG |
|:---|---:|---:|---:|---:|---:|---:|
| Dialect-specific Zipformers | 33.64 | 29.32 | 29.99 | 12.12 | 14.25 | **23.86** |
| Multi-dialect Zipformer, predicted dialect ID | 36.78 | 32.80 | 28.85 | 12.81 | 16.33 | **25.51** |

The final dialect-specific models improve over their labeling models on all five dialects. SAU/SYR use the NST r4 recognizers. The multi-dialect model additionally uses separately labeled MSA speech and achieves 10.83% WER on MSA, which is excluded from the averages above.

The manuscript also evaluates eight conditions from **MASC, Casablanca, and SADA**, and reports gains from fine-tuning **Qwen3-ASR-1.7B** with GigaSpeech 3 supervision.

## 📜 Usage and License

The corpus, derived transcripts, metadata, and model checkpoints described in the manuscript are intended **for scientific research and evaluation only**. This release does not grant commercial-use permission. Underlying recordings remain subject to their respective rights holders and source-platform terms; software recipes are licensed separately. Refer to the [dataset terms](https://huggingface.co/datasets/Antares97/GigaSpeech3/blob/main/README.md#terms-of-use).

Transcripts are machine-generated and may contain errors. Country-level dialect labels are coarse, and web-source coverage, code-switching, and numerical expressions remain limitations. Use human-transcribed references for evaluation.

## 📝 Citation

Please cite the accompanying manuscript if you find this work useful. Publication details will be updated when available.

```bibtex
@misc{liu2026gigaspeech3,
  title = {GigaSpeech 3: Scaling Multi-Dialect Arabic Speech Recognition with Weakly Labeled Data},
  author = {Liu, Junzhe and Tu, Yujie and Wang, Peng and Wang, Haoran and Lin, Guodong and Liao, Leyan and Zhang, Chengquan and Yang, Yifan and Gao, Zhifu and Li, Xiangang and Cheng, Qinyuan and Qiu, Xipeng and Zhang, Weiqiang and Chen, Guoguo and Yu, Kai and Chen, Xie},
  year = {2026},
  howpublished = {Manuscript and dataset},
  url = {https://huggingface.co/datasets/Antares97/GigaSpeech3}
}
```

## 💬 Contact

For questions, contributions, or removal requests, open a [GitHub issue](https://github.com/Antares-97/GigaSpeech3/issues) or [dataset discussion](https://huggingface.co/datasets/Antares97/GigaSpeech3/discussions). Avoid posting sensitive personal information publicly.

## 🔗 Related Projects

- [GigaSpeech](https://github.com/SpeechColab/GigaSpeech): large-scale English ASR corpus.
- [GigaSpeech 2](https://github.com/SpeechColab/GigaSpeech2): multilingual ASR corpus for low-resource languages.
- [GigaSpeechBench](https://github.com/SpeechColab/GigaSpeechBench): real-world multilingual speech-to-text benchmark.
