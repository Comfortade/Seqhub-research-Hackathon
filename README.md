# SeqHub — Earnings Call Intelligence Pipeline

>  Hackathon 2026 Submission

An end-to-end pipeline that extracts structured intelligence from raw earnings call audio — transcription, speaker diarization, topic segmentation, boundary detection, and sentiment analysis.

---

## Pipeline Architecture
```
Raw Audio (.mp3)
      │
      ▼
WhisperX large-v3 (Transcription)
      │
      ▼
PyAnnote 3.1 (Speaker Diarization)
      │
      ▼
WhisperX Forced Alignment (Word-level timestamps)
      │
      ▼
E5-large-v2 + PELT (Topic Boundary Detection)
      │
      ▼
Boundary-based Clustering + c-TF-IDF (Topic Segmentation)
      │
      ▼
GPT-4o-mini (Topic Labeling)
      │
      ▼
FinBERT + GPT-4o-mini Hybrid (Sentiment Analysis)
      │
      ▼
Structured JSON Output
```

---

## Results

| Metric | Score | Direction |
|---|---|---|
| WER | 0.0737 | Lower is better |
| Macro-F1 | 0.8185 | Higher is better |
| NMI | 0.6364 | Higher is better |
| C_v | 0.6429 | Higher is better |
| WindowDiff | 0.4737 | Lower is better |
| Pk | 0.4211 | Lower is better |

---

## Models 

| Stage | Model | Source |
|---|---|---|
| Transcription | WhisperX large-v3 | Open source |
| Diarization | PyAnnote 3.1 | HuggingFace (gated) |
| Embeddings | E5-large-v2 | HuggingFace |
| Sentiment | FinBERT | HuggingFace |
| Topic Labeling | GPT-4o-mini | OpenAI API |

---

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/comfortade/seqhub-reach-hackathon.git
cd seqhub-reach-hackathon
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
apt-get install -y ffmpeg
```

### 3. Set up API keys
You need two keys:
- HuggingFace token (for PyAnnote — must accept model terms)
- OpenAI API key (for GPT-4o-mini topic labeling)

### 4. Accept PyAnnote model terms
Visit and click Accept on both:
- huggingface.co/pyannote/speaker-diarization-3.1
- huggingface.co/pyannote/segmentation-3.0

### 5. Run the notebook
Open `notebook.ipynb` and run all cells in order.

---

## Input / Output

**Input:** Any earnings call audio file (.mp3 or .wav)

**Output:** Structured JSON containing:
```json
{
  "turns": [
    {
      "turn_id": 0,
      "speaker": "CEO",
      "text": "Q4 was a milestone quarter...",
      "start": 1.29,
      "end": 6.14,
      "topic_id": 1,
      "topic_label": "Financial Performance",
      "topic_keywords": ["milestone", "record", "growth"],
      "sentiment": "positive",
      "is_boundary": false
    }
  ],
  "metrics": {
    "WER": 0.0737,
    "Macro_F1": 0.8185,
    "NMI": 0.6364,
    "C_v": 0.6429,
    "WindowDiff": 0.4737,
    "Pk": 0.4211
  }
}
```

---

## System Design

### Key Decisions

**WhisperX over plain Whisper** — WhisperX adds VAD and forced alignment producing word-level timestamps critical for accurate diarization merging.

**PyAnnote 3.1 for diarization** — State of the art open source DER. Requires HuggingFace token and manual license acceptance.

**E5-large-v2 for embeddings** — Outperforms MiniLM and MPNet on semantic similarity tasks. Critical for boundary detection quality.

**PELT with speaker bonus** — Pure cosine similarity misses boundaries at speaker changes. Adding a speaker change bonus signal improved WindowDiff by 0.21.

**FinBERT + GPT hybrid sentiment** — FinBERT handles high-confidence predictions. GPT-4o-mini handles ambiguous cases and the mixed sentiment class which FinBERT cannot natively produce.

### Tradeoffs

- Topic coherence (C_v) is limited by transcript length — longer calls would score higher
- GPT labeling is constrained to known domain taxonomy — realistic for production earnings call systems




