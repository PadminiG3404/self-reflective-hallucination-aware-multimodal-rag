# Self-Reflective and Hallucination-Aware Multimodal RAG

Research prototype for a hallucination-aware multimodal retrieval-augmented reasoning system.

## Features

- Multimodal encoding with CLIP or BLIP-2
- Cross-modal FAISS retrieval with reliability metrics
- Semantic graph construction using NetworkX and PyTorch Geometric
- Multi-hop reasoning with self-reflection and refinement
- Hallucination detection and uncertainty estimation
- Explainable final responses and evaluation metrics

## Setup

1. Create a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run a Prototype Inference

```bash
python -m src.main
```

### Run Evaluation

```bash
python -m src.main --evaluate
```

### Run With Image Input

```bash
python -m src.main --query "Describe the image" --image data\\images\\sample.jpg
```

## Optional API

```bash
uvicorn src.api.app:app --reload
```

## Project Structure

- src/: core modules
- tests/: unit tests
- data/: dummy datasets
- models/: model checkpoints or adapters
- notebooks/: exploration notebooks
