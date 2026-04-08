# LLM-Enhanced Multimodal Recommendation for Amazon Beauty

## Overview

This repository implements a two-stage recommendation system for Amazon beauty products.

1. **Stage 1: Multimodal item representation**
   - Text embeddings from `all-MiniLM-L6-v2`
   - Image embeddings from `CLIP ViT-L/14`
2. **Stage 2: Graph-based retrieval**
   - LightGCN over the user-item interaction graph
   - Item-item relations such as `also_bought`
3. **Stage 3: Profile-based reranking**
   - A local Qwen2.5-14B user-preference profiler
   - Candidate reranking using base score, history similarity, relation overlap, metadata consistency, and optional LLM profile similarity

The retrieval backbone learns user and item embeddings from interaction structure and multimodal item features. The reranker then refines the top candidates with finer-grained semantic and relation-aware signals.

## Current Pipeline

### Best Verified Result

- Checkpoint: `outputs/checkpoints/final_main.pt`
- Config: `configs/default.yaml`
- Test retrieval-only with the final backbone: `HR@10 = 0.79253`, `NDCG@10 = 0.56435`
- Test end-to-end with reranking: `HR@10 = 0.84825`, `NDCG@10 = 0.63746`

The strongest end-to-end setup now combines:

- a **weighted LightGCN backbone**: 2 propagation layers, ego embeddings, learnable layer weights, and refreshed uniform negatives with learning rate `0.0008`
- **relation-dominant reranking with a larger candidate pool**

The current default config uses `candidate_pool_size = 1500` and the tuned profile-based reranker above.

### Data

- Dataset: Amazon Reviews 2023, `All_Beauty`
- Processed files: `data/processed/all_beauty/`
- Splits: leave-one-out train / val / test

### Model

- Baselines: Popularity, MF, BPR-MF, LightGCN
- Main model: `HybridModel`
  - LightGCN retrieval backbone
  - text/image side features
  - relation aggregation
- Reranker: `LLMProfileReranker`
  - heuristic profile fallback
  - local Qwen profile generation when enabled

## Quick Start

Activate the environment first:

```bash
conda activate g41_project
```

### 1. Data Setup

```bash
bash scripts/run_data_setup.sh
```

### 2. Download Product Images

Without this step, the image modality degrades badly.

```bash
python src/data_setup/download_images.py --data_dir data/processed/all_beauty --workers 64
```

### 3. Extract Multimodal Features

```bash
python scripts/extract_features.py --config configs/default.yaml
```

### 4. Train

Full model with reranking:

```bash
bash scripts/run_train.sh
```

Fast smoke test without reranking:

```bash
python -m src.trainer --config configs/default.yaml --no_rerank
```

One-command pipeline run:

```bash
python scripts/run_full_pipeline.py --python_exec /opt/anaconda3/envs/g41_project/bin/python --enable_llm_profile_demo
```

### 5. Evaluate

```bash
bash scripts/run_eval.sh --model outputs/checkpoints/best_model.pt
```

Search reranker weights on the validation split:

```bash
python scripts/tune_reranker.py --config configs/default.yaml --model_path outputs/checkpoints/final_main.pt --split val --max_pool 300 --trials 300
```

### 6. Inference

```bash
bash scripts/run_infer.sh --model outputs/checkpoints/best_model.pt --user AE22XHMBOBJBXUFCTNYLFMD4UKMA --top_n 10
```

Enable the real local-Qwen profile reranker:

```bash
python -m src.inference --config configs/default.yaml --model_path outputs/checkpoints/best_model.pt --user_id AE22XHMBOBJBXUFCTNYLFMD4UKMA --enable_llm_profile
```

Or force the heuristic profile reranker:

```bash
python -m src.inference --config configs/default.yaml --model_path outputs/checkpoints/best_model.pt --user_id AE22XHMBOBJBXUFCTNYLFMD4UKMA --disable_llm_profile
```

## Ablation

```bash
python -m src.trainer --config configs/default.yaml --no_image
python -m src.trainer --config configs/default.yaml --no_text
python -m src.trainer --config configs/default.yaml --no_relation
python -m src.trainer --config configs/default.yaml --no_rerank
```

## Key Files

- `src/trainer.py`: end-to-end training
- `src/evaluator.py`: checkpoint evaluation
- `src/inference.py`: recommendation demo
- `src/runtime.py`: shared runtime assembly
- `src/models/hybrid_model.py`: multimodal graph retriever
- `src/rerankers/profile_based.py`: LLM-enhanced reranker
- `src/llm_profiles.py`: heuristic / local-Qwen user profiler
- `src/data_setup/download_images.py`: image downloader

## Important Notes

- The previously generated image features were not reliable because local product images were missing. The current pipeline fixes this by downloading or directly reading image URLs.
- The current image branch uses three stabilization steps: semantic alignment from CLIP image space into the text space, confidence-weighted visual injection, and a light text-image alignment regularizer during training.
- The LLM stage is used in **reranking**, where it summarizes user preferences and contributes semantic profile signals without changing the retrieval backbone.
