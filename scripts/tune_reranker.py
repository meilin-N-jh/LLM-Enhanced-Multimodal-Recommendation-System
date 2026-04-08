"""Search relation-aware reranker weights on a validation split."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import create_reranker, load_checkpoint_bundle
from src.utils import get_device, load_config


def compute_metrics(signal_rows: list[np.ndarray], weights: np.ndarray, pool: int) -> dict[str, float]:
    """Evaluate a linear reranker on cached candidate signals."""
    hr5 = ndcg5 = hr10 = ndcg10 = hr20 = ndcg20 = 0.0
    for rows in signal_rows:
        sub = rows[:pool]
        ranked = sub[:, 6][np.argsort(-(sub[:, :6] @ weights))]
        for k in (5, 10, 20):
            top = ranked[:k]
            if top.max() <= 0:
                continue
            rank = int(np.where(top > 0)[0][0]) + 1
            gain = 1.0 / math.log2(rank + 1)
            if k == 5:
                hr5 += 1
                ndcg5 += gain
            elif k == 10:
                hr10 += 1
                ndcg10 += gain
            else:
                hr20 += 1
                ndcg20 += gain

    total = max(len(signal_rows), 1)
    return {
        "hr@5": hr5 / total,
        "ndcg@5": ndcg5 / total,
        "hr@10": hr10 / total,
        "ndcg@10": ndcg10 / total,
        "hr@20": hr20 / total,
        "ndcg@20": ndcg20 / total,
    }


def cache_signals(bundle: dict, model, reranker, split: str, max_pool: int) -> list[np.ndarray]:
    """Precompute per-user candidate signals so weight search is cheap."""
    eval_df = bundle["val_df"] if split == "val" else bundle["test_df"]
    train_df = bundle["train_df"]
    idx2item = bundle["idx2item"]
    item2idx = bundle["item2idx"]
    user2idx = bundle["user2idx"]
    user_histories = train_df.sort_values(["user_id", "timestamp"]).groupby("user_id")["item_id"].apply(list).to_dict()
    exclude_items = {user_id: set(group["item_id"].tolist()) for user_id, group in train_df.groupby("user_id")}

    rows: list[np.ndarray] = []
    with torch.no_grad():
        user_embeddings, item_embeddings = model.compute_final_embeddings()
        for user_id, group in eval_df.groupby("user_id"):
            if user_id not in user2idx:
                continue

            ground_truth = set(group["item_id"].tolist())
            scores = model.predict(
                user2idx[user_id],
                exclude_items.get(user_id, set()),
                user_embeddings=user_embeddings,
                item_embeddings=item_embeddings,
            )
            top_indices = scores.argsort()[::-1][:max_pool]
            candidates = [idx2item[i] for i in top_indices]
            base_scores = {item_id: float(scores[item2idx[item_id]]) for item_id in candidates}
            normalized_base = reranker._normalize_base_scores(candidates, base_scores)
            history = user_histories.get(user_id, [])

            user_rows = []
            for item_id in candidates:
                user_rows.append(
                    [
                        normalized_base.get(item_id, 0.0),
                        reranker.compute_history_similarity(item_id, history),
                        reranker.compute_history_image_similarity(item_id, history),
                        reranker.compute_profile_similarity(user_id, item_id, history),
                        reranker.compute_relation_overlap(item_id, history),
                        reranker.compute_metadata_overlap(item_id, history),
                        1.0 if item_id in ground_truth else 0.0,
                    ]
                )
            rows.append(np.array(user_rows, dtype=np.float32))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune reranker weights with cached candidate signals")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_path", type=str, default="outputs/checkpoints/final_main.pt")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--max_pool", type=int, default=300)
    parser.add_argument("--pool_values", type=int, nargs="+", default=[80, 120, 200, 300])
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/rerank_tuning_results.json",
        help="Where to save the ranked trial list",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    device = get_device(config.get("device", "cuda"))
    _, bundle, model = load_checkpoint_bundle(args.model_path, config=config, device=device)
    reranker = create_reranker(config, bundle, device)
    if reranker is None:
        raise RuntimeError("Reranker is disabled. Enable reranking before running the search.")

    signal_rows = cache_signals(bundle, model, reranker, split=args.split, max_pool=args.max_pool)

    current_weights = np.array(
        [
            config["reranking"]["profile_based"]["w_base"],
            config["reranking"]["profile_based"]["w_history_text"],
            config["reranking"]["profile_based"]["w_history_image"],
            config["reranking"]["profile_based"]["w_llm_profile"],
            config["reranking"]["profile_based"]["w_relation"],
            config["reranking"]["profile_based"]["w_metadata"],
        ],
        dtype=np.float32,
    )
    current_weights = current_weights / current_weights.sum()

    trials = []
    current_pool = min(config["reranking"]["candidate_pool_size"], args.max_pool)
    current_metrics = compute_metrics(signal_rows, current_weights, current_pool)
    trials.append(
        {
            "label": "current",
            "pool": current_pool,
            "weights": current_weights.tolist(),
            "metrics": current_metrics,
            "score": current_metrics["hr@10"] + current_metrics["ndcg@10"],
        }
    )

    alpha = np.array([0.4, 0.6, 0.8, 0.4, 4.0, 1.2], dtype=np.float32)
    for _ in range(args.trials):
        weights = np.random.dirichlet(alpha).astype(np.float32)
        pool = random.choice(args.pool_values)
        metrics = compute_metrics(signal_rows, weights, pool)
        trials.append(
            {
                "label": "random_search",
                "pool": pool,
                "weights": weights.tolist(),
                "metrics": metrics,
                "score": metrics["hr@10"] + metrics["ndcg@10"],
            }
        )

    trials.sort(key=lambda row: row["score"], reverse=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trials[:50], indent=2), encoding="utf-8")

    print(f"Cached {len(signal_rows)} users from split={args.split}")
    print(json.dumps(trials[0], indent=2))
    print(f"Saved top results to {output_path}")


if __name__ == "__main__":
    main()
