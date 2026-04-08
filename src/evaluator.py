"""Evaluate a trained checkpoint on validation or test data."""

from __future__ import annotations

import argparse

import torch

from src.metrics import compute_metrics
from src.runtime import create_reranker, load_checkpoint_bundle
from src.utils import get_device, load_config


def evaluate_model(model, bundle, config, reranker=None, split="test"):
    """Run ranking evaluation on the requested split."""
    eval_df = bundle["val_df"] if split == "val" else bundle["test_df"]
    train_df = bundle["train_df"]
    idx2item = bundle["idx2item"]
    item2idx = bundle["item2idx"]
    user2idx = bundle["user2idx"]
    user_histories = train_df.sort_values(["user_id", "timestamp"]).groupby("user_id")["item_id"].apply(list).to_dict()

    exclude_items = {
        user_id: set(group["item_id"].tolist())
        for user_id, group in train_df.groupby("user_id")
    }

    predictions = []
    ground_truth = []
    top_n = config["evaluation"]["top_n"]
    candidate_pool = max(top_n, config.get("reranking", {}).get("candidate_pool_size", top_n))

    with torch.no_grad():
        user_embeddings, item_embeddings = model.compute_final_embeddings()
        for user_id, group in eval_df.groupby("user_id"):
            if user_id not in user2idx:
                continue
            scores = model.predict(
                user2idx[user_id],
                exclude_items.get(user_id, set()),
                user_embeddings=user_embeddings,
                item_embeddings=item_embeddings,
            )
            top_indices = scores.argsort()[::-1][:candidate_pool]
            candidates = [idx2item[i] for i in top_indices]

            if reranker is not None:
                base_scores = {item_id: float(scores[item2idx[item_id]]) for item_id in candidates}
                reranked = reranker.rerank(user_id, candidates, user_histories.get(user_id, []), base_scores)
                top_items = [row[0] for row in reranked[:top_n]]
            else:
                top_items = candidates[:top_n]

            predictions.append(top_items)
            ground_truth.append(group["item_id"].tolist())

    return compute_metrics(predictions, ground_truth, config["evaluation"]["k_values"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--enable_llm_profile", action="store_true")
    parser.add_argument("--disable_llm_profile", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_rerank:
        config["ablation"]["no_rerank"] = True
    if args.enable_llm_profile:
        config.setdefault("reranking", {}).setdefault("llm_profile", {})["enabled"] = True
    if args.disable_llm_profile:
        config.setdefault("reranking", {}).setdefault("llm_profile", {})["enabled"] = False

    device = get_device(config.get("device", "cuda"))
    checkpoint, bundle, model = load_checkpoint_bundle(args.model_path, config=config, device=device)
    reranker = None if config["ablation"].get("no_rerank", False) else create_reranker(config, bundle, device)

    metrics = evaluate_model(model, bundle, config, reranker=reranker, split=args.split)
    print(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")
    print(f"Evaluated on {args.split}: {metrics}")


if __name__ == "__main__":
    main()
