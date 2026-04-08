"""Train the simplified LLM-enhanced multimodal recommender."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader import BPRDataset, collate_fn_bpr
from src.metrics import compute_metrics
from src.preprocess import create_negative_samples
from src.runtime import build_hybrid_model, create_reranker, prepare_runtime_bundle
from src.utils import ensure_dir, get_device, load_config, set_seed


class Trainer:
    """Trainer for the hybrid retrieval model."""

    def __init__(self, config, model, bundle, device, reranker=None):
        self.config = config
        self.model = model
        self.bundle = bundle
        self.device = device
        self.reranker = reranker

        self.train_df = bundle["train_df"]
        self.user2idx = bundle["user2idx"]
        self.item2idx = bundle["item2idx"]
        self.idx2item = bundle["idx2item"]
        self.user_histories = bundle["train_df"].sort_values(["user_id", "timestamp"]).groupby("user_id")[
            "item_id"
        ].apply(list).to_dict()

        self.epochs = config["training"]["epochs"]
        self.batch_size = config["training"]["batch_size"]
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.eval_interval = config["training"]["eval_interval"]
        self.early_stopping_patience = config["training"]["early_stopping_patience"]
        self.negatives_per_positive = config["training"].get("negatives_per_positive", 1)
        self.negative_sampling_strategy = config["training"].get("negative_sampling_strategy", "uniform")
        self.hard_negative_ratio = config["training"].get("hard_negative_ratio", 0.0)
        self.popular_negative_ratio = config["training"].get("popular_negative_ratio", 0.0)
        self.popularity_alpha = config["training"].get("popularity_alpha", 1.0)
        self.refresh_negatives_each_epoch = config["training"].get("refresh_negatives_each_epoch", True)
        self.grad_clip_norm = config["training"].get("grad_clip_norm", 0.0)
        self.base_seed = config["training"]["seed"]

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        relations_df = bundle.get("relations")
        self.relation_lookup = {}
        if relations_df is not None and not relations_df.empty:
            for _, row in relations_df.iterrows():
                src = row["item_id"]
                dst = row["related_item_id"]
                self.relation_lookup.setdefault(src, set()).add(dst)
                self.relation_lookup.setdefault(dst, set()).add(src)

        self.dataloader = None
        self._refresh_dataloader(epoch=0)

    def _refresh_dataloader(self, epoch: int) -> None:
        """Rebuild the training loader with fresh negatives."""
        samples = create_negative_samples(
            self.train_df,
            list(self.item2idx.keys()),
            n_neg=self.negatives_per_positive,
            seed=self.base_seed + epoch,
            strategy=self.negative_sampling_strategy,
            relation_lookup=self.relation_lookup,
            hard_negative_ratio=self.hard_negative_ratio,
            popular_negative_ratio=self.popular_negative_ratio,
            popularity_alpha=self.popularity_alpha,
        )
        dataset = BPRDataset(samples, self.user2idx, self.item2idx)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_bpr,
        )

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch."""
        if epoch == 0 or self.refresh_negatives_each_epoch:
            self._refresh_dataloader(epoch)

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.dataloader, desc="Training", leave=False):
            user_idx = batch["user_idx"].to(self.device)
            pos_idx = batch["pos_item_idx"].to(self.device)
            neg_idx = batch["neg_item_idx"].to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.bpr_loss(user_idx, pos_idx, neg_idx)
            loss.backward()
            if self.grad_clip_norm and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _recommend_for_user(self, user_id: str, scores: np.ndarray, top_n: int, use_reranker: bool) -> list[str]:
        candidate_pool = top_n
        if use_reranker and self.reranker is not None:
            candidate_pool = max(top_n, self.config.get("reranking", {}).get("candidate_pool_size", top_n))

        top_indices = np.argsort(-scores)[:candidate_pool]
        candidates = [self.idx2item[i] for i in top_indices]

        if not use_reranker or self.reranker is None:
            return candidates[:top_n]

        base_scores = {item_id: float(scores[self.item2idx[item_id]]) for item_id in candidates}
        user_history = self.user_histories.get(user_id, [])
        reranked = self.reranker.rerank(user_id, candidates, user_history, base_scores)
        return [row[0] for row in reranked[:top_n]]

    def evaluate(self, eval_df, train_df=None, use_reranker=True):
        """Evaluate the model on validation or test data."""
        self.model.eval()
        exclude_items = {}

        if train_df is not None and not train_df.empty:
            for user_id, group in train_df.groupby("user_id"):
                exclude_items[user_id] = set(group["item_id"].tolist())

        predictions = []
        ground_truth = []
        top_n = self.config["evaluation"]["top_n"]
        user_embeddings = None
        item_embeddings = None

        with torch.no_grad():
            user_embeddings, item_embeddings = self.model.compute_final_embeddings()
            for user_id, group in eval_df.groupby("user_id"):
                if user_id not in self.user2idx:
                    continue
                user_gt = group["item_id"].tolist()
                scores = self.model.predict(
                    user_idx=self.user2idx[user_id],
                    exclude_items=exclude_items.get(user_id, set()),
                    user_embeddings=user_embeddings,
                    item_embeddings=item_embeddings,
                )
                top_items = self._recommend_for_user(user_id, scores, top_n, use_reranker=use_reranker)
                predictions.append(top_items)
                ground_truth.append(user_gt)

        return compute_metrics(predictions, ground_truth, self.config["evaluation"]["k_values"])

    def train(self, val_df=None, test_df=None):
        """Run full training loop."""
        best_metric = -np.inf
        patience_counter = 0
        best_state = None
        monitor_df = val_df if val_df is not None and not val_df.empty else test_df

        for epoch in range(self.epochs):
            loss = self.train_epoch(epoch)
            print(f"Epoch {epoch + 1}/{self.epochs} loss={loss:.4f}")

            if monitor_df is None or (epoch + 1) % self.eval_interval != 0:
                continue

            metrics = self.evaluate(monitor_df, self.train_df, use_reranker=False)
            hr10 = metrics.get("hr@10", metrics.get("hr@5", 0.0))
            print(f"Validation metrics (retrieval only): {metrics}")

            if hr10 > best_metric:
                best_metric = hr10
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if test_df is None or test_df.empty:
            return {}

        final_metrics = self.evaluate(
            test_df,
            self.train_df,
            use_reranker=not self.config["ablation"].get("no_rerank", False),
        )
        print(f"Final test metrics: {final_metrics}")
        return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Train the hybrid recommender")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best_model.pt",
        help="Filename for the saved checkpoint under outputs/checkpoints",
    )
    parser.add_argument("--no_image", action="store_true", help="Disable multimodal image contribution")
    parser.add_argument("--no_text", action="store_true", help="Disable multimodal text contribution")
    parser.add_argument("--no_relation", action="store_true", help="Disable item relation graph")
    parser.add_argument("--no_rerank", action="store_true", help="Disable stage-3 reranking")
    parser.add_argument("--enable_llm_profile", action="store_true", help="Enable local Qwen user-profile generation")
    parser.add_argument("--disable_llm_profile", action="store_true", help="Use heuristic profiles instead of the local LLM")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_image:
        config["ablation"]["no_image"] = True
    if args.no_text:
        config["ablation"]["no_text"] = True
    if args.no_relation:
        config["ablation"]["no_relation"] = True
    if args.no_rerank:
        config["ablation"]["no_rerank"] = True
    if args.enable_llm_profile:
        config.setdefault("reranking", {}).setdefault("llm_profile", {})["enabled"] = True
    if args.disable_llm_profile:
        config.setdefault("reranking", {}).setdefault("llm_profile", {})["enabled"] = False

    set_seed(config["training"]["seed"])
    device = get_device(config.get("device", "cuda"))
    print(f"Using device: {device}")

    bundle = prepare_runtime_bundle(config, device=device)
    features = bundle["features"]
    print(
        "Loaded data:",
        f"train={len(bundle['train_df'])}",
        f"val={len(bundle['val_df'])}",
        f"test={len(bundle['test_df'])}",
        f"users={len(bundle['user2idx'])}",
        f"items={len(bundle['item2idx'])}",
    )
    print(
        "Feature availability:",
        f"text={'yes' if features.get('text') is not None else 'no'}",
        f"image={'yes' if features.get('image') is not None else 'no'}",
        f"fused={'yes' if features.get('fused') is not None else 'no'}",
    )

    model = build_hybrid_model(config, bundle, device)
    reranker = create_reranker(config, bundle, device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Reranker: {reranker.__class__.__name__ if reranker else 'disabled'}")

    trainer = Trainer(config, model, bundle, device, reranker=reranker)
    metrics = trainer.train(val_df=bundle["val_df"], test_df=bundle["test_df"])

    checkpoint_dir = Path(config["data"]["output_dir"]) / "checkpoints"
    ensure_dir(checkpoint_dir)
    checkpoint_path = checkpoint_dir / args.checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user2idx": bundle["user2idx"],
            "idx2user": bundle["idx2user"],
            "item2idx": bundle["item2idx"],
            "idx2item": bundle["idx2item"],
            "config": config,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
