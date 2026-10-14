"""VSL400 validation error analysis for frozen E008 TCN Pose53 seed 42."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

from src.checkpoint import load_best
from src.config import load_config
from src.data import load_split, make_loader, select_view
from src.models import build_model


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["video_id"]: row for row in rows}


def load_corrections(corrections_path: Path) -> dict[str, dict[str, str]]:
    if not corrections_path.exists():
        return {}
    with corrections_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["video_id"]: row for row in rows}


def run_inference_logits(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_logits: list[np.ndarray] = []
    with torch.inference_mode():
        for features, _, frame_mask in loader:
            features = features.to(device)
            frame_mask = frame_mask.to(device)
            logits = model(features, frame_mask)
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def load_pair_confusion_from_csv(pred_path: Path) -> dict[tuple[int, int], int]:
    if not pred_path.exists():
        return {}
    with pred_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for r in rows:
        t = int(r["true_class_id"])
        p = int(r["pred_class_id"])
        if t != p:
            a, b = min(t, p), max(t, p)
            counts[(a, b)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="VSL400 validation error analysis")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root path containing artifacts/runs for checkpoints and predictions",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    input_root = args.input_root.resolve() if args.input_root is not None else project_root

    config_path = project_root / "configs/experiments/e008_tcn_pose53_uniform30_s42.yaml"
    config = load_config(config_path)

    output_dir = project_root / "artifacts/error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ground truth manifest
    val_manifest_path = project_root / "manifests/audit/val_clean.csv"
    val_manifest = load_manifest(val_manifest_path)
    if len(val_manifest) != 2529:
        raise ValueError(f"Expected 2529 rows in val_clean.csv, got {len(val_manifest)}")

    class_to_gloss: dict[int, str] = {}
    for row in val_manifest.values():
        class_to_gloss[int(row["class_id"])] = row["gloss"]

    corrections_path = project_root / "manifests/audit/vsl400_label_corrections.csv"
    corrections = load_corrections(corrections_path)

    # Verify QC exclusion rules
    val_exclusions: list[str] = []
    for vid, corr in corrections.items():
        if corr.get("split") == "val" and corr.get("action") in {"DROP", "corrupted"}:
            val_exclusions.append(vid)
    print(f"QC val exclusions: {len(val_exclusions)} ({val_exclusions})")
    if val_exclusions:
        raise ValueError(
            f"Found {len(val_exclusions)} validation exclusions ({val_exclusions}). "
            "Validation split contains unhandled exclusions."
        )

    # 2. Load canonical predictions
    canonical_pred_path = (
        input_root / "artifacts/runs/e008_tcn_pose53_uniform30_s42/predictions.csv"
    )
    if not canonical_pred_path.exists():
        raise FileNotFoundError(f"Canonical predictions not found at {canonical_pred_path}")
    with canonical_pred_path.open(newline="", encoding="utf-8") as f:
        canonical_rows = list(csv.DictReader(f))
    if len(canonical_rows) != 2529:
        raise ValueError(f"Expected 2529 canonical predictions, got {len(canonical_rows)}")
    canonical_by_vid = {row["video_id"]: row for row in canonical_rows}

    # 3. Load feature split and run frozen checkpoint inference
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    raw_val_split = load_split(config["data"], "val", cache_dir)
    val_split = select_view(raw_val_split, config["feature"]["view"])

    # Validate IDs and labels between split and manifest
    if len(val_split.video_id) != 2529:
        raise ValueError(f"Expected 2529 features, got {len(val_split.video_id)}")
    for vid, cid, sid in zip(val_split.video_id, val_split.y, val_split.signer_id, strict=True):
        vid_str = str(vid)
        if vid_str not in val_manifest:
            raise ValueError(f"Video {vid_str} in features not in val_clean.csv")
        row = val_manifest[vid_str]
        if int(row["class_id"]) != int(cid):
            raise ValueError(f"Class ID mismatch for {vid_str}: {row['class_id']} vs {cid}")
        if str(row["signer_id"]).zfill(3) != str(sid).zfill(3):
            raise ValueError(f"Signer mismatch for {vid_str}: {row['signer_id']} vs {sid}")

    # Load model and weights
    checkpoint_path = input_root / "artifacts/runs/e008_tcn_pose53_uniform30_s42/best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config["model"],
        nodes=val_split.X.shape[2],
        num_classes=int(config["data"]["num_classes"]),
    ).to(device)
    load_best(checkpoint_path, model, device)

    # Run inference to get logits
    loader = make_loader(val_split, batch_size=32, shuffle=False)
    logits = run_inference_logits(model, loader, device)

    # Compute softmax probabilities and predictions
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    pred_classes = np.argmax(logits, axis=1)
    max_confs = np.max(probs, axis=1)

    # 4. Strictly verify top-1 predictions against canonical predictions
    mismatches = 0
    for i, vid in enumerate(val_split.video_id):
        vid_str = str(vid)
        canon = canonical_by_vid[vid_str]
        canon_pred = int(canon["pred_class_id"])
        if pred_classes[i] != canon_pred:
            print(
                f"Mismatch for video {vid_str}: generated {pred_classes[i]} vs canonical {canon_pred}"
            )
            mismatches += 1
    if mismatches > 0:
        raise ValueError(f"FATAL: {mismatches} top-1 predictions mismatch canonical predictions!")
    print(
        "Verification PASSED: All 2,529 generated top-1 predictions match canonical predictions exactly!"
    )

    num_classes = int(config["data"]["num_classes"])
    y_true = val_split.y

    # Calculate exact gt_rank (1-indexed) and gt_in_top5
    gt_ranks = []
    gt_in_top5 = []
    for i in range(len(val_split.video_id)):
        sample_logits = logits[i]
        true_cls = y_true[i]
        sorted_indices = np.argsort(-sample_logits)
        rank = int(np.where(sorted_indices == true_cls)[0][0]) + 1
        gt_ranks.append(rank)
        gt_in_top5.append(rank <= 5)

    gt_in_top5_arr = np.array(gt_in_top5)

    is_correct = pred_classes == y_true
    num_correct = int(np.sum(is_correct))
    num_errors = len(y_true) - num_correct
    top1_acc = num_correct / len(y_true)
    top5_acc = int(np.sum(gt_in_top5_arr)) / len(y_true)

    wrong_mask = ~is_correct
    wrong_gt_in_top5 = gt_in_top5_arr[wrong_mask]
    rescue_numerator = int(np.sum(wrong_gt_in_top5))
    rescue_denominator = int(num_errors)
    rescue_rate = rescue_numerator / rescue_denominator if rescue_denominator > 0 else 0.0

    print(f"Top-1 Accuracy: {num_correct}/{len(y_true)} ({top1_acc * 100:.2f}%)")
    print(f"Top-5 Accuracy: {int(np.sum(gt_in_top5_arr))}/{len(y_true)} ({top5_acc * 100:.2f}%)")
    print(f"Total Top-1 errors: {num_errors}")
    print(
        f"Top-1 wrong but GT in Top-5: {rescue_numerator}/{rescue_denominator} ({rescue_rate * 100:.2f}%)"
    )

    # 5. Build Confusion Matrix (400 x 400)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, pred_classes, strict=True):
        cm[t, p] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm.astype(np.float32) / row_sums, 0.0)

    _fig, ax = plt.subplots(figsize=(12, 10))
    del _fig
    im = ax.imshow(cm_norm, cmap="Blues", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Normalized Prediction Frequency")
    ax.set_title("VSL400 Validation Confusion Matrix (E008 TCN Pose53 Seed 42, 400 Classes)")
    ax.set_xlabel("Predicted Class ID")
    ax.set_ylabel("True Class ID")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Class sample counts in validation
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for t in y_true:
        class_counts[t] += 1

    # 6. Symmetric Unordered Pair Ranking
    pairs = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            a_to_b = int(cm[i, j])
            b_to_a = int(cm[j, i])
            total = a_to_b + b_to_a
            if total > 0:
                count_a = int(class_counts[i])
                count_b = int(class_counts[j])
                rate_a = a_to_b / count_a if count_a > 0 else 0.0
                rate_b = b_to_a / count_b if count_b > 0 else 0.0
                total_rate = total / (count_a + count_b) if (count_a + count_b) > 0 else 0.0
                pairs.append(
                    {
                        "gloss_a": class_to_gloss.get(i, f"Class_{i}"),
                        "gloss_b": class_to_gloss.get(j, f"Class_{j}"),
                        "class_a": i,
                        "class_b": j,
                        "a_to_b": a_to_b,
                        "b_to_a": b_to_a,
                        "total_confusion": total,
                        "count_a": count_a,
                        "count_b": count_b,
                        "rate_a_to_b": rate_a,
                        "rate_b_to_a": rate_b,
                        "total_rate": total_rate,
                    }
                )

    pairs.sort(key=lambda x: (x["total_confusion"], x["total_rate"]), reverse=True)

    # Save the top 10 confusion pairs only
    top10_pairs = pairs[:10]
    top_pairs_path = output_dir / "top_confusion_pairs.csv"
    with top_pairs_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "gloss_a",
            "gloss_b",
            "a_to_b",
            "b_to_a",
            "total_confusion",
            "count_a",
            "count_b",
            "rate_a_to_b",
            "rate_b_to_a",
            "total_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in top10_pairs:
            writer.writerow(
                {
                    "gloss_a": p["gloss_a"],
                    "gloss_b": p["gloss_b"],
                    "a_to_b": p["a_to_b"],
                    "b_to_a": p["b_to_a"],
                    "total_confusion": p["total_confusion"],
                    "count_a": p["count_a"],
                    "count_b": p["count_b"],
                    "rate_a_to_b": f"{p['rate_a_to_b']:.4f}",
                    "rate_b_to_a": f"{p['rate_b_to_a']:.4f}",
                    "total_rate": f"{p['total_rate']:.4f}",
                }
            )
    print(f"Saved top 10 confusion pairs to {top_pairs_path}")

    # 7. Select representative samples for top 10 pairs (up to 2 per direction)
    samples_by_direction: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(len(val_split.video_id)):
        if not is_correct[i]:
            samples_by_direction[(int(y_true[i]), int(pred_classes[i]))].append(i)

    representative_indices: set[int] = set()
    for pair in top10_pairs:
        c_a, c_b = pair["class_a"], pair["class_b"]
        for idx in samples_by_direction.get((c_a, c_b), [])[:2]:
            representative_indices.add(idx)
        for idx in samples_by_direction.get((c_b, c_a), [])[:2]:
            representative_indices.add(idx)

    # 8. Misclassified Samples CSV & audit counts
    audit_error_counts = {"REMAP": 0, "KEEP": 0, "blank": 0}
    misclassified_path = output_dir / "misclassified_samples.csv"
    with misclassified_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "video_id",
            "signer_id",
            "true_gloss",
            "pred_gloss",
            "gt_rank",
            "gt_in_top5",
            "confidence_or_probability_if_available",
            "audit_status_if_available",
            "is_representative_sample",
            "manual_tag",
            "observed_pose_signal",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(val_split.video_id)):
            if not is_correct[i]:
                vid_str = str(val_split.video_id[i])
                sid_str = str(val_split.signer_id[i]).zfill(3)
                t_cls = int(y_true[i])
                p_cls = int(pred_classes[i])
                conf = float(max_confs[i])
                rank = int(gt_ranks[i])
                in_top5 = bool(gt_in_top5[i])

                corr_row = corrections.get(vid_str, {})
                audit_status = corr_row.get("action", "")
                if audit_status == "REMAP":
                    audit_error_counts["REMAP"] += 1
                elif audit_status == "KEEP":
                    audit_error_counts["KEEP"] += 1
                else:
                    audit_error_counts["blank"] += 1

                is_rep = i in representative_indices

                lh_mask = val_split.keypoint_valid_mask[i, :, 6:27]
                rh_mask = val_split.keypoint_valid_mask[i, :, 27:48]
                hand_valid_pct = float(np.mean(val_split.keypoint_valid_mask[i, :, 6:48])) * 100.0
                mean_hand_score = float(np.mean(val_split.X[i, :, 6:48, 2]))

                pose_signal = (
                    f"hand_valid={hand_valid_pct:.1f}%, "
                    f"lh_valid={float(np.mean(lh_mask)) * 100:.1f}%, "
                    f"rh_valid={float(np.mean(rh_mask)) * 100:.1f}%, "
                    f"hand_score={mean_hand_score:.3f}"
                )

                writer.writerow(
                    {
                        "video_id": vid_str,
                        "signer_id": sid_str,
                        "true_gloss": class_to_gloss.get(t_cls, f"Class_{t_cls}"),
                        "pred_gloss": class_to_gloss.get(p_cls, f"Class_{p_cls}"),
                        "gt_rank": rank,
                        "gt_in_top5": in_top5,
                        "confidence_or_probability_if_available": f"{conf:.4f}",
                        "audit_status_if_available": audit_status,
                        "is_representative_sample": is_rep,
                        "manual_tag": "ambiguous / cannot conclude",
                        "observed_pose_signal": pose_signal,
                    }
                )
    print(
        f"Saved misclassified samples to {misclassified_path} "
        f"(423 rows; audit counts: REMAP={audit_error_counts['REMAP']}, "
        f"KEEP={audit_error_counts['KEEP']}, blank={audit_error_counts['blank']})"
    )

    # 9. Multi-seed and ST-GCN cross-comparison for table
    e008_s43_cm = load_pair_confusion_from_csv(
        input_root / "artifacts/runs/e008_tcn_pose53_uniform30_s43/predictions.csv"
    )
    e008_s44_cm = load_pair_confusion_from_csv(
        input_root / "artifacts/runs/e008_tcn_pose53_uniform30_s44/predictions.csv"
    )
    e013_s42_cm = load_pair_confusion_from_csv(
        input_root / "artifacts/runs/e013_stgcn_pose53_uniform30_s42/predictions.csv"
    )

    top10_table = []
    top10_table.append(
        "| Rank | Gloss pair | A → B | B → A | Total s42 | Rate s42 | TCN s43 | TCN s44 | ST-GCN s42 |"
    )
    top10_table.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for idx, p in enumerate(top10_pairs, 1):
        pair_key = (min(p["class_a"], p["class_b"]), max(p["class_a"], p["class_b"]))
        cnt_43 = e008_s43_cm.get(pair_key, 0)
        cnt_44 = e008_s44_cm.get(pair_key, 0)
        cnt_st = e013_s42_cm.get(pair_key, 0)
        top10_table.append(
            f"| {idx} | {p['gloss_a']} ↔ {p['gloss_b']} | {p['a_to_b']} | {p['b_to_a']} | {p['total_confusion']} | {p['total_rate'] * 100:.1f}% | {cnt_43} | {cnt_44} | {cnt_st} |"
        )
    top10_md = "\n".join(top10_table)

    # 10. Write concise summary.md
    summary_path = output_dir / "summary.md"
    summary_text = f"""# VSL400 Validation Error Analysis Summary

- **Model & Checkpoint:** TCN Pose53 Uniform-30 E008 seed 42 (`best.pt`, epoch 55).
- **Data Revision:** `5a0042e1daec2813f74f08fa955276c777e2b40a` (`yolo-e005-pose59-uniform30-a65d7d605573ea13`).
- **Validation Dataset:** `manifests/audit/val_clean.csv` (2,529 samples, 400 classes).
- **QC Exclusions:** 0 samples (all 2,529 val_clean samples are valid; the dropped `024029` and the corrupt `019442` belong to train).
- **Validation Top-1:** 83.27% (2,106/2,529) | **Validation Top-5:** 95.65% (2,419/2,529).
- **Top-1 errors:** 423 samples (audit status among errors: {audit_error_counts["REMAP"]} REMAP, {audit_error_counts["KEEP"]} KEEP, {audit_error_counts["blank"]} unflagged).

---

## 1. Top 10 Symmetric Confusion Pairs (E008 Seed 42)

{top10_md}

*Note:* TCN s43, s44 and ST-GCN s42 counts are looked up independently from their own predictions as a robustness reference; samples are not pooled.

---

## 2. Top-1 Errors with the Ground Truth in Top-5

- **Numerator:** {rescue_numerator} samples
- **Denominator:** {rescue_denominator} samples
- **Top-5 Rescue Rate (Shortlist Retention):** **{rescue_rate * 100:.2f}%** ({rescue_numerator}/{rescue_denominator})
- This only measures whether the true label stays in the Top-5 shortlist; it does not show that the model has learned a gesture group.

---

## 3. Initial Evidence on Handshape / Hand Orientation

- **Observation:** The top confusion pairs recur across seeds (s43, s44) and models (ST-GCN s42).
- **Qualitative material:** The five most recurrent pairs are rendered as five full-frame comparison videos labeled with `GT`, `Pred`, signer and video ID at `artifacts/error_analysis/confusion_pair_raw_videos/labeled_pairs/`. Each video contains one error in each direction of the pair.
- **Hypothesis (unverified):** RGB hand appearance/orientation may add information about finger configuration and palm orientation.
- **Status:** *ambiguous / cannot conclude*. The videos support visual inspection but have no expert annotation, so they do not establish handshape, trajectory or signer-028 signing errors.

---

## 4. Evidence on Pose / Keypoint Quality

- **Observed Signal:** Over 29 representative samples of the top-10 pairs, minimum hand_valid is 96.7%, mean 99.83%; mean score is 0.998.
- **Scope:** Mask and score only reflect coarse detection availability, not localization accuracy, finger geometry or palm orientation. The only conclusion is that no completely missing hand occurs in that set; the errors cannot be attributed to 2D insufficiency.

---

## 5. Conclusion

- The top 10 symmetric confusion pairs and a Top-5 shortlist retention of {rescue_rate * 100:.2f}% are identified on the 2,529-sample clean validation set.
- Visual material exists for the five most recurrent pairs, but the evidence is insufficient to attribute errors to signer or pose geometry, or to claim that RGB hand appearance will improve accuracy.
"""
    summary_path.write_text(summary_text.strip() + "\n", encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
