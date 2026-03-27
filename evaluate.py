"""
MLflow experiment: evaluate face recognition accuracy at different thresholds.
Run this script after building the gallery to log metrics.

Usage:
    python evaluate.py --test_dir test_pairs/ --model ArcFace
"""

import argparse
import mlflow
import numpy as np
from pathlib import Path
from deepface import DeepFace
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)


def evaluate(test_dir: str, model_name: str, thresholds: list[float]):
    """
    Evaluate verification accuracy over a set of image pairs.

    Expected test_dir layout:
        test_pairs/
            same/       <- pairs of the same person
                pair_01_a.jpg
                pair_01_b.jpg
                pair_02_a.jpg
                pair_02_b.jpg
            different/  <- pairs of different people
                pair_01_a.jpg
                pair_01_b.jpg
    """
    test_path = Path(test_dir)
    pairs = []

    for same_img in sorted((test_path / "same").glob("*_a.*")):
        b = same_img.with_name(same_img.name.replace("_a.", "_b."))
        if b.exists():
            pairs.append((str(same_img), str(b), 1))  # label=1 (same)

    for diff_img in sorted((test_path / "different").glob("*_a.*")):
        b = diff_img.with_name(diff_img.name.replace("_a.", "_b."))
        if b.exists():
            pairs.append((str(diff_img), str(b), 0))  # label=0 (different)

    if not pairs:
        print(f"No pairs found in {test_dir}")
        return

    print(f"Evaluating {len(pairs)} pairs...")

    # Compute distances
    distances, labels = [], []
    for img1, img2, label in pairs:
        try:
            result = DeepFace.verify(
                img1_path=img1, img2_path=img2,
                model_name=model_name,
                detector_backend="mtcnn",
                distance_metric="cosine",
                enforce_detection=False,
            )
            distances.append(result["distance"])
            labels.append(label)
        except Exception as e:
            print(f"  Skipped pair ({img1}): {e}")

    distances = np.array(distances)
    labels    = np.array(labels)

    # MLflow logging
    mlflow.set_experiment("face_recognition_eval")
    with mlflow.start_run(run_name=f"{model_name}_eval"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("n_pairs", len(labels))
        mlflow.log_param("n_same", labels.sum())
        mlflow.log_param("n_different", (1 - labels).sum())

        # AUC
        auc = roc_auc_score(labels, 1 - distances)  # lower dist = more similar
        mlflow.log_metric("auc", round(auc, 4))
        print(f"\nAUC: {auc:.4f}")

        # Threshold sweep
        best_acc, best_thresh = 0, 0
        for tau in thresholds:
            preds = (distances <= tau).astype(int)
            acc = (preds == labels).mean()
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()
            far = fp / max(fp + tn, 1)  # False Accept Rate
            frr = fn / max(fn + tp, 1)  # False Reject Rate

            mlflow.log_metrics({
                f"acc_tau_{tau}":  round(float(acc),  4),
                f"far_tau_{tau}":  round(float(far),  4),
                f"frr_tau_{tau}":  round(float(frr),  4),
            })
            print(f"  tau={tau:.2f} | acc={acc:.3f} | FAR={far:.3f} | FRR={frr:.3f}")

            if acc > best_acc:
                best_acc, best_thresh = acc, tau

        mlflow.log_metric("best_accuracy",  round(best_acc,   4))
        mlflow.log_param( "best_threshold", best_thresh)
        print(f"\nBest threshold: {best_thresh} → accuracy: {best_acc:.4f}")

        # Log final confusion matrix at best threshold
        preds = (distances <= best_thresh).astype(int)
        cm = confusion_matrix(labels, preds)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(classification_report(labels, preds, target_names=["different","same"]),
                        "classification_report.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="test_pairs")
    parser.add_argument("--model",    default="ArcFace")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.2, 0.3, 0.4, 0.5, 0.6])
    args = parser.parse_args()
    evaluate(args.test_dir, args.model, args.thresholds)
