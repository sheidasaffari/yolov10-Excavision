# ultralytics/models/yolov10/train.py
"""
YOLOv10DetectionTrainer with MLflow support
-------------------------------------------
This file extends Ultralytics' DetectionTrainer by adding automatic MLflow
experiment tracking.  Only two small hooks are used:
• start an MLflow run before training
• log parameters + artifacts after training
"""

from copy import copy
from pathlib import Path

import mlflow
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK

from .model import YOLOv10DetectionModel
from .val import YOLOv10DetectionValidator


class YOLOv10DetectionTrainer(DetectionTrainer):
    # ----------------------------------------------------------
    # ❶  Required YOLOv10 overrides (unchanged logic)
    # ----------------------------------------------------------
    def get_validator(self):
        """Return a DetectionValidator for YOLOv10."""
        # store loss names for pretty printing
        self.loss_names = (
            "box_om",
            "cls_om",
            "dfl_om",
            "box_oo",
            "cls_oo",
            "dfl_oo",
        )
        return YOLOv10DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLOv10 detection model."""
        model = YOLOv10DetectionModel(
            cfg, nc=self.data["nc"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    # ----------------------------------------------------------
    # ❷  MLflow integration
    # ----------------------------------------------------------
    def train(self, *args, **kwargs):  # noqa: D401
        """Wrap the parent .train() with MLflow logging."""
        run_name = f"YOLOv10-{self.args.name or 'exp'}"
        mlflow.start_run(run_name=run_name)

        # ── Log high-level parameters ──────────────────────────
        param_keys = [
            "model",
            "data",
            "epochs",
            "batch",
            "imgsz",
            "optimizer",
            "lr0",
            "momentum",
            "weight_decay",
        ]
        for k in param_keys:
            if hasattr(self.args, k):
                mlflow.log_param(k, getattr(self.args, k))

        # ── Call the original Ultralytics training loop ───────
        results = super().train(*args, **kwargs)

        # ── Log final metrics if available ─────────────────────
        if hasattr(self, "metrics") and isinstance(self.metrics, dict):
            # metrics dict keys vary by version; log whatever exists
            for k, v in self.metrics.items():
                # only scalars (ignore lists like per-class metrics)
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

        # ── Log useful artifacts ───────────────────────────────
        best_pt = self.save_dir / "weights" / "best.pt"
        res_png = self.save_dir / "results.png"
        res_csv = self.save_dir / "results.csv"

        for p in (best_pt, res_png, res_csv):
            if p.exists():
                mlflow.log_artifact(str(p))

        mlflow.end_run()
        return results



##Original train.py
# from ultralytics.models.yolo.detect import DetectionTrainer
# from .val import YOLOv10DetectionValidator
# from .model import YOLOv10DetectionModel
# from copy import copy
# from ultralytics.utils import RANK

# class YOLOv10DetectionTrainer(DetectionTrainer):
#     def get_validator(self):
#         """Returns a DetectionValidator for YOLO model validation."""
#         self.loss_names = "box_om", "cls_om", "dfl_om", "box_oo", "cls_oo", "dfl_oo", 
#         return YOLOv10DetectionValidator(
#             self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
#         )

#     def get_model(self, cfg=None, weights=None, verbose=True):
#         """Return a YOLO detection model."""
#         model = YOLOv10DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
#         if weights:
#             model.load(weights)
#         return model
