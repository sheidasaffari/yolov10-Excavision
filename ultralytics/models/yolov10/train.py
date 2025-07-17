"""
YOLOv10DetectionTrainer  with integrated MLflow
-----------------------------------------------
• Wraps Ultralytics' DetectionTrainer so every train() call is logged.
• Logs: key hyper-parameters, final scalar metrics, best.pt + result files.
"""

from copy import copy
from pathlib import Path
import os

import mlflow
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK

from .model import YOLOv10DetectionModel
from .val import YOLOv10DetectionValidator


class YOLOv10DetectionTrainer(DetectionTrainer):
    # --------------------------------------------------------------------- #
    # ❶  Standard YOLOv10 overrides (unchanged training/validation logic)   #
    # --------------------------------------------------------------------- #
    def get_validator(self):
        """Return the custom DetectionValidator for YOLOv10."""
        self.loss_names = (
            "box_om", "cls_om", "dfl_om",
            "box_oo", "cls_oo", "dfl_oo",
        )
        return YOLOv10DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO-v10 detection model (optionally loading weights)."""
        model = YOLOv10DetectionModel(
            cfg, nc=self.data["nc"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    # --------------------------------------------------------------------- #
    # ❷  MLflow integration                                                 #
    # --------------------------------------------------------------------- #
    def train(self, *args, **kwargs):  # noqa: D401  (flake8 docstring style)
        """Run training and log everything to MLflow."""
        print("🔥  Using custom YOLOv10DetectionTrainer with MLflow enabled")

        # Ensure local file-based tracking store (runs/mlflow)
        mlflow.set_tracking_uri("file://" + os.path.abspath("runs/mlflow"))

        # Close any run left open by previous crashes (safety)
        if mlflow.active_run():
            mlflow.end_run()

        run_name = f"YOLOv10-{self.args.name or 'exp'}"

        with mlflow.start_run(run_name=run_name):
            # ── Log high-level hyper-parameters ───────────────────────────
            param_keys = [
                "model", "data", "epochs", "batch", "imgsz",
                "optimizer", "lr0", "momentum", "weight_decay",
            ]
            for k in param_keys:
                if hasattr(self.args, k):
                    mlflow.log_param(k, getattr(self.args, k))

            # ── Call the original Ultralytics training loop ──────────────
            results = super().train(*args, **kwargs)

            # ── Log final scalar metrics (ignore lists / per-class arrays) ─
            if hasattr(self, "metrics") and isinstance(self.metrics, dict):
                for k, v in self.metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

            # ── Log key artifacts ────────────────────────────────────────
            artifacts = [
                self.save_dir / "weights" / "best.pt",
                self.save_dir / "results.png",
                self.save_dir / "results.csv",
            ]
            for p in artifacts:
                if p.exists():
                    mlflow.log_artifact(str(p))

            # context manager auto-ends the run
            return results



# #Original train.py
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
