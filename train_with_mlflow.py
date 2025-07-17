# train_with_mlflow.py
from ultralytics import YOLO
import mlflow

EXPT = "YOLOv10-Excavator"
RUN = "v10n_pretrained_416"
MODEL = "yolov10n.pt"
YAML = "data/excavision.yaml"

PARAMS = dict(imgsz=416, batch=16, amp=True, epochs=100, workers=2, device=0)

mlflow.set_experiment(EXPT)
with mlflow.start_run(run_name=RUN):
    mlflow.log_param("model", MODEL)
    mlflow.log_param("dataset_yaml", YAML)
    mlflow.log_params(PARAMS)

    model = YOLO(MODEL)
    results = model.train(data=YAML, name=RUN, **PARAMS)

    for k, v in results.metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k.replace("metrics/", ""), v)

    mlflow.log_artifacts(f"runs/detect/{RUN}")
