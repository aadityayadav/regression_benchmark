import os
import joblib
import time

def save_model(dataset_type, estimator, model_name, output_dir="regression_benchmark/results/models"):
    output_dir = os.path.join(output_dir, dataset_type, time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(estimator, path)
    print(f"[{model_name}] Saved to: {path}")
