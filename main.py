import os
import pandas as pd
from models.registry import model_registry
from training.trainer import train_all_models
from evaluation.summarize import generate_summary
from evaluation.feature_importance import aggregate_feature_importance
from utils.io import save_model

# Update the dataset type
dataset_type = "raw"

# Load dataset
data_dir = os.path.join(os.getcwd(), "regression_benchmark/data")
train_dataset = pd.read_csv(os.path.join(data_dir, "train_raw.csv"))
test_dataset = pd.read_csv(os.path.join(data_dir, "test_raw.csv"))

X = train_dataset.drop(columns=["Capacitance (F/g)"])
y = train_dataset["Capacitance (F/g)"]
X_test = test_dataset.drop(columns=["Capacitance (F/g)"])
y_test = test_dataset["Capacitance (F/g)"]

# Train models
results = train_all_models(X, y, X_test, y_test, model_registry)

# Save models
for res in results:
    if res["success"]:
        save_model(dataset_type, res["estimator"], res["model"])

# Summary
summary_df = generate_summary(results)
summary_df.to_csv(f"regression_benchmark/results/model_summary_{dataset_type}.csv", index=False)

# Feature importance
feature_df = aggregate_feature_importance(results, X)
feature_df.to_csv(f"regression_benchmark/results/feature_importance_{dataset_type}.csv", index=False)
