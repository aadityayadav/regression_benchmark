import os
import pandas as pd
from models.registry import model_registry, ensemble_model_registry
from training.trainer import train_all_models
from evaluation.summarize import generate_summary
from evaluation.feature_importance import aggregate_feature_importance
from utils.io import save_model

# 1. Raw test
# Update the dataset type
# dataset_type = "raw"

# # Load dataset
# data_dir = os.path.join(os.getcwd(), "data")
# train_dataset = pd.read_csv(os.path.join(data_dir, "train_raw.csv"))
# test_dataset = pd.read_csv(os.path.join(data_dir, "test_raw.csv"))

# X = train_dataset.drop(columns=["Capacitance (F/g)"])
# y = train_dataset["Capacitance (F/g)"]
# X_test = test_dataset.drop(columns=["Capacitance (F/g)"])
# y_test = test_dataset["Capacitance (F/g)"]

# # 2. Normalized test
# # Update the dataset type
# dataset_type = "scaled"

# # Load dataset
# data_dir = os.path.join(os.getcwd(), "data")
# train_dataset = pd.read_csv(os.path.join(data_dir, "train_scaled.csv"))
# test_dataset = pd.read_csv(os.path.join(data_dir, "test_scaled.csv"))

# X = train_dataset.drop(columns=["Capacitance (F/g)"])
# y = train_dataset["Capacitance (F/g)"]
# X_test = test_dataset.drop(columns=["Capacitance (F/g)"])
# y_test = test_dataset["Capacitance (F/g)"]

# 3. Normalized test with outliers removed
# Update the dataset type
dataset_type = "scaled_removed_outliers"

# Load dataset
data_dir = os.path.join(os.getcwd(), "data")
train_dataset = pd.read_csv(os.path.join(data_dir, "train_scaled_removed_outliers.csv"))
test_dataset = pd.read_csv(os.path.join(data_dir, "test_scaled_removed_outliers.csv"))

X = train_dataset.drop(columns=["Capacitance (F/g)"])
y = train_dataset["Capacitance (F/g)"]
X_test = test_dataset.drop(columns=["Capacitance (F/g)"])
y_test = test_dataset["Capacitance (F/g)"]

# Model Registry Experiments
# results = train_all_models(X, y, X_test, y_test, model_registry)

##################################################

# Ensemble Model Registry Experiments

# Train models
results = train_all_models(X, y, X_test, y_test, ensemble_model_registry)

# Save models
for res in results:
    if res["success"]:
        save_model(dataset_type, res["estimator"], res["model"])

# Summary
summary_df = generate_summary(results)
# summary_df.to_csv(os.path.join(os.getcwd(),f"results/model_summary_{dataset_type}.csv"), index=False)
summary_df.to_csv(os.path.join(os.getcwd(),f"results/ensemble_model_summary_{dataset_type}.csv"), index=False)

# Feature importance
feature_df = aggregate_feature_importance(results, X)
# feature_df.to_csv(os.path.join(os.getcwd(),f"results/feature_importance_{dataset_type}.csv"), index=False)
feature_df.to_csv(os.path.join(os.getcwd(),f"results/ensemble_feature_importance_{dataset_type}.csv"), index=False)
