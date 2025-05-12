import pandas as pd

def generate_summary(results):
    records = []
    for res in results:
        records.append({
            "Model": res["model"],
            "Success": res["success"],
            "CV R²": res["score_mean_cv"],
            "Test R²": res["r2_test"],
            "Test RMSE": res["rmse_test"],
            "Test MAE": res["mae_test"],
            "Test MAPE": res["mape_test"],
            "Normalized RMSE": res["normalized_rmse_test"],
            "Best Hyperparameters": res["best_params"],
            "Error" if not res["success"] else "": res.get("error", "")
        })
    return pd.DataFrame(records).sort_values(by="Test R²", ascending=False, na_position="last")
