import pandas as pd
import shap

def aggregate_feature_importance(results, X):
    all_importance = []

    for res in results:
        if not res["success"] or res["estimator"] is None:
            continue
        model = res["estimator"]
        try:
            explainer = shap.Explainer(model.predict, X)
            shap_vals = explainer(X)
            mean_importance = shap_vals.abs.mean(0).values
            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Mean Importance": mean_importance,
                "Model": res["model"]
            })
            all_importance.append(importance_df)
        except Exception as e:
            print(f"SHAP failed for {res['model']}: {e}")
            continue

    return pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()
