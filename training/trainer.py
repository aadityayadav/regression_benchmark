from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import traceback

def train_all_models(X_train, y_train, X_test, y_test, model_registry, scoring="r2", cv=5, n_jobs=-1):
    results = []
    for name, config in model_registry.items():
        print(f"Training: {name}")
        model = config["model"]
        param_grid = config["params"]
        try:
            search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)

            results.append({
                "model": name,
                "success": True,
                "score_mean_cv": search.best_score_,
                "best_params": search.best_params_,
                "estimator": best_model,
                "r2_test": r2_score(y_test, y_pred),
                "rmse_test": mean_squared_error(y_test, y_pred, squared=False),
                "mae_test": mean_absolute_error(y_test, y_pred)
            })
        except Exception as e:
            print(f"[{name}] failed: {e}")
            traceback.print_exc()
            results.append({
                "model": name,
                "success": False,
                "score_mean_cv": None,
                "best_params": {},
                "estimator": None,
                "r2_test": None,
                "rmse_test": None,
                "mae_test": None,
                "error": str(e)
            })
    return results
