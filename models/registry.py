from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor

model_registry = {
    "Ridge": {
        "model": Ridge(),
        "params": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    },
    "Lasso": {
        "model": Lasso(),
        "params": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    },
    "BayesianRidge": {
        "model": BayesianRidge(),
        "params": {
            "alpha_1": [1e-6, 1e-5, 1e-4],
            "lambda_1": [1e-6, 1e-5, 1e-4]
        }
    },
    "KNN": {
        "model": KNeighborsRegressor(),
        "params": {
            "n_neighbors": [3, 5, 7, 9]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(),
        "params": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None]
        }
    },
    "ExtraTrees": {
        "model": ExtraTreesRegressor(),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5]
        }
    },
    "AdaBoost": {
        "model": AdaBoostRegressor(),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1]
        }
    },
    "HistGradientBoosting": {
        "model": HistGradientBoostingRegressor(),
        "params": {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_iter": [100, 200],
            "max_depth": [3, 5, None]
        }
    },
    "SVR": {
        "model": SVR(),
        "params": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"]
        }
    },
    "MLP": {
        "model": MLPRegressor(
            max_iter=5000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        ),
        "params": {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(verbosity=0),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }
    },
    # "LightGBM": {
    #     "model": LGBMRegressor(),
    #     "params": {
    #         "n_estimators": [50, 100, 200],
    #         "learning_rate": [0.01, 0.05, 0.1],
    #         "num_leaves": [31, 64],
    #         "min_child_samples": [5, 10],
    #         "force_col_wise": True
    #     }
    # }
}


ensemble_model_registry = {
    "Stacking_RidgeMeta": {
        "model": StackingRegressor(
            estimators=[
                ("ridge", Ridge(alpha=10.0)),
                ("extratrees", ExtraTreesRegressor(n_estimators=100, max_depth=None)),
                ("gboost", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3))
            ],
            final_estimator=Ridge(alpha=10.0),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        "params": {
            "final_estimator__alpha": [0.1, 1.0, 10.0, 100.0]
        }
    },
    "Stacking_LassoMeta": {
        "model": StackingRegressor(
            estimators=[
                ("ridge", Ridge(alpha=10.0)),
                ("extratrees", ExtraTreesRegressor(n_estimators=100, max_depth=None)),
                ("gboost", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3))
            ],
            final_estimator=Lasso(alpha=1.0),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        "params": {
            "final_estimator__alpha": [0.01, 0.1, 1.0, 10.0]
        }
    },
    "Stacking_BayesianRidgeMeta": {
        "model": StackingRegressor(
            estimators=[
                ("ridge", Ridge(alpha=10.0)),
                ("extratrees", ExtraTreesRegressor(n_estimators=100, max_depth=None)),
                ("gboost", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3))
            ],
            final_estimator=BayesianRidge(alpha_1=1e-6, lambda_1=1e-6),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        "params": {
            "final_estimator__alpha_1": [1e-6, 1e-5],
            "final_estimator__lambda_1": [1e-6, 1e-5]
        }
    },
    "Stacking_MLPMeta": {
        "model": StackingRegressor(
            estimators=[
                ("ridge", Ridge(alpha=10.0)),
                ("extratrees", ExtraTreesRegressor(n_estimators=100, max_depth=None)),
                ("gboost", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3))
            ],
            final_estimator=MLPRegressor(max_iter=2000, early_stopping=True, validation_fraction=0.1),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        "params": {
            "final_estimator__hidden_layer_sizes": [(50,), (100,)],
            "final_estimator__alpha": [0.0001, 0.001]
        }
    }
}