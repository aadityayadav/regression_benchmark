from sklearn.linear_model import Ridge, Lasso, BayesianRidge, ElasticNet
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
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline


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
            "n_estimators": [10, 20, 30, 40, 50, 60, 80, 100, 125, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [1, 2, 3, 4, 5, 6, 7]
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

# Shared GBBoost hyperparameters
gboost_param_grid = {
    "gboost__gradientboostingregressor__n_estimators": [20, 30, 40, 50, 60, 80],
    "gboost__gradientboostingregressor__learning_rate": [0.01, 0.05, 0.1],
    "gboost__gradientboostingregressor__max_depth": [1, 2, 3, 4, 5, 6, 7, 8]
}

ensemble_model_registry = {
    "Stacking_RidgeMeta": {
        "model": StackingRegressor(
            estimators=[
                ("ridge", make_pipeline(Ridge())),
                ("extratrees", make_pipeline(ExtraTreesRegressor())),
                ("gboost", make_pipeline(GradientBoostingRegressor()))
            ],
            final_estimator=Ridge(),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        "params": {
            "final_estimator__alpha": [0.1, 1.0, 10.0, 100.0],
            "ridge__ridge__alpha": [0.1, 1.0, 10.0, 100.0],
            "extratrees__extratreesregressor__n_estimators": [50, 100, 200],
            "extratrees__extratreesregressor__max_depth": [3, 5, 10, None],
            **gboost_param_grid
        }
    },
    "Stacking_LassoMeta": {
        "model": StackingRegressor(
            estimators=[
                ("ridge", make_pipeline(Ridge())),
                ("extratrees", make_pipeline(ExtraTreesRegressor())),
                ("gboost", make_pipeline(GradientBoostingRegressor()))
            ],
            final_estimator=Lasso(),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        "params": {
            "final_estimator__alpha": [0.01, 0.1, 1.0, 10.0],
            "ridge__ridge__alpha": [0.1, 1.0, 10.0, 100.0],
            "extratrees__extratreesregressor__n_estimators": [50, 100, 200],
            "extratrees__extratreesregressor__max_depth": [3, 5, 10, None],
            **gboost_param_grid
        }
    },
    "Stacking_BayesianRidgeMeta": {
        "model": StackingRegressor(
            estimators=[
                ("ridge", make_pipeline(Ridge())),
                ("extratrees", make_pipeline(ExtraTreesRegressor())),
                ("gboost", make_pipeline(GradientBoostingRegressor()))
            ],
            final_estimator=BayesianRidge(),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        "params": {
            "final_estimator__alpha_1": [1e-6, 1e-5, 1e-4],
            "final_estimator__lambda_1": [1e-6, 1e-5, 1e-4],
            "ridge__ridge__alpha": [1.0, 10.0, 100.0],
            "extratrees__extratreesregressor__n_estimators": [100, 200],
            **gboost_param_grid
        }
    },
    "Stacking_MLPMeta": {
        "model": StackingRegressor(
            estimators=[
                ("ridge", make_pipeline(Ridge())),
                ("extratrees", make_pipeline(ExtraTreesRegressor())),
                ("gboost", make_pipeline(GradientBoostingRegressor()))
            ],
            final_estimator=MLPRegressor(max_iter=3000, early_stopping=True, validation_fraction=0.1),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        "params": {
            "final_estimator__hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "final_estimator__alpha": [0.0001, 0.001, 0.01],
            "ridge__ridge__alpha": [10.0],
            "extratrees__extratreesregressor__n_estimators": [100],
            **gboost_param_grid
        }
    },
    "Stacking_ElasticNetMeta": {
        "model": StackingRegressor(
            estimators=[
                ("ridge", make_pipeline(Ridge())),
                ("extratrees", make_pipeline(ExtraTreesRegressor())),
                ("gboost", make_pipeline(GradientBoostingRegressor()))
            ],
            final_estimator=ElasticNet(),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        "params": {
            "final_estimator__alpha": [0.01, 0.1, 1.0],
            "final_estimator__l1_ratio": [0.1, 0.5, 0.9],
            "ridge__ridge__alpha": [1.0, 10.0],
            **gboost_param_grid
        }
    },
    "Stacking_DiverseBaseModels": {
        "model": StackingRegressor(
            estimators=[
                ("ridge", make_pipeline(Ridge())),
                ("svr", make_pipeline(SVR())),
                ("gboost", make_pipeline(GradientBoostingRegressor())),
                ("mlp", make_pipeline(MLPRegressor(max_iter=3000, early_stopping=True)))
            ],
            final_estimator=Ridge(),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        "params": {
            "final_estimator__alpha": [1.0, 10.0],
            "svr__svr__C": [10, 50, 100],
            "svr__svr__kernel": ["rbf"],
            "mlp__mlpregressor__hidden_layer_sizes": [(50,), (100,)],
            "mlp__mlpregressor__alpha": [0.001, 0.01],
            **gboost_param_grid
        }
    },
    "Stacking_CatBoostMeta": {
        "model": StackingRegressor(
            estimators=[
                ("ridge", make_pipeline(Ridge())),
                ("extratrees", make_pipeline(ExtraTreesRegressor())),
                ("gboost", make_pipeline(GradientBoostingRegressor()))
            ],
            final_estimator=CatBoostRegressor(verbose=0),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        "params": {
            "final_estimator__learning_rate": [0.01, 0.05],
            "final_estimator__depth": [3, 5],
            "final_estimator__iterations": [100, 200],
            "ridge__ridge__alpha": [10.0],
            **gboost_param_grid
        }
    }
}