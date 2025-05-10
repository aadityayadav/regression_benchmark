from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

model_registry = {
    "Ridge": {
        "model": Pipeline([("scaler", StandardScaler()), ("reg", Ridge())]),
        "params": {"reg__alpha": [0.1, 1.0, 10.0]}
    },
    "Lasso": {
        "model": Pipeline([("scaler", StandardScaler()), ("reg", Lasso())]),
        "params": {"reg__alpha": [0.01, 0.1, 1.0]}
    },
    "BayesianRidge": {
        "model": Pipeline([("scaler", StandardScaler()), ("reg", BayesianRidge())]),
        "params": {"reg__alpha_1": [1e-6, 1e-5], "reg__lambda_1": [1e-6, 1e-5]}
    },
    "KNN": {
        "model": Pipeline([("scaler", StandardScaler()), ("reg", KNeighborsRegressor())]),
        "params": {"reg__n_neighbors": [3, 5, 7]}
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(),
        "params": {"max_depth": [3, 5, 10]}
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {"n_estimators": [50, 100], "max_depth": [None, 10]}
    },
    "SVR": {
        "model": Pipeline([("scaler", StandardScaler()), ("reg", SVR())]),
        "params": {"reg__C": [1, 10], "reg__kernel": ["linear", "rbf"]}
    },
    "MLP": {
        "model": Pipeline([("scaler", StandardScaler()), ("reg", MLPRegressor(max_iter=1000))]),
        "params": {"reg__hidden_layer_sizes": [(50,), (100,)], "reg__alpha": [0.0001, 0.01]}
    },
}
