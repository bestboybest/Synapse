from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
import numpy as np
import joblib

models = {
#    "LogisticRegression": {
#       "pipeline": Pipeline([
#            ("scaler", StandardScaler()),
#            ("clf", LogisticRegression(max_iter=500))
#        ]),
#        "param_grid": {
#            "scaler": [StandardScaler(), RobustScaler()],
#            "clf__C": [0.01, 0.1, 1, 10]
#        }
#    },

    "KNN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        "param_grid": {
            "scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()],
            "clf__n_neighbors": [3, 5, 7, 11],
            "clf__weights": ["uniform", "distance"]
        }
    },

    "RandomForest": {
        "pipeline": RandomForestClassifier(),
        "param_grid": {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    },

    "GradientBoosting": {
        "pipeline": GradientBoostingClassifier(),
        "param_grid": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }
    },

    "LinearSVM": {
        "pipeline": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(dual=False, max_iter=5000))
         ]),
        "param_grid": {
        "scaler": [StandardScaler(), RobustScaler()],
        "clf__C": [0.01, 0.1, 1, 10, 100]
    },

    "SVM": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC())
        ]),
        "param_grid": {
            "scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()],
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", 0.01, 0.1],
            "clf__kernel": ["rbf"]
        }
    },
}
}

data = np.load("/content/drive/MyDrive/synapse/features.npz")
x = data["x"]
y = data["y"]
groups = data["groups"]

gkf = GroupKFold(n_splits = 5)

results = {}

for name, cfg in models.items():
    print("\n==============================")
    print(f"Training model: {name}", flush = True)
    print("==============================")

    grid = GridSearchCV(
        estimator=cfg["pipeline"],
        param_grid=cfg["param_grid"],
        cv=gkf,
        scoring="f1_macro",
        n_jobs=-1,     
        verbose=2
    )

    grid.fit(x, y, groups=groups)

    results[name] = {
        "best_score": grid.best_score_,
        "best_params": grid.best_params_,
        "best_estimator": grid.best_estimator_
    }

    joblib.dump(results, "/content/drive/MyDrive/synapse/gridsearch_partial.joblib")

    print(f"Best Macro F1: {grid.best_score_:.4f}")
    print(f"Best Params: {grid.best_params_}")


#Final results
#GradientBoosting: 0.4872 
#RandomForest: 0.4750
#LogisticRegression: 0.4307
#KNN: 0.4260
#LinearSVM: 0.4189
# (ran out of time to test SVM but analysis shows that SVM might not be so effective)