import numpy as np
from scipy.stats import loguniform
import algorithm as alg
import warnings
import os
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

path = "C:/Users/kryst/OneDrive/Documents/Škola/Vysoká/bakalarska_prace/results/local_fifth_run/"

corr_m = np.array([[1,0.2,0.7,0.2,0],
                   [0.2,1,0.2,0.7,0],
                   [0.7,0.2,1,0.2,0],
                   [0.2,0.7,0.2,1,0],
                   [0,0,0,0,1]])

corr = np.array([[1,0,0,0,0],
                 [0,1,0,0,0],
                 [0,0,1,0,0],
                 [0,0,0,1,0],
                 [0,0,0,0,1]])

n1 = 50
n2 = 500
n3 = 5000

# přidat vynechanou proměnnou nekorelovanou s ostatními prediktory (zatím s normální distribucí,
# později možná s logistickou)
# přidat nenormálního rozdělení prediktorů (chi-square dist., uniform dist.)
# přidat naive gaussian bayes
scenarios = {"Scénář 1": {"n_obs": n1, "corr_matrix": corr, "positive_class_ratio": 0.3},
             "Scénář 2": {"n_obs": n2, "corr_matrix": corr, "positive_class_ratio": 0.3},
             "Scénář 3": {"n_obs": n3, "corr_matrix": corr, "positive_class_ratio": 0.3},
             "Scénář 4": {"n_obs": n1, "corr_matrix": corr_m, "positive_class_ratio": 0.3},
             "Scénář 5": {"n_obs": n2, "corr_matrix": corr_m, "positive_class_ratio": 0.3},
             "Scénář 6": {"n_obs": n3, "corr_matrix": corr_m, "positive_class_ratio": 0.3},
             "Scénář 7": {"n_obs": n1, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "auto_corr": True},
             "Scénář 8": {"n_obs": n2, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "auto_corr": True},
             "Scénář 9": {"n_obs": n3, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "auto_corr": True},
             "Scénář 10": {"n_obs": n1, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "noise_norm": True},
             "Scénář 11": {"n_obs": n2, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "noise_norm": True},
             "Scénář 12": {"n_obs": n3, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "noise_norm": True},
             "Scénář 13": {"n_obs": n1, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "omitted_var": True},
             "Scénář 14": {"n_obs": n2, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "omitted_var": True},
             "Scénář 15": {"n_obs": n3, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "omitted_var": True},
             "Scénář 16": {"n_obs": n1, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "endogeneity": True},
             "Scénář 17": {"n_obs": n2, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "endogeneity": True},
             "Scénář 18": {"n_obs": n3, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "endogeneity": True},
             "Scénář 19": {"n_obs": n1, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "heteroskedasticity": True},
             "Scénář 20": {"n_obs": n2, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "heteroskedasticity": True},
             "Scénář 21": {"n_obs": n3, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "heteroskedasticity": True},
             "Scénář 22": {"n_obs": n1, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "nonnormal_features": True},
             "Scénář 23": {"n_obs": n2, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "nonnormal_features": True},
             "Scénář 24": {"n_obs": n3, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "nonnormal_features": True},
             "Scénář 25": {"n_obs": n1, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "nonlinear": True},
             "Scénář 26": {"n_obs": n2, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "nonlinear": True},
             "Scénář 27": {"n_obs": n3, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "nonlinear": True},
             "Scénář 28": {"n_obs": n1, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "auto_corr": True,
                           "noise_norm": True, "endogeneity": True, "heteroskedasticity": True, "nonlinear": True,
                           "omitted_var": True, "nonnormal_features": True},
             "Scénář 29": {"n_obs": n2, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "auto_corr": True,
                           "noise_norm": True, "endogeneity": True, "heteroskedasticity": True, "nonlinear": True,
                           "omitted_var": True, "nonnormal_features": True},
             "Scénář 30": {"n_obs": n3, "corr_matrix": corr_m, "positive_class_ratio": 0.3, "auto_corr": True,
                           "noise_norm": True, "endogeneity": True, "heteroskedasticity": True, "nonlinear": True,
                           "omitted_var": True, "nonnormal_features": True},
}

#######################################################################################################
#######################################################################################################
# Hyperparameter grid
grid = {"XGBoost": {"model__n_estimators": np.arange(50,350,5), "model__max_depth": np.arange(2,20,1),
                    "model__learning_rate": np.linspace(0.03,0.3,30), "model__lambda": loguniform(1e-5, 1),
                    "model__subsample": np.linspace(0.8,1,20), "model__colsample_bytree": np.linspace(0.8,1,20)},
        "Náhodné lesy": {"model__n_estimators": np.arange(100,400,20), "model__max_depth": np.arange(3,25,1),
                         "model__max_features": ["sqrt", "log2"], "model__criterion": ["gini","entropy","log_loss"]},
        "NN": {"model__hidden_layer_sizes": [(1,),(2,),(4,),(8,),(16,),(2,2),(4,4),(8,8),(16,16),(4,4,4),(8,8,8)],
                "model__activation": ["logistic", "tanh", "relu"], "model__alpha": loguniform(1e-5, 1),
                "model__learning_rate_init": loguniform(1e-4, 1e-1)},
        "LR (WOE)": {"scaler__bins": np.arange(5, 13, 1)}}

n_scenarios = 500
n_iter_hyperparams = 100

alg.run_scenarios([9,12], scenarios, n_scenarios, grid, n_iter_hyperparams, path)


# jeden čistý case, jinak všude multikolinearita (u 2 vysoká, jinak 0.2-0.4)
# porušení jednoho předpokladu s multikolinearitou
# nejhorší case
# shap nedělat pro LR
# zkusit shap s multikolinearitou
# zkusit běh s 5000 - 3,1 h
# přidat heteroskedasticitu
# upravit ECE binování