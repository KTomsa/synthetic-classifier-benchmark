import shap
from sklearnex import patch_sklearn
patch_sklearn()
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer, log_loss, matthews_corrcoef, recall_score, precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TunedThresholdClassifierCV, StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import xgboost as xgb
from shap import KernelExplainer
import time
from tqdm import tqdm
from . import data_generation as dg

class WoeBinning(BaseEstimator, TransformerMixin):
    """Třída pro binning a výpočet vážených odchylek (WOE) pro proměnné v datech.
    Používá se k transformaci číselných proměnných na WOE hodnoty, které jsou užitečné pro modely jako je logistická regrese.
    Vytváří biny na základě kvantilů a vypočítává WOE hodnoty pro každou binu.
    Třída implementuje metody fit, transform a fit_transform pro učení a aplikaci transformace na data.
    """
    def __init__(self, bins: int):
        self.bins = bins
        self.woe_dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.woe_dict = {}
        overall_goods_bads = y.loc[y==0].count()/y.loc[y==1].count()
        data = pd.concat([X, y], axis=1)
        variables_to_bin = data.select_dtypes(include=["float64"]).columns.tolist()
        quantiles = np.linspace(0, 1, self.bins+1)[1:-1]
        self.var_bin_edges = {}
        for variable in variables_to_bin:
            bin_edges = data[variable].quantile(quantiles)
            bin_edges = [-np.inf] + bin_edges.tolist() + [np.inf]
            self.var_bin_edges[variable] = bin_edges
            data[variable] = np.digitize(data[variable], bins=bin_edges)
            var_dist = data.groupby(by=[variable, data.columns[-1]], observed=False)[variable].count().unstack()
            var_dist.fillna(0, inplace=True)
            if np.any(var_dist == 0):
                var_dist.loc[np.any(var_dist == 0, axis=1),:] += 0.5
            var_dist["WOE"] = np.log((var_dist[0]/var_dist[1])/overall_goods_bads)
            self.woe_dict[variable] = var_dist["WOE"].to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        new_data = X.copy()
        if type(new_data) == np.ndarray:
            new_data = pd.DataFrame(new_data, columns=[f"X{i+1}" for i in range(new_data.shape[1])])
        for variable in self.woe_dict.keys():
            new_data[variable] = np.digitize(new_data[variable], bins=self.var_bin_edges[variable])
            new_data[variable] = new_data[variable].map(self.woe_dict[variable]).astype(float)
        return new_data

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
       return self.fit(X, y).transform(X)

def roc_optim(y_true, y_pred):
    """Funkce k výpočtu rozdílu mezi True Positive Rate a False Positive Rate pro binární klasifikaci.

    :param y_true: np.array - Pozorované hodnoty.
    :param y_pred: np.array - Predikce modelu."""
    tpr = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    fpr = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0)
    return tpr - fpr

def expected_calibration_error(y_true, y_pred, n_bins=5):
    """Funkce k výpočtu očekávané chyby kalibrace (kalibrace pravděpodobností).

    :param y_true: np.array - Pozorované hodnoty.
    :param y_pred: np.array - Predikce modelu.
    :param n_bins: int - Počet binů pro kalibraci."""

    bins = np.linspace(0.5, 1, n_bins + 1)

    y_pred_prob_all = np.array([y_pred, 1 - y_pred]).T
    y_pred_max = np.max(y_pred_prob_all, axis=1)

    bin_indices = np.digitize(y_pred_max, bins, right=True) - 1

    ece = 0.0
    for i in range(n_bins):
        in_bin = bin_indices == i
        if np.sum(in_bin) > 0:
            bin_accuracy = np.mean(y_true[in_bin] == (y_pred[in_bin] > 0.5))
            bin_confidence = np.mean(y_pred_max[in_bin])
            bin_weight = np.sum(in_bin) / len(y_pred)
            ece += bin_weight * abs(bin_confidence - bin_accuracy)

    return ece

def std_errors(X_train, y_pred):
    """Funkce k výpočtu standardních chyb pro predikce modelu.

    :param X_train: pd.DataFrame - Trénovací data.
    :param y_pred: np.array - Predikce modelu."""

    X_train_ones = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    W = np.zeros((X_train.shape[0], X_train.shape[0]))
    np.fill_diagonal(W, y_pred*(1 - y_pred))

    # Compute the standard errors
    # if singular matrix use pseudo inverse
    try:
        std_errors = np.sqrt(np.diag(np.linalg.inv((X_train_ones.T @ W) @ X_train_ones)))
    except:
        std_errors = np.sqrt(np.diag(np.linalg.pinv((X_train_ones.T @ W) @ X_train_ones)))

    return std_errors

def train_evaluate(scenario, hyperparmeter_grid: dict, n_data: int, n_iter: int):
    """Naučí logistickou regresi, XGBoost, náhodné lesy a neuronovou síť na datech pomocí náhodného prozkoumávání hyperparametrů.
    Poté najde nejlepší rozhodovací práh a vypočte evaluační a interpretační metriky pro každý model a každý dataset.

    :param scenario: Dict -
        Slovník s parametry pro generování dat (např. počet pozorování, korelační matice, atd.).

    :param hyperparmeter_grid: Dict -
        Slovník s hyperparametry pro modely (XGBoost, XGBoost (WOE), Náhodné lesy, ...).

    :param n_data: int -
        Počet datasetů, které se mají vygenerovat a na kterých se modely naučí.

    :param n_iter: int -
        Počet iterací pro náhodné prozkoumávání hyperparametrů.
    """

    #start = time.time()

    eval_metrics = {
        "Accuracy": {"LR": [], "XGBoost": [], "Náhodné lesy": [], "NN": [], "LR (WOE)": []},
        "Gini": {"LR": [], "XGBoost": [], "Náhodné lesy": [], "NN": [], "LR (WOE)": []},
        "LogLoss": {"LR": [], "XGBoost": [], "Náhodné lesy": [], "NN": [], "LR (WOE)": []},
        "MCC": {"LR": [], "XGBoost": [], "Náhodné lesy": [], "NN": [], "LR (WOE)": []},
        "Recall": {"LR": [], "XGBoost": [], "Náhodné lesy": [], "NN": [], "LR (WOE)": []},
        "Precision": {"LR": [], "XGBoost": [], "Náhodné lesy": [], "NN": [], "LR (WOE)": []},
        "ECE": {"LR": [], "XGBoost": [], "Náhodné lesy": [], "NN": [], "LR (WOE)": []}
    }

    shap_values = {"LR": [], "XGBoost": [], "Náhodné lesy": [], "NN": [], "LR (WOE)": []}

    effects = {
        "Coef": [],
        "Coef (WOE)": [],
        "STD": [],
        "STD (WOE)": []
    }

    shap_all = {"LR": [], "XGBoost": [], "Náhodné lesy": [], "NN": [], "LR (WOE)": []}
    shap_x_test = {"LR": [], "XGBoost": [], "Náhodné lesy": [], "NN": [], "LR (WOE)": []}

    best_hyperparams = {"XGBoost": [], "Náhodné lesy": [], "NN": []}

    threshold_scorer = make_scorer(roc_optim, greater_is_better=True)

    for i, data in enumerate(tqdm(dg.data_loop(n_data, scenario), desc="Running scenario", total=n_data, position=1,
                                  leave=False)):
        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["Target"]), data["Target"],
                                                            test_size=0.2, stratify=data["Target"], random_state=i)

        models = {"LR": LogisticRegression(penalty=None, random_state=i),
                  "LR (WOE)": LogisticRegression(penalty=None, random_state=i),
                  "XGBoost": xgb.XGBClassifier(random_state=i, n_jobs=None),
                  "Náhodné lesy": RandomForestClassifier(random_state=i),#, max_samples=rf_sample_size),
                  "NN": MLPClassifier(early_stopping=True, validation_fraction=0.2,
                                      n_iter_no_change=30, batch_size=32, max_iter=250, random_state=i)}

        for m, model in enumerate(models.keys()):

            steps = [("model", models[model])]

            # if j == 0 and model == "NN":
                # steps = [("scaler", StandardScaler())] + steps
            if model == "LR (WOE)":
                steps = [("scaler", WoeBinning(10))] + steps

            model_pipeline = Pipeline(steps)

            if model != "LR" and model != "LR (WOE)":
                search_cv = RandomizedSearchCV(model_pipeline, hyperparmeter_grid[model], n_iter=n_iter, scoring="roc_auc",
                                          cv=StratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=i),
                                               random_state=i, n_jobs=-1, refit=False)
                search_cv.fit(X_train, y_train)
                model_pipeline = model_pipeline.set_params(**search_cv.best_params_)

            threshold_tuner = TunedThresholdClassifierCV(model_pipeline, scoring=threshold_scorer,
                                                         cv=StratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=i), random_state=i)
            # if predictions are constant
            try:
                threshold_tuner.fit(X_train, y_train)
                model_estimator = threshold_tuner.estimator_
            except:
                print("Error in threshold tuning, using best model without tuning.")
                model_pipeline.fit(X_train, y_train)
                threshold_tuner = model_pipeline
                model_estimator = model_pipeline

            if model == "LR" or model == "LR (WOE)":
                tuned_estimator = model_estimator
                if model == "LR":
                    coef = tuned_estimator.named_steps["model"].coef_[0]
                    effects["Coef"].append(coef)
                    std_errors_coef = std_errors(X_train, tuned_estimator.predict_proba(X_train)[:, 1])
                    effects["STD"].append(std_errors_coef)
                if model == "LR (WOE)":
                    coef = tuned_estimator.named_steps["model"].coef_[0]
                    effects["Coef (WOE)"].append(coef)
                    std_errors_coef = std_errors(tuned_estimator.named_steps["scaler"].transform(X_train),
                                                 tuned_estimator.predict_proba(X_train)[:, 1])
                    effects["STD (WOE)"].append(std_errors_coef)

            n_explain = 50
            if X_test.shape[0] < n_explain:
                n_explain = X_test.shape[0]
            rand_obs = np.random.choice(X_test.shape[0], n_explain, replace=False)
            if model != "LR (WOE)":
                background_data = shap.kmeans(X_train, 10)
                explainer = KernelExplainer(lambda x: threshold_tuner.predict_proba(x), background_data, seed=i,
                                            feature_names=X_train.columns.tolist())
                shap_val_test = explainer.shap_values(X_test.values[rand_obs, :], silent=True)[:,:,1]
                shap_values[model].append(np.abs(shap_val_test).mean(axis=0))
                shap_all[model].append(shap_val_test)
                shap_x_test[model].append(X_test.values[rand_obs, :])
            else:
                background_data = shap.kmeans(model_estimator.named_steps["scaler"].transform(X_train), 10)
                explainer = KernelExplainer(lambda x: threshold_tuner.predict_proba(x), background_data, seed=i,
                                            feature_names=X_train.columns.tolist())
                shap_val_test = explainer.shap_values(model_estimator.named_steps["scaler"].
                                                      transform(X_test.values[rand_obs, :]), silent=True)[:, :, 1]
                shap_values[model].append(np.abs(shap_val_test).mean(axis=0))
                shap_all[model].append(shap_val_test)
                shap_x_test[model].append(X_test.values[rand_obs, :])

            y_pred = threshold_tuner.predict(X_test)
            y_pred_proba = threshold_tuner.predict_proba(X_test)[:, 1]
            positive_class_ratio = np.sum(y_train == 1) / len(y_train)

            # print(f"Gini: {2 * roc_auc_score(y_test, y_pred_proba) - 1}")
            eval_metrics["Accuracy"][model].append(np.mean(y_pred == y_test))
            eval_metrics["Gini"][model].append(2 * roc_auc_score(y_test, y_pred_proba) - 1)
            eval_metrics["LogLoss"][model].append(log_loss(y_test, y_pred_proba,
                sample_weight=[1 if y == 1 else positive_class_ratio/(1-positive_class_ratio) for y in y_test]))
            eval_metrics["MCC"][model].append(matthews_corrcoef(y_test, y_pred))
            eval_metrics["Recall"][model].append(recall_score(y_test, y_pred))
            eval_metrics["Precision"][model].append(precision_score(y_test, y_pred))
            eval_metrics["ECE"][model].append(expected_calibration_error(y_test, y_pred_proba))

            if model != "LR" and model != "LR (WOE)":
                best_hyperparams[model].append(search_cv.best_params_)

    return eval_metrics, effects, shap_values, best_hyperparams, shap_x_test, shap_all

