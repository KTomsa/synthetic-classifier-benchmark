from tqdm import tqdm
from . import algorithm_training as at
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import copy

matplotlib.use("Agg")

# Evaluation functions
def plot_coefficients(regression_coef, nonlinear=False, auto_corr=False, plot_true=True, save=False, path=None, scenario=None):
    real_coef_linear = [0.8, -0.6, 0.6, -0.9]
    real_coef_nonlinear = [1.3, -1.3, 0.1, -0.5]
    if nonlinear:
        real_coef = real_coef_nonlinear
    else:
        real_coef = real_coef_linear
    if auto_corr:
        real_coef.append(0.5)  # Adding the coefficient for the autocorrelation term
    num_coef = len(regression_coef[0])
    coef_names = [f"X{i+1}" for i in range(num_coef)]
    coefs = {}
    for name in coef_names:
        coefs[name] = []
    for iter in range(len(regression_coef)):
        for i, name in enumerate(coef_names):
            coefs[name].append(regression_coef[iter][i])
    data = pd.DataFrame(coefs)
    fig, ax = plt.subplots(2, 2, figsize=(20, 12))
    for i, name in enumerate(coef_names[:4]):
        sns.kdeplot(data=data, x=name, ax=ax[i//2, i%2])
        ax[i//2, i%2].set_ylabel("Hustota pravděpodobnosti")
        if plot_true:
            ax[i//2, i%2].axvline(x=real_coef[i], color="red", linestyle="--")
    if save:
        if path:
            plt.savefig(path+"coefficients_"+str(scenario)+".png")
            plt.close(fig)
    else:
        plt.show()
    # create separate plot for autocorrelation coefficient if it exists
    if auto_corr:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(data=data, x="X5", ax=ax)
        ax.set_ylabel("Hustota pravděpodobnosti")
        if plot_true:
            ax.axvline(x=real_coef[4], color="red", linestyle="--")
        if save:
            if path:
                plt.savefig(path+"coefficients_autocorr_"+str(scenario)+".png")
                plt.close(fig)
        else:
            plt.show()

def plot_metric(metric_list, metric_name, title, asc=False, save=False, path=None, scenario=None):
    data = pd.DataFrame(metric_list)
    data = data.melt(var_name="Model", value_name=metric_name)
    plt.figure(figsize=(20, 12))
    sns.boxplot(data=data, x="Model", y=metric_name, order=data.groupby("Model")[metric_name].median().sort_values(ascending=asc).index)
    plt.ylabel(title)
    # plt.title(title)
    if save:
        if path:
            plt.savefig(path+"metric_"+metric_name+"_"+str(scenario)+".png")
            plt.close()
    else:
        plt.show()

def metrics_to_csv(metrics, save=False, path=None, scenario=None):
    metric_dict = copy.deepcopy(metrics)
    for metric in metric_dict.keys():
        for model in metric_dict[metric].keys():
            metric_dict[metric][model] = np.round(np.median(metric_dict[metric][model]), 3)
    data = pd.DataFrame(metric_dict)
    data = data.reset_index(names=["Model"])
    data = data.round(3)
    data.columns = ["Model", "Správnost", "Gini", "WLL", "MCC", "Přesnost", "Senzitivita", "ECE"]
    print(data.to_latex(index=False, decimal=",", float_format="%.3f", column_format="l" + "c" * (len(data.columns)-1), escape=False, label="tab:metrics", caption="Hodnoty metrik"))
    if save:
        if path:
            data.to_csv(path+"metrics_"+str(scenario)+".csv", index=False)
            data_all = pd.DataFrame(metrics)
            data_all = data_all.explode(data_all.columns.tolist())
            data_all = data_all.reset_index(names=["Model"])
            data_all.to_csv(path+"metrics_all_"+str(scenario)+".csv", index=False)

def avg_bias(coef, nonlinear=False, auto_corr=False, save=False, path=None, scenario=None):
    true_coef_linear = np.array([0.8, -0.6, 0.6, -0.9])
    true_coef_nonlinear = np.array([1.3, -1.3, 0.1, -0.5])
    if nonlinear:
        true_coef = true_coef_nonlinear
    else:
        true_coef = true_coef_linear
    if auto_corr:
        true_coef = np.append(true_coef, 0.5)  # Adding the coefficient for the autocorrelation term
    est_coef = np.vstack(coef)
    est_coef_avg = np.mean(est_coef, axis=0)
    est_coef_ci_low, est_coef_ci_high = np.quantile(est_coef, [0.025, 0.975], axis=0)
    abs_bias = true_coef - est_coef_avg
    rel_bias = np.abs(abs_bias) / np.abs(true_coef)
    mse = np.mean((true_coef - est_coef) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_coef - est_coef), axis=0)
    print(f"Absolutní bias: {abs_bias}")
    print(f"Relativní bias: {rel_bias}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    if save:
        if path:
            coef_names = [f"X{i + 1}" for i in range(len(true_coef))]
            data = pd.DataFrame({"Scénář": [scenario for i in range(len(true_coef))], "Prediktor": coef_names, "2,5% CI": est_coef_ci_low,
                                 "97,5% CI": est_coef_ci_high, "Průměr koef.": est_coef_avg, "Absolutní bias": abs_bias,
                                 "Relativní bias": rel_bias, "MSE": mse, "RMSE": rmse, "MAE": mae})
            data.to_csv(path+"bias_"+str(scenario)+".csv", index=False)

def std_bias(coef, std_err, save=False, path=None, scenario=None):
    est_coef = np.vstack(coef)
    est_err = np.vstack(std_err)[:,1:]
    est_err_mean = np.mean(est_err, axis=0)
    est_coef_std = np.std(est_coef, axis=0)
    abs_bias = est_coef_std - est_err_mean
    rel_bias = np.abs(abs_bias) / np.abs(est_coef_std)
    mse = np.mean((est_coef_std - est_err) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(est_coef_std - est_err), axis=0)
    print(f"Absolutní bias: {abs_bias}")
    print(f"Relativní bias: {rel_bias}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    if save:
        if path:
            coef_names = [f"X{i + 1}" for i in range(est_coef.shape[1])]
            data = pd.DataFrame({"Scénář": [scenario for i in range(est_coef.shape[1])], "Prediktor": coef_names,
                                 "Prům. σ koef.": est_err_mean, "σ koef.": est_coef_std, "Absolutní bias": abs_bias,
                                 "Relativní bias": rel_bias, "MSE": mse, "RMSE": rmse, "MAE": mae})
            data.to_csv(path+"bias_std_"+str(scenario)+".csv", index=False)

def plot_shap_values(shap_values, auto_corr=False, save=False, path=None, scenario=None):
    num_coef = len(shap_values["XGBoost"][0])
    coef_names = [f"X{i + 1}" for i in range(num_coef)]
    shap = {"Model": []}
    for name in coef_names:
        shap[name] = []
    for model in shap_values.keys():
        for iter in range(len(shap_values["XGBoost"])):
            for i, name in enumerate(coef_names):
                shap[name].append(shap_values[model][iter][i])
            shap["Model"].append(model)
    data = pd.DataFrame(shap)
    fig, ax = plt.subplots(2, 2, sharey=True, figsize=(20, 12))
    for i, name in enumerate(coef_names[:4]):
        sns.boxplot(data=data, x="Model", y=name, ax=ax[i//2, i%2])
        ax[i//2, i%2].set_title(name)
        ax[i//2, i%2].set_ylabel("SHAP hodnota")
    if save:
        if path:
            plt.savefig(path+"shap_"+str(scenario)+".png")
            plt.close(fig)
    else:
        plt.show()
    if auto_corr:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data, x="Model", y="X5", ax=ax)
        ax.set_title("X5 (Autocorrelation term)")
        ax.set_ylabel("SHAP hodnota")
        if save:
            if path:
                plt.savefig(path+"shap_autocorr_"+str(scenario)+".png")
                plt.close(fig)
        else:
            plt.show()

def plot_hyperparams(metric, model_params, model, ylabel):
    data = pd.DataFrame(metric)
    data = data[[model]]
    model_params = model_params[model]
    hyperparams = list(model_params[0].keys())
    param_val = {}
    for param in hyperparams:
        param_val[param] = []
    for i, dataset in enumerate(model_params):
        for param in hyperparams:
            param_val[param].append(dataset[param])
    data = data.merge(pd.DataFrame(model_params), left_index=True, right_index=True)
    data = data.melt(id_vars=[model], var_name="Hyperparameter", value_name="value")
    fig, ax = plt.subplots(len(data["Hyperparameter"].unique())//3+1, 3)
    hyperparams_uni = data["Hyperparameter"].unique()
    for i, param in enumerate(hyperparams_uni):
        sns.scatterplot(data=data.loc[data["Hyperparameter"]==param,:], x="value", y=model, ax=ax[i//3, i%3])
        ax[i//3, i%3].set_title(param)
        ax[i//3, i%3].set_ylabel(ylabel)
        ax[i//3, i%3].set_xlabel(param)
    plt.show()

def save_hyperparams(model_params, path=None, scenario=None):
    if path:
        for model in model_params.keys():
            data = pd.DataFrame(model_params[model])
            data.to_csv(f"{path}hyperparams_{model}_{str(scenario)}.csv", index=False)

def plot_shap_vs_real(shap_dict, real_values_dict, model_name, auto_corr=False, save=False, path=None, scenario=None):

    shap_values_list = shap_dict[model_name]
    real_values_list = real_values_dict[model_name]

    shap_values = np.vstack(shap_values_list)
    real_values = np.vstack(real_values_list)

    num_features = shap_values.shape[1]

    feature_names = [f'X{i + 1}' for i in range(num_features)]

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    # fig.suptitle(model_name)

    for i in range(num_features-1 if auto_corr else num_features):
        sns.scatterplot(x=real_values[:, i], y=shap_values[:, i], ax=axes[i // 2, i % 2], alpha=0.3)
        axes[i // 2, i % 2].set_xlabel(feature_names[i])
        axes[i // 2, i % 2].set_ylabel(f'SHAP hodnota')
        axes[i // 2, i % 2].set_title(feature_names[i])

    plt.tight_layout()
    if save:
        if path:
            plt.savefig(f"{path}shap_real_{model_name}_{str(scenario)}.png")
            plt.close(fig)
    else:
        plt.show()
    # create separate plot for autocorrelation coefficient if it exists
    if auto_corr:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=real_values[:, -1], y=shap_values[:, -1], ax=ax, alpha=0.3)
        ax.set_xlabel('X5 (Autocorrelation term)')
        ax.set_ylabel('SHAP hodnota')
        ax.set_title('X5 (Autocorrelation term)')
        if save:
            if path:
                plt.savefig(f"{path}shap_real_autocorr_{model_name}_{str(scenario)}.png")
                plt.close(fig)
        else:
            plt.show()

def save_coefficients(coef, save=False, path=None, scenario=None):
    coefs = np.vstack(coef)
    coef_names = [f"X{i + 1}" for i in range(coefs.shape[1])]
    data = pd.DataFrame(coefs, columns=coef_names)
    if save:
        if path:
            data.to_csv(path + "coeff_" + str(scenario) + ".csv", index=False)

def save_std_err(std_err, save=False, path=None, scenario=None):
    std_err = np.vstack(std_err)
    coef_names = [f"X{i}" for i in range(std_err.shape[1])]
    data = pd.DataFrame(std_err, columns=coef_names)
    if save:
        if path:
            data.to_csv(path + "std_err_" + str(scenario) + ".csv", index=False)

def save_shap_values(shap_values, x_values, save=False, path=None, scenario=None):
    shap_dict = {}
    x_dict = {}
    for model in shap_values.keys():
        shap_dict[model] = np.vstack(shap_values[model])
        coef_names = [f"X{i + 1}" for i in range(shap_dict[model].shape[1])]
        data_shap = pd.DataFrame(shap_dict[model], columns=coef_names)
        x_dict[model] = np.vstack(x_values[model])
        data_x = pd.DataFrame(x_dict[model], columns=coef_names)
        if save:
            if path:
                data_shap.to_csv(path + "shap_" + model + "_" + str(scenario) + ".csv", index=False)
                data_x.to_csv(path + "x_values_" + model + "_" + str(scenario) + ".csv", index=False)


#######################################################################################################
#######################################################################################################

def run_scenarios(scenarios_list: list, scenarios: dict, n_scenarios: int, grid: dict, n_iter_hyperparams: int, path: str = None):
    """Runs selected scenarios from the scenario dictionary.

    :param scenarios_list: List of scenario numbers to run
    :param scenarios: Dictionary containing all scenarios
    :param n_scenarios: Number of datasets to generate
    :param grid: Hyperparameter grid for model training
    :param n_iter_hyperparams: Number of iterations for hyperparameter random search optimization
    :param path: Path to save the results
    """
    for scenario in tqdm(scenarios_list, position=0, leave=False, desc="Running selected scenarios"):

        try:
            nonlinear = scenarios["Scénář "+str(scenario)]["nonlinear"]
        except:
            nonlinear = False

        try:
            auto_corr = scenarios["Scénář "+str(scenario)]["auto_corr"]
        except:
            auto_corr = False

        eval_metrics, effects, shap_values, best_hyperparams, shap_x_test, shap_all = at.train_evaluate(
            scenarios["Scénář "+str(scenario)], grid, n_scenarios, n_iter_hyperparams)

        plot_coefficients(effects["Coef"], nonlinear, auto_corr, True, True, path, scenario)

        avg_bias(effects["Coef"], nonlinear, auto_corr, True, path, scenario)
        std_bias(effects["Coef"], effects["STD"], True, path, scenario)

        plot_metric(eval_metrics["Accuracy"], "Accuracy", "Správnost", False, True, path, scenario)
        plot_metric(eval_metrics["Gini"], "Gini", "Gini", False, True, path, scenario)
        plot_metric(eval_metrics["LogLoss"], "LogLoss", "WLL", True, True, path, scenario)
        plot_metric(eval_metrics["MCC"], "MCC", "MCC", False, True, path, scenario)
        plot_metric(eval_metrics["Precision"], "Precision", "Přesnost", False, True, path, scenario)
        plot_metric(eval_metrics["Recall"], "Recall", "Senzitivita", False, True, path, scenario)
        plot_metric(eval_metrics["ECE"], "ECE", "ECE", True, True, path, scenario)

        metrics_to_csv(eval_metrics, True, path, scenario)

        plot_shap_values(shap_values, auto_corr, True, path, scenario)

        plot_shap_vs_real(shap_all, shap_x_test, "LR", auto_corr, True, path, scenario)
        plot_shap_vs_real(shap_all, shap_x_test, "XGBoost", auto_corr, True, path, scenario)
        plot_shap_vs_real(shap_all, shap_x_test, "Náhodné lesy", auto_corr, True, path, scenario)
        plot_shap_vs_real(shap_all, shap_x_test, "NN", auto_corr, True, path, scenario)
        plot_shap_vs_real(shap_all, shap_x_test, "LR (WOE)", auto_corr, True, path, scenario)

        save_coefficients(effects["Coef"], True, path, scenario)
        save_std_err(effects["STD"], True, path, scenario)
        save_shap_values(shap_all, shap_x_test, True, path, scenario)

        save_hyperparams(best_hyperparams, path, scenario)



