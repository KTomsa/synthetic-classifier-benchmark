import numpy as np
import pandas as pd

def generate_data(n_obs: int, corr_matrix: np.array, noise_norm: bool = False, endogeneity: bool = False,
                  auto_corr: bool = False, heteroskedasticity: bool = False, positive_class_ratio: float = 0.5,
                  nonlinear: bool = False, omitted_var: bool = False, nonnormal_features: bool = False,
                  nonlinear_predictor: bool = False, noncausal_predictor: bool = False,
                  random_state: int = None) -> pd.DataFrame:
    """
    Generates a dataset with a binary target variable and four features. The target variable is generated based on
    a linear or non-linear function of the features, with the option to introduce heteroskedasticity, endogeneity,
    autocorrelation, and noise.

    :param n_obs: number of observations to generate
    :param corr_matrix: correlation matrix for the features
    :param noise_norm: if True, generates noise from a normal distribution; if False, generates noise from a logistic distribution
    :param endogeneity: introduces endogeneity in the model
    :param auto_corr: introduces autocorrelation in the third feature
    :param heteroskedasticity: introduces heteroskedasticity in the noise
    :param positive_class_ratio: ratio of positive class in the target variable
    :param nonlinear: if True, generates a non-linear target function; if False, generates a linear target function
    :param omitted_var: if True, generates feature that is used in target function, but is later omitted
    :param nonnormal_features: if True, generates the first two features from a chi-square distribution
    :param random_state: random seed for reproducibility
    :return: a pandas DataFrame containing the generated dataset
    """

    if random_state is not None:
        np.random.seed(random_state)


    std_array = np.array([1, 1, 1, 1, 1, 1])
    mean_array = np.array([0, 0, 0, 0, 0, 0])

    cov_matrix = np.outer(std_array, std_array) * corr_matrix

    # features
    predictors = [np.random.multivariate_normal(mean=mean_array, cov=cov_matrix, size=n_obs)]

    generated_data = np.column_stack(predictors)

    if nonnormal_features:
        generated_data[:, 0] = np.random.chisquare(1, size=n_obs) - 1  # shift to have mean 0
        generated_data[:, 1] = -np.random.uniform(-np.sqrt(3), np.sqrt(3), size=n_obs)  # uniform distribution with mean 0 and variance 1

    # noise (normal dist. violation of the standard logistic dist. assumption)
    if noise_norm:
        noise = np.random.normal(0, np.sqrt(np.pi**2/3), size=n_obs)
    else:
        noise = np.random.logistic(0, 1, size=n_obs)

    # heteroskedasticity (violation of the homoscedasticity assumption)
    if heteroskedasticity:
        lin_multiplier = np.linspace(0.5, 1.5, n_obs)
        for i in range(n_obs):
            noise[i] = noise[i] * lin_multiplier[i]

    # endogeneity (violation of the exogeneity assumption)
    if endogeneity:
        generated_data[:, 2] = generated_data[:, 2] + 0.2 * noise
        generated_data[:, 2] = (generated_data[:, 2] - np.mean(generated_data[:, 2])) / np.std(generated_data[:, 2])

    # non-linear target function (violation of the linearity in parameters assumption and model specification)
    if nonlinear:
        target_fce = np.abs(1.3 * generated_data[:, 0]) + np.sin(-1.3 * generated_data[:, 1]) +\
                     0.1 * generated_data[:, 2] - 0.5 * generated_data[:, 3]
    # linear target function
    else:
        target_fce = 0.8 * generated_data[:, 0] - 0.6 * generated_data[:, 1] + 0.6 * generated_data[:, 2] - \
                 0.9 * generated_data[:, 3]

    # add omitted variable to target function
    if omitted_var:
        target_fce = target_fce + 0.15 * generated_data[:, 5]

    if nonlinear_predictor:
        target_fce = target_fce - 0.1 * generated_data[:, 3]**2

    # autocorrelation in regressors (violation of the random sampling assumption)
    if auto_corr:
        for i in range(1, n_obs):
            target_fce[i] = 0.5 * target_fce[i - 1] + target_fce[i]

    # threshold
    if np.quantile(target_fce, 1-positive_class_ratio) < 0:
        threshold_adj = target_fce - np.quantile(target_fce, 1-positive_class_ratio)
    else:
        threshold_adj = target_fce - np.quantile(target_fce, 1-positive_class_ratio)

    target_fce = threshold_adj + noise

    # application of decision boundary
    target = (target_fce > 0).astype(int)

    # omit the last variable (variable must be omitted in all cases)
    generated_data = generated_data[:, :-1]  # remove the last column

    # omit noncausal predictor if not specified
    if not noncausal_predictor:
        generated_data = generated_data[:, :-1]  # remove the last column

    # add non-linear predictor if specified
    if nonlinear_predictor:
        generated_data = np.column_stack((generated_data, generated_data[:, 3]**2))

    # create DataFrame
    # if autocorrelation create variable with target lag
    if auto_corr:
        target_lag = np.zeros(n_obs)
        target_lag[0] = target[0]  # first value remains the same
        for i in range(1, n_obs):
            target_lag[i] = target[i - 1]
        data = np.column_stack((generated_data, target_lag, target))
        column_names = [f"X{i + 1}" for i in range(len(data[0, :])-1)] + ["Target"]
    else:
        column_names = [f"X{i + 1}" for i in range(len(generated_data[0, :]))] + ["Target"]
        data = np.column_stack((generated_data, target))

    data = pd.DataFrame(data, columns=column_names)
    data["Target"] = data["Target"].astype(int)
    if auto_corr:
        data[data.columns[-2]] = data[data.columns[-2]].astype(int)
    return data

# def data_loop(n_loops: int, scenarios: dict) -> dict:
#
#     data_list = {}
#
#     for key in scenarios.keys():
#         data_list[key] = []
#
#     for scenario in scenarios.keys():
#         for i in range(n_loops):
#
#             data_list[scenario].append(generate_data(**scenarios[scenario], random_state=i))
#
#     return data_list

def data_loop(n_loops: int, scenario: dict) -> list:
        for i in range(n_loops):
            yield generate_data(**scenario, random_state=i)


