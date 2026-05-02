import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from statsmodels.stats.anova import AnovaRM
import scikit_posthocs as sp

def load_data(path: str, file_name: str, max_n: int) -> pd.DataFrame:
    """Loops through all files in the specified path and loads the simulations data into a DataFrame.
    expects files to be in the format {file_name}_{i}.csv where i is the index of the file.

    :param path: The path to the directory containing the files.
    :param file_name: The base name of the files to load.
    :param max_n: The maximum index of the files to load.
    :return: A DataFrame containing the coefficient data from all files.
    """

    # Initialize an empty list to store the DataFrames
    df_list = []

    # Add possible number of observations used in simulations
    N_obs = [50, 500, 5000]

    # Loop through all files in the specified path, add number of possible observations and number of simulation
    # Number of simulation is changed every 3 dataframes, e.g. files 1,2,3 have scenario 1, files 4,5,6 scenario 2, etc.
    for i in range(1, max_n + 1):
        file_path = f"{path}/{file_name}_{i}.csv"
        df = pd.read_csv(file_path)
        df['N'] = N_obs[(i - 1) % len(N_obs)]
        df['Scenario'] = (i - 1) // len(N_obs) + 1
        df_list.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # Convert Scenario and N to string variables
    combined_df['Scenario'] = combined_df['Scenario'].astype(str)
    combined_df['N'] = combined_df['N'].astype(str)

    return combined_df

def remove_outliers(df: pd.DataFrame, scenarios: dict, second_df: pd.DataFrame = None, only_dropna: bool = False) \
        -> pd.DataFrame | list:
    """
        Remove rows from the first DataFrame that contain outliers in any numerical column, based on
        the IQR (boxplot) method or NaN values. These observations are also removed from the second DataFrame,
        if specified. If only_dropna is True, only rows with NaN values in the second DataFrame are removed.

        :param df: The first DataFrame from which to remove outliers.
        :param second_df: The second DataFrame from which to remove corresponding rows.
        :param scenarios: A dictionary containing the scenarios and list of N to remove the outliers.
        :param only_dropna: If True, only rows with NaN values are removed
        :return: The cleaned DataFrame(s) with outliers and NaN values removed.
        """
    df_clean = df.copy()
    if second_df is not None:
        df_sec_clean = second_df.copy()
    numeric_cols = df_clean.select_dtypes(include='number').columns

    # Use only the first 4 numeric columns
    numeric_cols = numeric_cols[:4]

    # Dictionary to store bounds
    bounds = {}

    # Calculate IQR-based bounds for each numeric column for specified scenarios and N
    for scenario, n_values in scenarios.items():
        for n in n_values:
            subset = df_clean[(df_clean['Scenario'] == scenario) & (df_clean['N'] == n)]
            for col in numeric_cols:
                Q1 = subset[col].quantile(0.25)
                Q3 = subset[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                bounds[(scenario, n, col)] = (lower_bound, upper_bound)

    # create a mask to identify outliers based on the calculated bounds
    outlier_mask = pd.Series(False, index=df_clean.index)

    # Loop through the bounds and update the outlier mask
    for (scenario, n, col), (lower_bound, upper_bound) in bounds.items():
        mask = (df_clean['Scenario'] == scenario) & (df_clean['N'] == n)
        outlier_mask |= ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)) & mask

    # Create a mask for NaN values in the first and second DataFrame if specified
    if second_df is not None:
        nan_mask = df_clean[numeric_cols].isna().any(axis=1) | df_sec_clean[numeric_cols].isna().any(axis=1)
    else:
        nan_mask = df_clean[numeric_cols].isna().any(axis=1)
        print(np.sum(nan_mask))

    # Combine the outlier mask and the NaN mask
    if only_dropna:
        final_mask = nan_mask
    else:
        final_mask = outlier_mask | nan_mask

    # Print the number of rows identified as outliers and/or containing NaN values for each scenario and N
    for scenario, n_values in scenarios.items():
        for n in n_values:
            scenario_n_mask = (df_clean['Scenario'] == scenario) & (df_clean['N'] == n)
            num_outliers = outlier_mask[scenario_n_mask].sum()
            num_nans = nan_mask[scenario_n_mask].sum()
            print(f"Scenario: {scenario}, N: {n} - Outliers: {num_outliers}, NaNs: {num_nans}")

    # Remove rows identified as outliers or containing NaN values
    df_clean = df_clean[~final_mask]
    if second_df is not None:
        df_sec_clean = df_sec_clean[~final_mask]
        return [df_clean, df_sec_clean]
    return df_clean

def calculate_ml_metric(df: pd.DataFrame, metric: str, statistic: str) -> pd.DataFrame:
    """
    Calculate the specified machine learning metric for each scenario, N and model.

    :param df: The DataFrame containing scenarios (rows), N (rows), models (rows) and metrics (columns).
    :param metric: The machine learning metric to use for the calculation.
    :param statistic: The statistic to calculate for the metric. One of 'mean', 'std', 'median', 'average_rank',
    'percent_best', 'percent_worst', 'quantile_05_95_diff'
    :return: A DataFrame with the calculated metric for each scenario, N and model.
    """

    # Prepare the DataFrame by assigning each simulation iteration an identifier
    unique_n = df['N'].unique()
    unique_scenarios = df['Scenario'].unique()
    unique_model = df['Model'].unique()
    n = df.loc[(df["N"] == unique_n[0]) & (df["Scenario"] == unique_scenarios[0]) & (df["Model"] == unique_model[0])].shape[0]

    df_copy = df.copy()

    for n_scen in unique_n:
        for scenario in unique_scenarios:
            for model in unique_model:
                df_copy.loc[(df_copy["N"] == n_scen) & (df_copy["Scenario"] == scenario) & (df_copy["Model"] == model), "Iteration"] = range(n)

    df_pivot = df_copy.pivot_table(index=["Scenario", "N", "Iteration"], columns="Model", values=metric)

    if statistic == "mean":
        result = df_pivot.groupby(["Scenario", "N"]).mean().reset_index()
    elif statistic == "std":
        result = df_pivot.groupby(["Scenario", "N"]).std().reset_index()
    elif statistic == "median":
        result = df_pivot.groupby(["Scenario", "N"]).median().reset_index()
    elif statistic == "average_rank":
        result = df_pivot.rank(axis=1, method='average', numeric_only=True, ascending=False).groupby(["Scenario", "N"]).mean().reset_index()
    elif statistic == "percent_best":
        ranked_df = df_pivot.rank(axis=1, method='average', ascending=False, numeric_only=True)
        for model in unique_model:
            ranked_df.loc[ranked_df[model] != 1, model] = 0
        result = ranked_df.groupby(["Scenario", "N"]).sum().reset_index()
        result[unique_model] = (result[unique_model] / n) * 100
    elif statistic == "percent_worst":
        ranked_df = df_pivot.rank(axis=1, method='dense', ascending=True, numeric_only=True)
        for model in unique_model:
            ranked_df.loc[ranked_df[model] != 1, model] = 0
        result = ranked_df.groupby(["Scenario", "N"]).sum().reset_index()
        result[unique_model] = (result[unique_model] / n) * 100
    elif statistic == "quantile_05_95_diff":
        quantiles = df_pivot.groupby(["Scenario", "N"]).quantile([0.05, 0.95]).unstack(level=-1)
        result = quantiles[unique_model][0.95] - quantiles[unique_model][0.05]
        result = result.reset_index()
    elif statistic == "no_statistic":
        result = df_pivot.reset_index()
    else:
        raise ValueError("Invalid statistic. Choose one of 'mean', 'std', 'median', 'average_rank', 'percent_best', "
                         "'percent_worst', 'quantile_05_95_diff', 'no_statistic'.")

    return result

def calculate_friedman_test(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Calculate the Friedman test p-value for each scenario checking if there is a significant difference between models.

    :param df: The DataFrame containing scenarios (rows), N (rows), models (rows) and metrics (columns).
    :param metric: The machine learning metric to use for the calculation.
    :return: A DataFrame with the calculated Friedman test p-value for each scenario.
    """

    # Prepare list to store results
    results = []

    unique_n = df['N'].unique()
    unique_scenarios = df['Scenario'].unique()
    unique_models = df['Model'].unique()
    
    # We need to simulate the 'Iteration' or leverage existing structure to align measurements
    df_copy = df.copy()
    
    # Assign iteration ID within each group
    # df_copy['Iteration'] = df_copy.groupby(['Scenario', 'N', 'Model']).cumcount()

    for n_val in unique_n:
        for scenario in unique_scenarios:
            # Filter data for current scenario and N
            subset = df_copy.loc[(df_copy["N"] == n_val) & (df_copy["Scenario"] == scenario),:].copy()

            n_iter = subset[subset["Model"] == unique_models[0]].shape[0]

            for m in unique_models:
                subset.loc[subset["Model"] == m, "Iteration"] = range(n_iter)
            
            # Pivot to have models as columns, indexed by Iteration
            # This ensures that we are comparing the corresponding iterations (paired samples)
            pivot_df = subset[["Iteration", "Model", metric]].pivot(index='Iteration', columns='Model', values=metric)

            pivot_df_before_n = pivot_df.shape[0]
            # Drop rows with NaN if any model failed for a specific iteration (to ensure paired samples)
            pivot_df = pivot_df.dropna()
            pivot_df_after_n = pivot_df.shape[0]
            if pivot_df_before_n != pivot_df_after_n:
                print(f"Warning: Dropped {pivot_df_before_n - pivot_df_after_n} iterations for Scenario {scenario}, N {n_val} due to NaN values.")

            if pivot_df.empty or len(pivot_df.columns) < 2:
                 raise ValueError(f"Not enough data to perform Friedman test for Scenario {scenario}, N {n_val}.")

            # Collect arrays for each model (column)
            data_lists = [pivot_df[col] for col in pivot_df.columns]
            
            try:
                stat, p_value = friedmanchisquare(*data_lists)
                results.append({
                    'Scenario': scenario,
                    'N': n_val,
                    'p-value': p_value,
                    'statistic': stat
                })
            except ValueError:
                 results.append({
                    'Scenario': scenario,
                    'N': n_val,
                    'p-value': np.nan,
                    'statistic': np.nan
                })

    return pd.DataFrame(results)

def calculate_nemenyi_test(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Calculate the Nemenyi post-hoc test p-values for each scenario.

    :param df: The DataFrame containing scenarios (rows), N (rows), models (rows) and metrics (columns).
    :param metric: The machine learning metric to use for the calculation.
    :return: A DataFrame with the calculated Nemenyi test p-values for each pair of models, scenario and N.
    """
    results = []

    unique_n = df['N'].unique()
    unique_scenarios = df['Scenario'].unique()
    unique_models = df['Model'].unique()
    
    df_copy = df.copy()

    for n_val in unique_n:
        for scenario in unique_scenarios:
            subset = df_copy.loc[(df_copy["N"] == n_val) & (df_copy["Scenario"] == scenario),:].copy()

            n_iter = subset[subset["Model"] == unique_models[0]].shape[0]

            for m in unique_models:
                subset.loc[subset["Model"] == m, "Iteration"] = range(n_iter)
            
            pivot_df = subset[["Iteration", "Model", metric]].pivot(index='Iteration', columns='Model', values=metric)
            pivot_df = pivot_df.dropna()
            
            if pivot_df.empty or len(pivot_df.columns) < 2:
                 continue

            try:
                # sp.posthoc_nemenyi_friedman expects data where columns are groups and rows are blocks
                nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_df)
                
                # Flatten the matrix
                nemenyi_results = nemenyi_results.stack().reset_index()
                nemenyi_results.columns = ['Model1', 'Model2', 'p-value']
                
                # Keep only unique pairs (upper triangle), exclude self-comparison
                nemenyi_results = nemenyi_results[nemenyi_results['Model1'] < nemenyi_results['Model2']]
                
                nemenyi_results['Scenario'] = scenario
                nemenyi_results['N'] = n_val
                
                results.append(nemenyi_results)
            except Exception as e:
                print(f"Error calculating Nemenyi for Scenario {scenario}, N {n_val}: {e}")

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def nemenyi_to_latex(df: pd.DataFrame, caption: str) -> str:
    """
    Convert the Nemenyi test results DataFrame into a LaTeX table format.

    :param df: The DataFrame containing the Nemenyi test results.
    :param caption: A string representing the caption for the LaTeX table.
    :return: A string representing the LaTeX table.
    """
    float_format = "%.3f"

    # Create a column for the pair comparison
    df['Comparison'] = df['Model1'] + " vs " + df['Model2']
    
    # Pivot the table to have scenarios and N as rows, and comparisons as columns
    pivot_df = df.pivot_table(index=['Scenario', 'N'], columns='Comparison', values='p-value')
    
    # Sort index to ensure correct order
    pivot_df = pivot_df.sort_index(level=[0, 1])

    pivot_df = pivot_df[["NN vs XGBoost"]]

    latex_table = pivot_df.to_latex(float_format=float_format, caption=caption, multirow=True,
                                          multicolumn=True, multicolumn_format="c")
    return latex_table

def calculate_repeated_measures_anova(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Calculate the Repeated Measures ANOVA p-value for each scenario checking if there is a significant difference between models.

    :param df: The DataFrame containing scenarios (rows), N (rows), models (rows) and metrics (columns).
    :param metric: The machine learning metric to use for the calculation.
    :return: A DataFrame with the calculated RM ANOVA p-value for each scenario.
    """
    results = []

    unique_n = df['N'].unique()
    unique_scenarios = df['Scenario'].unique()
    unique_models = df['Model'].unique()
    
    df_copy = df.copy()

    for n_val in unique_n:
        for scenario in unique_scenarios:
            subset = df_copy.loc[(df_copy["N"] == n_val) & (df_copy["Scenario"] == scenario),:].copy()

            n_iter = subset[subset["Model"] == unique_models[0]].shape[0]

            for m in unique_models:
                 # Assign iteration ID for pairing
                subset.loc[subset["Model"] == m, "Iteration"] = range(n_iter)
            
            pivot_df = subset[["Iteration", "Model", metric]].pivot(index='Iteration', columns='Model', values=metric)
            pivot_df = pivot_df.dropna()
            
            if pivot_df.empty or len(pivot_df.columns) < 2:
                 continue

            # Melt back for AnovaRM
            clean_subset = pivot_df.reset_index().melt(id_vars='Iteration', var_name='Model', value_name=metric)
            
            try:
                aovrm = AnovaRM(clean_subset, metric, 'Iteration', within=['Model'])
                res = aovrm.fit()
                
                # res.anova_table is a DataFrame
                p_value = res.anova_table['Pr > F'].iloc[0]
                f_stat = res.anova_table['F Value'].iloc[0]
                
                results.append({
                    'Scenario': scenario,
                    'N': n_val,
                    'p-value': p_value,
                    'statistic': f_stat
                })
            except Exception as e:
                print(f"Error calculating RM ANOVA for Scenario {scenario}, N {n_val}: {e}")
                results.append({
                    'Scenario': scenario,
                    'N': n_val,
                    'p-value': np.nan,
                    'statistic': np.nan
                })

    return pd.DataFrame(results)

def anova_to_latex(df: pd.DataFrame, caption: str) -> str:
    """
    Convert the RM ANOVA test results DataFrame into a LaTeX table format.

    :param df: The DataFrame containing the RM ANOVA results.
    :param caption: A string representing the caption for the LaTeX table.
    :return: A string representing the LaTeX table.
    """
    float_format = "%.3f"

    df_without_stat = df.drop(columns=['statistic'], inplace=False)

    df_without_stat["Scenario"] = df_without_stat["Scenario"].astype(int)

    # pivot N
    df_without_stat = df_without_stat.pivot(index='Scenario', columns='N', values='p-value')

    latex_table = df_without_stat.to_latex(float_format=float_format, caption=caption, multirow=True)
    return latex_table

def ml_metric_to_latex(df: pd.DataFrame, caption: str) -> str:
    """
    Convert the machine learning metric DataFrame into a LaTeX table format.

    :param df: The DataFrame containing the machine learning metric information.
    :param caption: A string representing the caption for the LaTeX table.
    :return: A string representing the LaTeX table.
    """
    float_format = "%.3f"

    # Try to pivot models
    df_long = df.melt(id_vars=["Scenario", "N"], var_name="Model", value_name="Metric")

    df_long["Scenario"] = df_long["Scenario"].astype(int)

    # pivot scebarios
    pivoted_df = df_long.pivot_table(index=["Model", "N"], columns="Scenario", values="Metric")

    # Put scenario and N into multiindex and export latex
    latex_table = pivoted_df.to_latex(float_format=float_format, caption=caption, multirow=True)
    return latex_table

def friedman_to_latex(df: pd.DataFrame, caption: str) -> str:
    """
    Convert the Friedman test results DataFrame into a LaTeX table format.

    :param df: The DataFrame containing the Friedman test results.
    :param caption: A string representing the caption for the LaTeX table.
    :return: A string representing the LaTeX table.
    """
    float_format = "%.2e"

    df_without_stat = df.drop(columns=['statistic'], inplace=False)

    df_without_stat["Scenario"] = df_without_stat["Scenario"].astype(int)

    df_without_stat["p-value"] = df_without_stat["p-value"].round(99)

    df_without_stat = df_without_stat.pivot(index='N', columns='Scenario', values='p-value')

    latex_table = df_without_stat.to_latex(float_format=float_format, caption=caption, multirow=True)
    return latex_table

def plot_ml_metric(df: pd.DataFrame, metric: str, title: str = None, ylabel: str = None, save_path: str = None) -> None:
    """
    Create a line plot for the specified machine learning metric across different scenarios and N.

    :param df: The DataFrame containing the machine learning metric information.
    :param metric: The machine learning metric to plot.
    :param title: The title of the plot.
    :param save_path: The path to save the plot. If None, the plot will not be saved.
    """

    # unpivot the DataFrame to have a long format suitable for seaborn
    df_long = df.melt(id_vars=['Scenario', 'N'], var_name='Model', value_name=metric)

    # make scenario ordering for facetgrid, so that 10 does not come before 2, etc.
    df_long['Scenario'] = df_long['Scenario'].astype(int)

    # plot the metric using seaborn (for each scenario make miniplot)
    g = sns.FacetGrid(df_long, col="Scenario", hue="Model", sharey=True, sharex=True, col_wrap=4,
                      height=1.25, aspect=1.15)
    g.map(sns.lineplot, "N", metric)

    # Manually hide x-axis labels for axes that are not in the bottom row of their column
    # This prevents the last label (e.g., "5000") from showing up on upper rows
    n_plots = len(g.axes)
    col_wrap = 4
    for i, ax in enumerate(g.axes):
        if i < n_plots - col_wrap:
             ax.tick_params(labelbottom=False)
             ax.set_xlabel('')

    if ylabel:
        g.set_axis_labels("N", ylabel)
    g.set_titles("Scenario {col_name}")
    plt.tight_layout()
    g.add_legend()
    if title:
        g.figure.suptitle(title, y=1.02)
    if save_path:
        plt.savefig(save_path)
    plt.show()
