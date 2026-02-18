import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

