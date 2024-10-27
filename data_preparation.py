import scorecardpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             accuracy_score, precision_score, recall_score)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import os


# Set random seed for reproducibility
def set_random_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Parameters:
    seed (int): The seed value to be set for the random number generator.
    """
    np.random.seed(seed)

# Plot styling setup
def set_plot_styling() -> None:
    """
    Set up the styling of plots using 'seaborn-pastel' theme.
    
    Customizes:
    - Font family: 'Times New Roman'
    - Axes title size, label size
    - Removes right and top spines from plots.
    """
    plt.style.use('seaborn-v0_8-pastel')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

# Load and preprocess data
def load_and_preprocess_data(file_path: str, index_col: Optional[float] =None) -> pd.DataFrame:
    """
    Load CSV data from the specified file path and preprocess the 'issue_d' column to a numeric format.
    
    Parameters:
    - file_path (str): The file path of the CSV data to load.
    
    Returns:
    - pd.DataFrame: The preprocessed DataFrame with 'issue_d' converted to a numeric timestamp.
    """
    df = pd.read_csv(file_path, index_col=index_col)
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    df['issue_d'] = pd.to_numeric(df['issue_d'], errors='coerce') / 10**9
    return df

# Extract numeric columns
def extract_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric columns from the given DataFrame, excluding the 'def' column if it exists.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame from which numeric columns will be extracted.
    
    Returns:
    - pd.DataFrame: A DataFrame containing only the numeric columns.
    """
    if 'def' in df.columns:
        df_cut = df.loc[:, df.columns != 'def']
    else:
        df_cut = df.copy()
    numeric_cols = df_cut.select_dtypes(include=['float64', 'int64']).columns
    return df_cut[numeric_cols]

# Bootstrap sampling for numeric columns
def bootstrap_sampling(df: pd.DataFrame, numeric_cols: List[str], n_bootstrap: int) -> Dict[str, Dict[str, float]]:
    """
    Perform bootstrap sampling to estimate the mean and variance for the specified numeric columns.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data to be sampled.
    numeric_cols (List[str]): A list of numeric column names to perform bootstrap sampling on.
    - n_bootstrap (int): The number of bootstrap samples to draw.
    
    Returns:
    - Dict[str, Dict[str, float]]: A dictionary where keys are column names and values are dictionaries 
    containing 'mean' and 'variance' of bootstrap samples.
    """
    bootstrap_results = {}
    for col in tqdm(numeric_cols, desc="Bootstrap Sampling"):
        means, variances = [], []
        for _ in range(n_bootstrap):
            sample = df[col].dropna().sample(frac=1, replace=True)
            means.append(np.mean(sample))
            variances.append(np.var(sample))
        bootstrap_results[col] = {
            'mean': np.mean(means),
            'variance': np.mean(variances)
        }
    return bootstrap_results

# Save results to JSON
def save_to_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save the given data to a JSON file.

    Parameters:
    - data (Dict[str, Any]): The data to be saved to the JSON file.
    - file_path (str): The path where the JSON file will be saved.
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Load results from JSON
def load_from_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Parameters:
    - file_path (str): The path to the JSON file to be loaded.

    Returns:
    - Dict[str, Any]: The data loaded from the JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

# Standardize columns
def standardize_columns(df: pd.DataFrame, bootstrap_estimators: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Standardize the columns of a DataFrame using the mean and variance from bootstrap estimators.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns to be standardized.
    - bootstrap_estimators (Dict[str, Dict[str, float]]): A dictionary containing the mean and variance for each column.

    Returns:
    - pd.DataFrame: The DataFrame with standardized columns.
    """
    for col, estimators in bootstrap_estimators.items():
        if col in df.columns:
            mean = estimators['mean']
            variance = estimators['variance']
            df[col] = (df[col] - mean) / np.sqrt(variance)
    return df

# Create is_leadership column based on job title
def classify_leadership(df: pd.DataFrame, leadership_keywords: List[str]) -> pd.DataFrame:
    """
    Create an 'is_leadership' column by classifying rows based on the 'emp_title' column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the 'emp_title' column.
    - leadership_keywords (List[str]): A list of keywords used to classify leadership roles.

    Returns:
    - pd.DataFrame: The DataFrame with an added 'is_leadership' column.
    """
    df['is_leadership'] = np.where(
        df['emp_title'].isna(), np.nan,
        df['emp_title'].str.contains('|'.join(leadership_keywords), case=False, na=False)
    )
    return df

# Sector classification
def classify_sectors(df: pd.DataFrame, sectors: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Classify job sectors based on keywords in the 'emp_title' column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the 'emp_title' column.
    - sectors (Dict[str, List[str]]): A dictionary where keys are sector names and values are lists of keywords for each sector.

    Returns:
    - pd.DataFrame: The DataFrame with added columns for each sector, indicating if the 'emp_title' matches sector keywords.
    """
    for sector, keywords in sectors.items():
        df[sector] = np.where(
            df['emp_title'].isna(), np.nan,
            df['emp_title'].str.contains('|'.join(keywords), case=False, na=False)
        )
    return df

# Drop a column from the DataFrame
def drop_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Drop the specified column from the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which to drop the column.
    - column_name (str): The name of the column to be dropped.

    Returns:
    - pd.DataFrame: The DataFrame without the specified column.
    """
    return df.drop(columns=[column_name])

# One-hot encode categorical columns
def one_hot_encode(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """
    Perform one-hot encoding on the specified categorical columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the columns to be encoded.
    - categorical_columns (List[str]): A list of categorical column names to be one-hot encoded.

    Returns:
    - pd.DataFrame: The DataFrame with one-hot encoded columns.
    """
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True) * 1

# Handle missing values in the dataset
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in numeric columns with the mean of each column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing missing values.

    Returns:
    - pd.DataFrame: The DataFrame with missing values imputed with the mean.
    """
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Check if a column is categorical
def is_categorical(df: pd.DataFrame, column_name: str, threshold: float = 0.0001) -> bool:
    """
    Check if a column is likely to be categorical based on its properties.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column to check.
    - column_name (str): The name of the column to check.
    threshold (float): A threshold that defines the maximum fraction of unique values for the column to be considered categorical (default is 0.0001).

    Returns:
    - bool: True if the column is likely categorical, False otherwise.
    """
    # Check if the column contains non-numeric data
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return True
    
    # Check the proportion of unique values
    unique_count = df[column_name].nunique()
    total_count = len(df[column_name])
    unique_ratio = unique_count / total_count
    
    if unique_ratio <= threshold:
        return True

    # Check if the numeric data is discrete (integers, not floats)
    is_discrete = df[column_name].apply(lambda x: float(x).is_integer()).all()
    if is_discrete:
        return True
    
    return False


def predict_missing_values(
    df: pd.DataFrame,
    df_imputed: pd.DataFrame,
    column: str,
    df_to_pred: pd.DataFrame = None,
    f1_threshold: float = 0.6,
    roc_auc_threshold: float = 0.6
) -> pd.DataFrame:
    """
    Predict and fill missing values for the specified column using classification or regression models.

    Parameters:
    - df (pd.DataFrame): Original DataFrame with missing values.
    - df_imputed (pd.DataFrame): DataFrame with imputed missing values used as a base for modeling.
    - column (str): Column name in which missing values will be predicted.
    - df_to_pred (pd.DataFrame): DataFrame with missing values to be filled using models from df (optional).
    - f1_threshold (float): Minimum F1 score threshold for classification model performance (default is 0.6).
    - roc_auc_threshold (float): Minimum ROC AUC score threshold for classification model performance (default is 0.6).

    Returns:
    Tuple:
        - pd.DataFrame: DataFrame with missing values filled based on the best model.
        - pd.DataFrame or None: If `df_to_pred` is provided, return the modified DataFrame; otherwise, return `None`.
    """
    y = df[column]
    X = df_imputed.drop(columns=[column])

    # Drop rows where y is missing
    X_complete = X[~y.isna()]
    y_complete = y.dropna()

    # If the entire column is missing, fill with mean or mode
    if y_complete.empty:
        if is_categorical(df, column):
            mode_value = y.mode()[0]
            df[column].fillna(mode_value, inplace=True)
        else:
            mean_value = y.mean()
            df[column].fillna(mean_value, inplace=True)
        
        if df_to_pred is not None:
            if is_categorical(df_to_pred, column):
                df_to_pred[column].fillna(mode_value, inplace=True)
            else:
                df_to_pred[column].fillna(mean_value, inplace=True)
        
        return df, df_to_pred

    # Split data for model training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_complete, y_complete, test_size=0.2, random_state=42
    )

    # Apply regression or classification based on the column type
    if not is_categorical(df, column):
        # Use regression models
        df = regression_fill(df, df_imputed, X_train, X_test, y_train, y_test, column)
        
        if df_to_pred is not None:
            df_to_pred = regression_fill(df_to_pred, df_imputed, X_train, X_test, y_train, y_test, column)
    else:
        # Use classification models
        df = classification_fill(
            df, df_imputed, X_train, X_test, y_train, y_test, column, f1_threshold, roc_auc_threshold
        )
        
        if df_to_pred is not None:
            df_to_pred = classification_fill(
                df_to_pred, df_imputed, X_train, X_test, y_train, y_test, column, f1_threshold, roc_auc_threshold
            )

    return df, df_to_pred


def regression_fill(
    df: pd.DataFrame,
    df_imputed: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    column: str
) -> pd.DataFrame:
    """
    Fill missing values using regression models and choose the best model based on Mean Squared Error (MSE).

    Parameters:
    - df (pd.DataFrame): Original DataFrame with missing values.
    - df_imputed (pd.DataFrame): DataFrame with imputed missing values used for modeling.
    - X_train (pd.DataFrame): Training feature data.
    - X_test (pd.DataFrame): Testing feature data.
    - y_train (pd.Series): Training target data.
    - y_test (pd.Series): Testing target data.
    - column (str): The column name in which missing values will be filled.

    Returns:
    - pd.DataFrame: DataFrame with missing values filled using the best performing regression model.
    """
    lin_reg = LinearRegression()
    catboost_reg = cb.CatBoostRegressor(verbose=0, random_state=42)
    elastic_net = ElasticNet(random_state=42)

    lin_reg.fit(X_train, y_train)
    catboost_reg.fit(X_train, y_train)
    elastic_net.fit(X_train, y_train)

    # Calculate predictions and metrics
    lin_reg_pred = lin_reg.predict(X_test)
    catboost_pred = catboost_reg.predict(X_test)
    elastic_net_pred = elastic_net.predict(X_test)

    mse_lin = mean_squared_error(y_test, lin_reg_pred)
    mse_catboost = mean_squared_error(y_test, catboost_pred)
    mse_enet = mean_squared_error(y_test, elastic_net_pred)

    # Choose the best model based on MSE
    best_model_name = min(
        [('Linear Regression', mse_lin), ('CatBoost', mse_catboost), ('ElasticNet', mse_enet)],
        key=lambda x: x[1]
    )[0]

    print(f"Best regression model for '{column}': {best_model_name}")

    # Fill missing values using the best model
    missing_idx = df[df[column].isna()].index
    X_missing = df_imputed.loc[missing_idx].drop(columns=[column])

    if best_model_name == 'Linear Regression':
        df.loc[missing_idx, column] = lin_reg.predict(X_missing)
    elif best_model_name == 'CatBoost':
        df.loc[missing_idx, column] = catboost_reg.predict(X_missing)
    else:
        df.loc[missing_idx, column] = elastic_net.predict(X_missing)

    return df


# Fill missing values with classification models
def classification_fill(
    df: pd.DataFrame,
    df_imputed: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    column: str,
    f1_threshold: float,
    roc_auc_threshold: float
) -> pd.DataFrame:
    """
    Fill missing values using classification models and choose the best model based on F1 score and ROC AUC.

    Parameters:
    - df (pd.DataFrame): Original DataFrame with missing values.
    - df_imputed (pd.DataFrame): DataFrame with imputed missing values used for modeling.
    - X_train (pd.DataFrame): Training feature data.
    - X_test (pd.DataFrame): Testing feature data.
    - y_train (pd.Series): Training target data.
    - y_test (pd.Series): Testing target data.
    - column (str): The column name in which missing values will be filled.
    - f1_threshold (float): Minimum F1 score for acceptable model performance.
    - roc_auc_threshold (float): Minimum ROC AUC score for acceptable model performance.

    Returns:
    - pd.DataFrame: DataFrame with missing values filled using the best performing classification model.
    """
    ada_boost = AdaBoostClassifier(random_state=42, algorithm='SAMME')
    log_reg = LogisticRegression(max_iter=1000)
    catboost_cls = cb.CatBoostClassifier(verbose=0, random_state=42)

    ada_boost.fit(X_train, y_train)
    log_reg.fit(X_train, y_train)
    catboost_cls.fit(X_train, y_train)

    ada_pred = ada_boost.predict(X_test)
    log_pred = log_reg.predict(X_test)
    cat_pred = catboost_cls.predict(X_test)

    f1_ada = f1_score(y_test, ada_pred, average='weighted')
    f1_log = f1_score(y_test, log_pred, average='weighted')
    f1_catboost = f1_score(y_test, cat_pred, average='weighted')

    roc_auc_ada = roc_auc_score(y_test, ada_pred, multi_class='ovr')
    roc_auc_log = roc_auc_score(y_test, log_pred, multi_class='ovr')
    roc_auc_catboost = roc_auc_score(y_test, cat_pred, multi_class='ovr')

    # Check if all models perform poorly
    if (f1_ada < f1_threshold or roc_auc_ada < roc_auc_threshold) and \
       (f1_log < f1_threshold or roc_auc_log < roc_auc_threshold) and \
       (f1_catboost < f1_threshold or roc_auc_catboost < roc_auc_threshold):
        mode_value = y_train.mode()[0]
        df[column] = df[column].fillna(mode_value)
        print(f"All classification models performed poorly for '{column}'. Filled missing values with mode.")
        return df

    # Choose the best model based on F1 score
    best_model_name = max(
        [('AdaBoost', f1_ada), ('Logistic Regression', f1_log), ('CatBoost', f1_catboost)],
        key=lambda x: x[1]
    )[0]

    print(f"Best classification model for '{column}': {best_model_name}")

    # Fill missing values using the best model
    missing_idx = df[df[column].isna()].index
    X_missing = df_imputed.loc[missing_idx].drop(columns=[column])

    if best_model_name == 'AdaBoost':
        df.loc[missing_idx, column] = ada_boost.predict(X_missing)
    elif best_model_name == 'Logistic Regression':
        df.loc[missing_idx, column] = log_reg.predict(X_missing)
    else:
        df.loc[missing_idx, column] = catboost_cls.predict(X_missing)

    return df


def extract_bins(tree: DecisionTreeClassifier) -> list:
    """
    Extract the split points (bins) from a trained decision tree model.

    Parameters:
    - tree (DecisionTreeClassifier): A trained decision tree classifier.

    Returns:
    - list: A sorted list of unique split points (thresholds) representing the bin edges.
    """
    left_indices = tree.tree_.children_left
    right_indices = tree.tree_.children_right
    split_points = []
    node_stack = [0]
    while node_stack:
        node = node_stack.pop()
        if left_indices[node] != right_indices[node]:
            split_value = tree.tree_.threshold[node]
            split_points.append(split_value)
            node_stack.append(left_indices[node])
            node_stack.append(right_indices[node])
    return sorted(set(split_points))


def calculate_woe_using_best_bins(X: pd.DataFrame, y: pd.Series, optimal_bins_results: dict) -> tuple:
    """
    Calculate the Weight of Evidence (WoE) values using the optimal number of bins for each feature.

    Parameters:
    - X (pd.DataFrame): Input features data.
    - y (pd.Series): Target binary variable (events and non-events).
    - optimal_bins_results (dict): Dictionary containing the optimal number of bins for each feature.

    Returns:
    - tuple: A tuple containing the WoE values DataFrame and a dictionary of WoE values and intervals for each feature.
    """
    woe_values = pd.DataFrame(index=y.index)
    woe_dict = {}

    for col, result in optimal_bins_results.items():
        if result['best_bins'] is not None:
            max_bins = result['best_bins']
            tree = DecisionTreeClassifier(max_leaf_nodes=max_bins, random_state=42)
            tree.fit(X[[col]], y)

            intervals = extract_bins(tree)
            intervals = [(-np.inf, intervals[0])] + [(intervals[i], intervals[i + 1]) for i in range(len(intervals) - 1)] + [(intervals[-1], np.inf)]

            woe_list = []
            for lower_bound, upper_bound in intervals:
                mask = (X[col] > lower_bound) & (X[col] <= upper_bound)
                event_count = y[mask].sum()
                non_event_count = mask.sum() - event_count

                event_rate = event_count / y.sum() if y.sum() > 0 else 0
                non_event_rate = non_event_count / (len(y) - y.sum()) if (len(y) - y.sum()) > 0 else 0

                woe = np.log((non_event_rate + 1e-10) / (event_rate + 1e-10))
                woe_list.append(woe)

            woe_series = pd.Series(index=X.index, dtype=float)
            for idx, (lower_bound, upper_bound) in enumerate(intervals):
                mask = (X[col] > lower_bound) & (X[col] <= upper_bound)
                woe_series[mask] = woe_list[idx]

            woe_values[col] = woe_series
            woe_dict[col] = {'woe_values': woe_list, 'intervals': intervals}

    return woe_values, woe_dict


def score_woe(woe_dict: dict) -> float:
    """
    Evaluate the quality of WoE intervals based on the variance of WoE values.

    Parameters:
    - woe_dict (dict): Dictionary containing WoE values for each feature.

    Returns:
    - float: The average variance of WoE values across all features.
    """
    scores = []
    for col, info in woe_dict.items():
        score = np.var(info['woe_values'])  # Calculate variance of WoE values
        scores.append(score)
    
    return np.mean(scores)


def grid_search_optimal_bins(X: pd.DataFrame, y: pd.Series, param_grid: list) -> dict:
    """
    Perform a grid search to find the optimal number of bins for each feature.

    Parameters:
    - X (pd.DataFrame): Input features data.
    - y (pd.Series): Target binary variable (events and non-events).
    - param_grid (list): List of candidate numbers of bins to evaluate.

    Returns:
    - dict: A dictionary with the optimal number of bins and scores for each feature.
    """
    best_results = {}
    
    for col in X.columns:
        if not is_categorical(X, col):
            best_score = float('inf')
            best_bins = None
            
            for max_bins in param_grid:
                tree = DecisionTreeClassifier(max_leaf_nodes=max_bins, random_state=42)
                tree.fit(X[[col]], y)
                intervals = extract_bins(tree)
                if not intervals:
                    continue
                
                intervals = [(-np.inf, intervals[0])] + [(intervals[i], intervals[i + 1]) for i in range(len(intervals) - 1)] + [(intervals[-1], np.inf)]
                woe_values = []

                for lower_bound, upper_bound in intervals:
                    mask = (X[col] > lower_bound) & (X[col] <= upper_bound)
                    event_count = y[mask].sum()
                    non_event_count = mask.sum() - event_count

                    event_rate = event_count / y.sum() if y.sum() > 0 else 0
                    non_event_rate = non_event_count / (len(y) - y.sum()) if (len(y) - y.sum()) > 0 else 0

                    woe = np.log((non_event_rate + 1e-10) / (event_rate + 1e-10))
                    woe_values.append(woe)

                woe_dict = {col: {'woe_values': woe_values}}
                score = score_woe(woe_dict)
                
                if score < best_score:
                    best_score = score
                    best_bins = max_bins
            
            best_results[col] = {
                'best_bins': best_bins,
                'best_score': best_score,
            }
    
    return best_results


def calculate_iv(woe_dict: dict, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Calculate the Information Value (IV) for each feature based on WoE values.

    Parameters:
    - woe_dict (dict): Dictionary containing WoE values and intervals for each feature.
    - X (pd.DataFrame): Input features data.
    - y (pd.Series): Target binary variable (events and non-events).

    Returns:
    - dict: A dictionary containing IV values for each feature.
    """
    iv_dict = {}

    for col, info in woe_dict.items():
        intervals = info['intervals']
        woe_values = info['woe_values']
        iv = 0

        for idx, (lower_bound, upper_bound) in enumerate(intervals):
            mask = (X[col] > lower_bound) & (X[col] <= upper_bound)
            event_count = y[mask].sum()
            non_event_count = mask.sum() - event_count

            event_rate = event_count / y.sum() if y.sum() > 0 else 0
            non_event_rate = non_event_count / (len(y) - y.sum()) if (len(y) - y.sum()) > 0 else 0

            iv_contribution = (non_event_rate - event_rate) * woe_values[idx]
            iv += iv_contribution
        
        iv_dict[col] = iv

    return iv_dict


def replace_values_with_woe(X: pd.DataFrame, woe_dict: dict, iv_dict: dict, iv_threshold: float) -> pd.DataFrame:
    """
    Replace the original feature values in X with their corresponding WoE values and filter by IV threshold.

    Parameters:
    - X (pd.DataFrame): Input features data.
    - woe_dict (dict): Dictionary containing WoE values and intervals for each feature.
    - iv_dict (dict): Dictionary containing IV values for each feature.
    - iv_threshold (float): Minimum IV value for a feature to be included.

    Returns:
    - pd.DataFrame: Transformed dataset where features are replaced by their WoE values.
    """
    X_woe = X.copy()
    filtered_columns = [col for col, iv in iv_dict.items() if iv > iv_threshold]

    for col in filtered_columns:
        intervals = woe_dict[col]['intervals']
        woe_values = woe_dict[col]['woe_values']

        for idx, (lower_bound, upper_bound) in enumerate(intervals):
            mask = (X[col] > lower_bound) & (X[col] <= upper_bound)
            X_woe.loc[mask, col] = woe_values[idx]

    return X_woe[filtered_columns]


def calculate_woe_with_iv_filter(X: pd.DataFrame, y: pd.Series, optimal_bins_results: dict, iv_threshold: float) -> pd.DataFrame:
    """
    Calculate WoE values, filter features by IV, and transform the dataset.

    Parameters:
    - X (pd.DataFrame): Input features data.
    - y (pd.Series): Target binary variable (events and non-events).
    - optimal_bins_results (dict): Dictionary containing the optimal number of bins for each feature.
    - iv_threshold (float): Minimum IV value for a feature to be included.

    Returns:
    - pd.DataFrame: Transformed dataset with filtered features replaced by WoE values.
    """
    woe_values, woe_dict = calculate_woe_using_best_bins(X, y, optimal_bins_results)
    iv_dict = calculate_iv(woe_dict, X, y)
    X_woe_filtered = replace_values_with_woe(X, woe_dict, iv_dict, iv_threshold)
    
    return X_woe_filtered


def get_woe_data(df: pd.DataFrame, target_col: str, json_file_path: str="woe_bins.json") -> pd.DataFrame:
    """Calculate WoE transformation for a dataset and save WoE bins to a JSON file.

    Args:
    - df (pd.DataFrame): The input DataFrame containing features and the target variable.
    - target_col (str): The name of the target variable column in the DataFrame.
    - json_file_path (str): The file path where the WoE bins will be saved in JSON format.

    Returns:
    - pd.DataFrame: A DataFrame containing the features transformed into WoE values.
    """
    # Variable filtering
    dt_s = sc.var_filter(df, y=target_col)
    
    # WoE binning
    bins = sc.woebin(dt_s, y=target_col)
    
    # WoE binning adjustment (interactive)
    breaks_adj = sc.woebin_adj(dt_s, target_col, bins)
    bins_adj = sc.woebin(dt_s, y=target_col, breaks_list=breaks_adj)
    
    # Apply WoE transformation to the dataset
    df_woe = sc.woebin_ply(dt_s, bins_adj)
    
    # Extract X dataset with WoE values (exclude the target column)
    X_woe = df_woe.loc[:, df_woe.columns != target_col]
    
    # Convert the bins to a dictionary
    bins_dict = {key: value.to_dict(orient='records') for key, value in bins_adj.items()}
    
    # Save the dictionary to a JSON file
    with open(json_file_path, 'w') as f:
        json.dump(bins_dict, f, indent=4)
    return X_woe


def main() -> tuple:
    """Main function to run the data processing and modeling pipeline.

    This function orchestrates the entire process from data loading,
    preprocessing, bootstrapping, missing value imputation, WoE calculation,
    and returns the target variable and feature sets.

    Returns:
        tuple: A tuple containing:
            - y (pd.Series): The target variable series.
            - X (pd.DataFrame): The feature DataFrame before WoE transformation.
            - X_woe_manual (pd.DataFrame): The feature DataFrame transformed using manual WoE.
            - X_woe_automatic (pd.DataFrame): The feature DataFrame transformed using automatic WoE.
            - df_to_pred (pd.DataFrame): The DataFrame for predictions with missing values imputed.
    """
    set_random_seed(42)

    # Set plot styling
    set_plot_styling()

    # Load data
    df = load_and_preprocess_data('data.csv')
    # Load data for prediction
    df_to_pred = load_and_preprocess_data('new_clients_for_scoring.csv', index_col=0)

    # Extract numeric columns
    df_cut = extract_numeric_columns(df)
    numeric_cols = df_cut.columns.tolist()

    if not 'bootstrap_estimators.json' in os.listdir():
        # Perform bootstrap sampling
        n_bootstrap = 1000
        bootstrap_results = bootstrap_sampling(df_cut, numeric_cols, n_bootstrap)
    
        # Save bootstrap results
        save_to_json(bootstrap_results, 'bootstrap_estimators.json')
        print("Bootstrap estimation complete. Results saved to 'bootstrap_estimators.json'.")
    
        # Load bootstrap estimators
        bootstrap_estimators = load_from_json('bootstrap_estimators.json')
    
        # Standardize columns
        df = standardize_columns(df, bootstrap_estimators)
        print("To-estimate-data standardization complete.")
    else:
        # Load the estimations from the JSON file
        with open('bootstrap_estimators.json', 'r') as file:
            estimations = json.load(file)
        
        # Standardization process
        for column in estimations.keys():
            mean = estimations[column]['mean']
            variance = estimations[column]['variance']
            std_dev = np.sqrt(variance)
            
            # Standardize the column
            df[column] = (df[column] - mean) / std_dev
        print("To-estimate-data standardization complete.")

    # Standardization for the prediction dataset
    with open('bootstrap_estimators.json', 'r') as file:
        estimations = json.load(file)
    
    for column in estimations.keys():
        mean = estimations[column]['mean']
        variance = estimations[column]['variance']
        std_dev = np.sqrt(variance)
        
        # Standardize the column
        df_to_pred[column] = (df_to_pred[column] - mean) / std_dev
    print("To-predict-data standardization complete.")

    # Leadership classification
    leadership_keywords = [
        'manager', 'director', 'supervisor', 'lead',
        'president', 'chief', 'head', 'executive', 'officer'
    ]
    df = classify_leadership(df, leadership_keywords)
    df_to_pred = classify_leadership(df_to_pred, leadership_keywords)

    # Sector classification
    sectors = {
        'healthcare': ['hospital', 'nurse', 'medical', 'clinic', 'healthcare', 'doctor', 'RN'],
        'education': ['school', 'teacher', 'professor', 'university', 'education', 'instructor'],
        'finance': ['bank', 'finance', 'account', 'investment', 'financial', 'loan'],
        'technology': ['engineer', 'developer', 'programmer', 'technician', 'technology', 'IT'],
        'government': ['government', 'federal', 'state', 'city', 'police', 'officer', 'military'],
    }
    df = classify_sectors(df, sectors)
    df_to_pred = classify_sectors(df_to_pred, sectors)

    # Drop emp_title column
    df = drop_column(df, 'emp_title')
    df_to_pred = drop_column(df_to_pred, 'emp_title')

    # One-hot encode categorical variables
    categorical_columns = ['purpose', 'addr_state', 'sub_grade', 'home_ownership']
    df = one_hot_encode(df, categorical_columns)
    df_to_pred = one_hot_encode(df_to_pred, categorical_columns)

    # Exclude 'def' column for imputation
    if 'def' in df.columns:
        df_i = df.drop(columns=['def'])
    else:
        df_i = df.copy()

    # Handle missing values by imputing (mean for numeric columns)
    df_imputed = impute_missing_values(df_i)

    # Ensure test data has all necessary columns (assign missing ones as 0s)
    for i in set(df_imputed.columns) - set(df_to_pred.columns):
        df_to_pred[i] = 0

    # Define thresholds for F1 and ROC AUC
    f1_threshold = 0.6  # Adjust this threshold as needed
    roc_auc_threshold = 0.6  # Adjust this threshold as needed

    # Predict and fill missing values in the original dataset
    for column in df_i.columns[df_i.isna().any()]:
        print(f"Processing column '{column}' for missing value imputation...")
        df, _ = predict_missing_values(df, df_imputed, column)

    print("Missing values imputed successfully in the training dataset.")

    # Now impute missing values in df_to_pred using the same logic
    for column in df_i.columns[df_to_pred.isna().any()]:
        print(f"Processing column '{column}' in the prediction dataset for missing value imputation...")
        _, df_to_pred = predict_missing_values(df, df_imputed, column, df_to_pred)

    print("Missing values imputed successfully in the prediction dataset.")

    # Save 'def' to a separate variable
    y = df['def']
    X = df.drop(['def'], axis=1)

    # Finding the optimal number of intervals
    param_grid = [2, 3, 4, 5, 6, 7, 8, 9, 10] 
    optimal_bins_results = grid_search_optimal_bins(X, y, param_grid=param_grid)  # Example grid search for bin optimization
    X_woe_manual = calculate_woe_with_iv_filter(X, y, optimal_bins_results, iv_threshold=0.02)

    # Getting X_woe_automatic
    with warnings.catch_warnings(action="ignore"):
        X_woe_automatic = get_woe_data(df, 'def')

    return y, X, X_woe_manual, X_woe_automatic, df_to_pred
