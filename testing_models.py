import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Any

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

import scorecardpy as sc
from tqdm import tqdm


# Function to split data into train and test sets
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets.

    Parameters:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training and testing sets for both features and target.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Function to train models and return predicted probabilities
def train_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, random_state: int = None) -> Dict[str, np.ndarray]:
    """
    Trains multiple models and returns predicted probabilities for the test set.

    Parameters:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
        random_state (int): Random seed for reproducibility.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing predicted probabilities for each model.
    """
    models = {
        'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'xgboost': XGBClassifier(random_state=random_state),
        'lightgbm': LGBMClassifier(random_state=random_state, verbose=-1),  # Make LightGBM silent
        'catboost': CatBoostClassifier(random_state=random_state, verbose=0),  # Make CatBoost silent
        'adaboost': AdaBoostClassifier(random_state=random_state, algorithm='SAMME')  # Add SAMME method
    }

    # Store predicted probabilities for each model
    pred_probas = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_probas[name] = model.predict_proba(X_test)[:, 1]  # Predict probabilities for class 1

    return pred_probas


# Function to compute TPR and TNR for multiple thresholds
def compute_tpr_tnr(y_true: np.ndarray, y_pred_proba: np.ndarray, thresholds: List[float]) -> Tuple[List[float], List[float]]:
    """
    Computes True Positive Rate (TPR) and True Negative Rate (TNR) for different thresholds.

    Parameters:
        y_true (np.ndarray): True binary labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
        thresholds (List[float]): List of threshold values to calculate TPR and TNR.

    Returns:
        Tuple[List[float], List[float]]: Lists of TPR and TNR values corresponding to each threshold.
    """
    tpr_list = []
    tnr_list = []

    for threshold in thresholds:
        # Convert probabilities to binary predictions based on threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Compute True Positive Rate (TPR) and True Negative Rate (TNR)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

        tpr_list.append(tpr)
        tnr_list.append(tnr)

    return tpr_list, tnr_list


# Function to display TPR and TNR for all models across thresholds
def display_tpr_tnr(y_test: np.ndarray, pred_probas: Dict[str, np.ndarray], thresholds: List[float]) -> None:
    """
    Displays the True Positive Rate (TPR) and True Negative Rate (TNR) for all models across thresholds.

    Parameters:
        y_test (np.ndarray): True test labels.
        pred_probas (Dict[str, np.ndarray]): Predicted probabilities for each model.
        thresholds (List[float]): List of thresholds to evaluate.

    Returns:
        None
    """
    metrics = {}

    for model_name, probas in pred_probas.items():
        tpr_all_thresholds = []
        tnr_all_thresholds = []

        # Calculate TPR and TNR for each experiment and each threshold
        for proba in probas:
            tpr_list, tnr_list = compute_tpr_tnr(y_test, proba, thresholds)
            tpr_all_thresholds.append(tpr_list)
            tnr_all_thresholds.append(tnr_list)

        # Average the results across all experiments for each threshold
        avg_tpr_thresholds = np.mean(tpr_all_thresholds, axis=0)
        avg_tnr_thresholds = np.mean(tnr_all_thresholds, axis=0)

        # Store metrics for the model
        metrics[model_name] = {
            'avg_tpr': avg_tpr_thresholds,
            'avg_tnr': avg_tnr_thresholds
        }

    # Display the TPR and TNR for each model at each threshold
    for model_name, metric in metrics.items():
        print(f"Model: {model_name}")
        for i, threshold in enumerate(thresholds):
            print(f"  Threshold {threshold:.2f}: TPR = {metric['avg_tpr'][i]:.4f}, TNR = {metric['avg_tnr'][i]:.4f}")
        print()


# Modify run_experiments to return y_test for each run
def run_experiments(X: pd.DataFrame, y: pd.Series, n_experiments: int = 50, test_size: float = 0.2) -> Tuple[Dict[str, List[np.ndarray]], np.ndarray]:
    """
    Runs multiple experiments to train various models and store their predicted probabilities.

    Parameters:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
        n_experiments (int): Number of experiments to run.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        Tuple[Dict[str, List[np.ndarray]], np.ndarray]: A dictionary containing predicted probabilities for each model,
                                                         and the last y_test for evaluation.
    """
    results = { 
        'logistic_regression': [],
        'xgboost': [],
        'lightgbm': [],
        'catboost': [],
        'adaboost': []
    }
    y_test_final = None

    for i in tqdm(range(n_experiments), desc='Running experiments'):
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=i)

        # Save the y_test from the last split for evaluation
        if i == n_experiments - 1:
            y_test_final = y_test

        # Train models and get predicted probabilities
        pred_probas = train_models(X_train, X_test, y_train, y_test, random_state=i)

        # Store predicted probabilities for each model
        for model_name, probas in pred_probas.items():
            results[model_name].append(probas)

    return results, y_test_final


# Main function to run the experiments and display metrics
def main_experiment(y: pd.Series, X: pd.DataFrame, X_woe_manual: pd.DataFrame, X_woe_automatic: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Runs the experiments on different datasets and displays TPR and TNR metrics.

    Parameters:
        y (pd.Series): Target variable.
        X (pd.DataFrame): Original features dataset.
        X_woe_manual (pd.DataFrame): Features dataset with manually calculated WoE.
        X_woe_automatic (pd.DataFrame): Features dataset with automatically calculated WoE.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: The results for each dataset.
    """
    thresholds = np.arange(0, 1.01, 0.01)  # Generate thresholds from 0 to 1

    # Run experiments on each dataset and get the last y_test for metric calculation
    results_X, y_test_X = run_experiments(X, y)
    results_X_woe_manual, y_test_X_woe_manual = run_experiments(X_woe_manual, y)
    results_X_woe_automatic, y_test_X_woe_automatic = run_experiments(X_woe_automatic, y)

    # Display TPR and TNR for each dataset at different thresholds
    print("Results for X:")
    display_tpr_tnr(y_test_X, results_X, thresholds)

    print("Results for X_woe_manual:")
    display_tpr_tnr(y_test_X_woe_manual, results_X_woe_manual, thresholds)

    print("Results for X_woe_automatic:")
    display_tpr_tnr(y_test_X_woe_automatic, results_X_woe_automatic, thresholds)

    # Save results
    np.save('results_X.npy', results_X)
    np.save('results_X_woe_manual.npy', results_X_woe_manual)
    np.save('results_X_woe_automatic.npy', results_X_woe_automatic)

    return results_X, results_X_woe_manual, results_X_woe_automatic


# Function to train models and get predictions
def train_and_predict(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, np.ndarray]:
    """
    Trains several models on the training dataset and returns their predictions on the test dataset.

    Parameters:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target (not used but kept for consistency).

    Returns:
        Dict[str, np.ndarray]: A dictionary of model names and their corresponding predictions.
    """
    random_state = 42
    model_predictions = {}

    # Logistic Regression
    lr = LogisticRegression(random_state=random_state, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict_proba(X_test)[:, 1]
    model_predictions['Logistic Regression'] = lr_pred

    # XGBoost
    xgb = XGBClassifier(random_state=random_state)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict_proba(X_test)[:, 1]
    model_predictions['XGBoost'] = xgb_pred

    # LightGBM
    lgbm = LGBMClassifier(random_state=random_state, verbose=-1)
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict_proba(X_test)[:, 1]
    model_predictions['LightGBM'] = lgbm_pred

    # CatBoost
    catboost = CatBoostClassifier(random_state=random_state, verbose=0)
    catboost.fit(X_train, y_train)
    catboost_pred = catboost.predict_proba(X_test)[:, 1]
    model_predictions['CatBoost'] = catboost_pred

    # AdaBoost
    adaboost = AdaBoostClassifier(random_state=random_state, algorithm='SAMME')
    adaboost.fit(X_train, y_train)
    adaboost_pred = adaboost.predict_proba(X_test)[:, 1]
    model_predictions['AdaBoost'] = adaboost_pred

    return model_predictions


# Function to plot K-S curves and ROC curves with difference between TPR and FPR
def plot_ks_and_roc_curves(model_predictions: Dict[str, np.ndarray], y_test: np.ndarray, dataset_name: str) -> None:
    """
    Plots K-S curves and ROC curves for model predictions.

    Parameters:
        model_predictions (Dict[str, np.ndarray]): Dictionary of model names and their predicted probabilities.
        y_test (np.ndarray): True labels for the test set.
        dataset_name (str): Name of the dataset being evaluated.
    """
    plt.figure(figsize=(20, 8))

    for i, (model_name, y_pred) in enumerate(model_predictions.items()):
        # Compute ROC and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        # Compute K-S statistic
        ks_statistic = max(tpr - fpr)
        ks_threshold = thresholds[np.argmax(tpr - fpr)]

        # Plot K-S curve
        plt.subplot(2, len(model_predictions), i + 1)
        population_percentile = np.linspace(0, 1, len(tpr))

        plt.plot(population_percentile, tpr, color='black', label='Good (TPR)')
        plt.plot(population_percentile, fpr, color='blue', label='Bad (FPR)')
        plt.plot(population_percentile, tpr - fpr, color='green', label='Difference (TPR - FPR)')
        plt.axvline(np.argmax(tpr - fpr) / len(tpr), color='red', linestyle='--', label=f'KS: {ks_statistic:.4f}')
        plt.title(f'{model_name} K-S Plot')
        plt.xlabel('% of population')
        plt.ylabel('% of total Good/Bad')
        plt.legend(loc='best')

        # Plot ROC curve
        plt.subplot(2, len(model_predictions), len(model_predictions) + i + 1)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})', lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc='lower right')

    plt.tight_layout()
    plt.suptitle(f'K-S and ROC Curve Comparison for {dataset_name}', y=1.05, fontsize=16)
    plt.show()

# Function to print KS statistics
def print_ks_statistics(model_predictions: Dict[str, np.ndarray], y_test: np.ndarray, dataset_name: str) -> None:
    """
    Prints KS statistics for each model based on predictions.

    Parameters:
        model_predictions (Dict[str, np.ndarray]): Dictionary of model names and their predicted probabilities.
        y_test (np.ndarray): True labels for the test set.
        dataset_name (str): Name of the dataset being evaluated.
    """
    print(f"\nKS Statistics for {dataset_name}:")
    for model_name, y_pred in model_predictions.items():
        ks_value = sc.perf_eva(y_test, y_pred, show_plot=False)['KS']
        print(f'{model_name}: KS = {ks_value:.2f}')


def evaluate_all_datasets(y: np.ndarray, X: pd.DataFrame, X_woe_manual: pd.DataFrame, X_woe_automatic: pd.DataFrame) -> None:
    """
    Evaluates models on all datasets and plots K-S and ROC curves.

    Parameters:
        y (np.ndarray): Target variable.
        X (pd.DataFrame): Original features dataset.
        X_woe_manual (pd.DataFrame): Features dataset with manually calculated WoE.
        X_woe_automatic (pd.DataFrame): Features dataset with automatically calculated WoE.
    """
    datasets = {
        'X': X,
        'X_woe_manual': X_woe_manual,
        'X_woe_automatic': X_woe_automatic
    }

    for dataset_name, X_data in datasets.items():
        # Combine X and y into one DataFrame
        data = X_data.assign(creditability=y)
        # Split data into train and test
        split_data = sc.split_df(data, 'creditability')
        train = split_data['train']
        test = split_data['test']

        # Separate features (X) and target (y)
        X_train = train.drop(columns='creditability')
        y_train = train['creditability']
        X_test = test.drop(columns='creditability')
        y_test = test['creditability']

        # Train models and get predictions
        model_predictions_train = train_and_predict(X_train, X_train, y_train, y_train)
        model_predictions_test = train_and_predict(X_train, X_test, y_train, y_test)
        
        # Plot K-S and ROC curves
        plot_ks_and_roc_curves(model_predictions_test, y_test, dataset_name)
        
        # Print KS statistics
        # print_ks_statistics(model_predictions_test, y_test, dataset_name)


def evaluate_models_and_plot_confusion_matrices(X: pd.DataFrame, y: np.ndarray, random_state: int = 42) -> None:
    """
    Evaluates models by training them and plotting confusion matrices for their predictions.

    Parameters:
        X (pd.DataFrame): Features dataset.
        y (np.ndarray): Target variable.
        random_state (int): Random seed for reproducibility.
    """
    # Step 1: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Step 2: Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'XGBoost': XGBClassifier(random_state=random_state),
        'LightGBM': LGBMClassifier(random_state=random_state, verbose=-1),  # Make LightGBM silent
        'CatBoost': CatBoostClassifier(random_state=random_state, verbose=0),  # Make CatBoost silent
        'AdaBoost': AdaBoostClassifier(random_state=random_state, algorithm='SAMME')  # Add SAMME method
    }

    # Step 3: Initialize a dictionary to store optimal thresholds and predictions
    optimal_thresholds = {}
    predictions = {}

    # Step 4: Fit models and calculate optimal thresholds
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        # Get predicted probabilities for class 1
        probs = model.predict_proba(X_test)[:, 1]

        # Calculate ROC curve and optimal threshold
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        # Store optimal threshold and make predictions based on it
        optimal_thresholds[model_name] = optimal_threshold
        preds = (probs >= optimal_threshold).astype(int)
        predictions[model_name] = preds

    # Step 5: Plot confusion matrices
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))

    for ax, (model_name, preds) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(cm, display_labels=['Non-Default (0)', 'Default (1)']).plot(ax=ax)
        ax.set_title(f'{model_name}\nOptimal Threshold: {optimal_thresholds[model_name]:.2f}')

    plt.tight_layout()
    plt.show()

    # # Print optimal thresholds
    # for model_name, threshold in optimal_thresholds.items():
    #     print(f'Optimal Threshold for {model_name}: {threshold:.2f}')