"""
improved_model.py
This script performs the same steps as build_model.py but with improvements:
- Hyperparameter tuning via GridSearchCV
- Cross-validation analysis
- Better evaluation metrics (F1, ROC-AUC)
"""

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score


def load_data(filepath):
    """Load featured data"""

    df = pd.read_csv(filepath)

    return df


def load_labels(filepath):
    """Load labels"""

    labels = pd.read_csv(filepath)

    return labels


def merge_with_labels(df, labels):
    """
    Merge featured data with labels
    """

    # Merge on nct_id
    df_merged = df.merge(labels, on='nct_id', how='inner')

    return df_merged


def drop_unnecessary_columns(df):
    """
    Drop columns that are:
    1. No variance (all same value)
    2. Replaced by derived features
    3. Not needed for modeling
    4. High cardinality
    5. Data leakage
    """

    columns_to_drop = [
        # No variance
        'phase',
        'study_type',

        # Derived features replace these
        'start_date',
        'completion_date',
        'primary_completion_date',
        'sponsor',
        'intervention_names',

        # Data leakage (status is outcome-related)
        'status',

        # High cardinality (too many categories)
        'conditions',
        'intervention_types',
    ]

    # Check which exist
    existing = [col for col in columns_to_drop if col in df.columns]
    missing = [col for col in columns_to_drop if col not in df.columns]

    if missing:
        print(f"\nNote: Columns not found (already dropped?): {missing}")

    print(f"\nDropping {len(existing)} columns:")
    for col in existing:
        print(f"  - {col}")

    df = df.drop(columns=existing)

    print(f"\nRemaining columns: {len(df.columns)}")

    return df


def prepare_features_for_modeling(df):
    """
    Separate features into categories for encoding/scaling
    """

    # Categorical columns (need one-hot encoding)
    categorical_cols = [
        'allocation',
        'intervention_model',
        'primary_purpose',
        'masking',
        'sponsor_type',
    ]

    # Continuous columns (need scaling)
    continuous_cols = [
        'enrollment',
        'trial_duration_months',
        'sponsor_trial_count',
        'design_quality_score',
    ]

    # Binary columns (keep as-is, 0/1)
    binary_cols = [
        'is_top_pharma',
        'has_placebo',
        'has_results',
        'enrollment_missing',
    ]

    # Add numeric results columns if they exist
    for col in ['num_outcome_measures', 'num_primary_outcomes', 'num_secondary_outcomes']:
        if col in df.columns:
            continuous_cols.append(col)


    return df, categorical_cols, continuous_cols, binary_cols


def apply_one_hot_encoding(df, categorical_cols):
    """
    One-hot encode categorical columns
    """

    # One-hot encode (drop_first=True to avoid multicollinearity)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Show what was created
    new_cols = set(df_encoded.columns) - set(df.columns)

    return df_encoded


def split_data(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets
    """

    # Separate features and target
    X = df.drop(columns=['nct_id', 'success'])
    y = df['success']
    nct_ids = df['nct_id']

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y, nct_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val size
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )


    return X_train, X_val, X_test, y_train, y_val, y_test, ids_train, ids_val, ids_test


def apply_scaling(X_train, X_val, X_test, continuous_cols):
    """
    Scale continuous features using StandardScaler
    Fit on train, transform on train/val/test
    """

    # Filter to only continuous columns that exist
    continuous_cols = [col for col in continuous_cols if col in X_train.columns]

    print(f"\nScaling {len(continuous_cols)} continuous features:")
    for col in continuous_cols:
        print(f"  - {col}")

    # Initialize scaler
    scaler = StandardScaler()

    # Fit on training data only
    scaler.fit(X_train[continuous_cols])

    # Transform all sets
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continuous_cols] = scaler.transform(X_train[continuous_cols])
    X_val_scaled[continuous_cols] = scaler.transform(X_val[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])


    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                       ids_train, ids_val, ids_test, scaler, output_dir='data'):
    """
    Save all processed data and scaler
    """

    # Save train set
    train_df = X_train.copy()
    train_df['nct_id'] = ids_train.values
    train_df['success'] = y_train.values
    train_file = f"{output_dir}/train.csv"
    train_df.to_csv(train_file, index=False)

    # Save validation set
    val_df = X_val.copy()
    val_df['nct_id'] = ids_val.values
    val_df['success'] = y_val.values
    val_file = f"{output_dir}/val.csv"
    val_df.to_csv(val_file, index=False)

    # Save test set
    test_df = X_test.copy()
    test_df['nct_id'] = ids_test.values
    test_df['success'] = y_test.values
    test_file = f"{output_dir}/test.csv"
    test_df.to_csv(test_file, index=False)


def tune_hyperparameters(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV

    Parameters:
    - X_train: Training features
    - y_train: Training labels

    Returns:
    - Best model from grid search
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING")
    print("="*80)

    # Define parameter grid (optimized for basic computers with 5-fold CV)
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [10, None],
    #     'min_samples_split': [2, 5],
    #     'class_weight': ['balanced', None],
    # }
    param_grid = {
        'min_samples_leaf':[1,3,10],
        'n_estimators':[100, 200, 300, 1000],
        'max_features':[0.1,0.5,1.],
        'max_samples':[0.5,None],
        'max_depth': [5, 10, None]
    }

    # Create base model
    rf = RandomForestClassifier(random_state=42)

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1', #'roc_auc',
        n_jobs=2
    )

    #total_combinations = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) *
    #                      len(param_grid['min_samples_split']) * len(param_grid['class_weight']))
    total_combinations = 3 * 4 * 3 * 2 * 3
    print(f"Starting grid search...")
    print(f"Testing {total_combinations} parameter combinations with 5-fold CV = {total_combinations * 5} total fits")
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV f1 score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_with_cross_validation(model, X_train, y_train, cv=5):
    """
    Evaluate model using cross-validation

    Parameters:
    - model: Trained model
    - X_train: Training features
    - y_train: Training labels
    - cv: Number of folds (default=5)

    Returns:
    - CV scores dictionary
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*80)

    # F1 scores
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=2)
    print(f"\nF1 Scores (cv={cv}): {cv_f1}")
    print(f"Mean F1: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")

    # Accuracy scores
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=2)
    print(f"\nAccuracy Scores (cv={cv}): {cv_acc}")
    print(f"Mean Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std() * 2:.4f})")

    return {
        'f1_scores': cv_f1,
        'accuracy_scores': cv_acc,
        'mean_f1': cv_f1.mean(),
        'mean_accuracy': cv_acc.mean()
    }


def model_building():
    # Load data
    train = pd.read_csv('data/train.csv')
    val = pd.read_csv('data/val.csv')
    test = pd.read_csv('data/test.csv')

    # Separate features and target
    X_train = train.drop(columns=['nct_id', 'success'])
    y_train = train['success']

    X_val = val.drop(columns=['nct_id', 'success'])
    y_val = val['success']

    X_test = test.drop(columns=['nct_id', 'success'])
    y_test = test['success']

    # Train baseline model (for comparison)
    print("="*80)
    print("BASELINE MODEL")
    print("="*80)
    baseline_model = RandomForestClassifier(random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_val_score = baseline_model.score(X_val, y_val)
    baseline_val_pred = baseline_model.predict(X_val)
    baseline_val_f1 = f1_score(y_val, baseline_val_pred)
    baseline_val_pred_proba = baseline_model.predict_proba(X_val)[:, 1]
    baseline_val_auc = roc_auc_score(y_val, baseline_val_pred_proba)
    print(f'Baseline Validation Accuracy: {baseline_val_score:.4f}')
    print(f'Baseline Validation F1 Score: {baseline_val_f1:.4f}')
    print(f'Baseline Validation ROC-AUC: {baseline_val_auc:.4f}')

    baseline_test_score = baseline_model.score(X_test, y_test)
    baseline_test_pred = baseline_model.predict(X_test)
    baseline_test_f1 = f1_score(y_test, baseline_test_pred)
    baseline_test_pred_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_test_auc = roc_auc_score(y_test, baseline_test_pred_proba)
    print(f'\nTest Accuracy: {baseline_test_score:.4f}')
    print(f'Test F1 Score: {baseline_test_f1:.4f}')
    print(f'Test ROC-AUC: {baseline_test_auc:.4f}')

    # NEW: Hyperparameter tuning
    tuned_model = tune_hyperparameters(X_train, y_train)
    print('done tuning')

    # NEW: Cross-validation
    cv_results = evaluate_with_cross_validation(tuned_model, X_train, y_train, cv=5)
    print('done cross validation')

    # Evaluate tuned model
    print("\n" + "="*80)
    print("IMPROVED MODEL EVALUATION")
    print("="*80)

    val_score = tuned_model.score(X_val, y_val)
    val_pred = tuned_model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred)
    val_pred_proba = tuned_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)

    print(f'Validation Accuracy: {val_score:.4f}')
    print(f'Validation F1 Score: {val_f1:.4f}')
    print(f'Validation ROC-AUC: {val_auc:.4f}')

    print("\nClassification Report (Validation):")
    print(classification_report(y_val, val_pred))

    test_score = tuned_model.score(X_test, y_test)
    test_pred = tuned_model.predict(X_test)
    test_f1 = f1_score(y_test, test_pred)
    test_pred_proba = tuned_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred_proba)

    print(f'\nTest Accuracy: {test_score:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Test ROC-AUC: {test_auc:.4f}')

    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_pred))

    # Feature importance
    importances = tuned_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    print("\nTop 15 Feature Importances:")
    print(feature_importance_df.head(15).to_string(index=False))

    # Comparison
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    print(f"Baseline Validation AUROC: {baseline_val_auc:.4f}")
    print(f"Improved Validation AUROC: {val_auc:.4f}")
    print(f"Improvement: {val_score - baseline_val_score:+.4f}")

    print(f"Baseline Testing AUROC: {baseline_test_auc:.4f}")
    print(f"Improved Testing AUROC: {test_auc:.4f}")
    print(f"Improvement: {test_score - baseline_test_score:+.4f}")

    # Save tuned model
    with open("improved_model.pkl", "wb") as f:
        pickle.dump(tuned_model, f)
    print("\nImproved model saved to: improved_model.pkl")

    # Save feature importance
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to: feature_importance.csv")


def main():
    """
    Main preprocessing pipeline
    """
    # Configuration
    features_file = 'data/t2d_phase1_featured.csv'
    labels_file = 'data/t2d_DRUG_trials_phase1_labels.csv'
    output_dir = 'data'

    # Load data
    df = load_data(features_file)
    labels = load_labels(labels_file)

    # Merge with labels
    df = merge_with_labels(df, labels)

    # Drop unnecessary columns
    df = drop_unnecessary_columns(df)

    # Categorize features
    df, categorical_cols, continuous_cols, binary_cols = prepare_features_for_modeling(df)

    # One-hot encode categorical
    df = apply_one_hot_encoding(df, categorical_cols)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, ids_train, ids_val, ids_test = split_data(
        df, test_size=0.2, val_size=0.2, random_state=42
    )

    # Scale continuous features
    X_train, X_val, X_test, scaler = apply_scaling(
        X_train, X_val, X_test, continuous_cols
    )

    # Save everything
    save_processed_data(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        ids_train, ids_val, ids_test,
        scaler,
        output_dir=output_dir
    )

    # Build and evaluate model
    model_building()


if __name__ == '__main__':
    main()
