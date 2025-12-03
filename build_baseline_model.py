"""
build_model.py
This script performs the following steps to prepare data for modeling:
- Drops unnecessary columns
- Merges with labels
- One-hot encoding
- Scaling
- Train/val/test split

and then builds a baseline Random Forest model for classification.
"""

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
        'title',
        
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

def model_building(X_train, X_val, X_test, y_train, y_val, y_test):
    # Load data

    # Separate features and target
    #X_train = train.drop(columns=['nct_id', 'success'])
    y_train = y_train.values

    #X_val = val.drop(columns=['nct_id', 'success'])
    y_val = y_val.values

    #X_test = test.drop(columns=['nct_id', 'success'])
    y_test = y_test.values

    # Train baseline model

    #model = RandomForestClassifier()
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    # save
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    # Validate model
    val_score = model.score(X_val, y_val)
    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred)
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)

    print(f'Validation Accuracy: {val_score:.4f}')
    print(f'Validation F1 Score: {val_f1:.4f}')
    print(f'Validation ROC-AUC: {val_auc:.4f}')

    # feature importance
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    print("Feature Importances:")
    print(feature_importance_df)

    # Test model
    test_score = model.score(X_test, y_test)
    test_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, test_pred)
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred_proba)

    print(f'\nTest Accuracy: {test_score:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Test ROC-AUC: {test_auc:.4f}')


def main():
    """
    Main preprocessing pipeline
    """
    # Configuration
    features_file = 'data/t2d_phase1_cleaned.csv'
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
    
    # Build and evaluate model
    model_building(X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == '__main__':
    main()
