"""
This script performs the following steps:
1. Drop columns with excessive missing values and not useful for Phase 1 prediction
2. Drop rows with missing start_date
3. Handle completion date missing values
4. Impute missing values in key design fields
"""

# imports

import pandas as pd
import numpy as np


def load_data(filepath):
    """Load clinical trial data"""
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} trials from {filepath}")
    print(f"Columns: {len(df.columns)}")
    
    return df


def drop_unnecessary_columns(df):
    """
    Step 1: Drop results-related columns that are mostly missing
    and not useful for Phase 1 prediction
    """
  
    columns_to_drop = [
        'primary_outcome_titles',
        'primary_outcome_descriptions', 
        'primary_outcome_timeframes',
        'primary_analyses_data',
        'all_p_values_primary',
        'all_param_values_primary',
        'all_ci_data_primary',
        'baseline_participants',
        'official_title'
    ]
    
    # Check which columns actually exist
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    missing_cols = [col for col in columns_to_drop if col not in df.columns]
    
    if missing_cols:
        print(f"\nWarning: These columns not found in data: {missing_cols}")
    
    print(f"\nDropping {len(existing_cols)} columns:")
    for col in existing_cols:
        missing_pct = 100 * df[col].isna().sum() / len(df)
        print(f"  - {col}: {df[col].isna().sum()} missing ({missing_pct:.1f}%)")
    
    df = df.drop(columns=existing_cols)
    
    print(f"\nRemaining columns: {len(df.columns)}")
    
    return df


def drop_missing_start_date(df):
    """
    Step 2: Drop rows with missing start_date
    Critical for calculating time-based features
    """
    
    missing_start = df['start_date'].isna().sum()
    print(f"\nTrials missing start_date: {missing_start}")
    
    if missing_start > 0:
        print(f"Percentage: {100 * missing_start / len(df):.1f}%")
        
        # Show what we're dropping
        print("\nTrials being dropped:")
        dropped = df[df['start_date'].isna()][['nct_id', 'title', 'status']]
        print(dropped.to_string(index=False))
        
        # Drop rows
        df = df[df['start_date'].notna()].copy()
        
        print(f"\nDropped {missing_start} trials")
        print(f"Remaining: {len(df)} trials")
    else:
        print("\nNo trials missing start_date")
    
    return df


def handle_completion_dates(df):
    """
    Step 3: Handle missing completion dates
    - Fill primary_completion_date with completion_date (and vice versa)
    - Drop rows where BOTH are missing
    """
    
    # Initial missing counts
    missing_primary = df['primary_completion_date'].isna().sum()
    missing_completion = df['completion_date'].isna().sum()
    both_missing = (df['primary_completion_date'].isna() & df['completion_date'].isna()).sum()
    
    print(f"\nBefore filling:")
    print(f"  Missing primary_completion_date: {missing_primary}")
    print(f"  Missing completion_date: {missing_completion}")
    print(f"  Missing BOTH: {both_missing}")
    
    # Fill primary with completion
    filled_primary = df['primary_completion_date'].isna() & df['completion_date'].notna()
    df.loc[filled_primary, 'primary_completion_date'] = df.loc[filled_primary, 'completion_date']
    print(f"\nFilled {filled_primary.sum()} primary_completion_date values using completion_date")
    
    # Fill completion with primary
    filled_completion = df['completion_date'].isna() & df['primary_completion_date'].notna()
    df.loc[filled_completion, 'completion_date'] = df.loc[filled_completion, 'primary_completion_date']
    print(f"Filled {filled_completion.sum()} completion_date values using primary_completion_date")

    # Check remaining
    still_both_missing = (df['primary_completion_date'].isna() & df['completion_date'].isna()).sum()
    
    if still_both_missing > 0:
        print(f"\n{still_both_missing} trials still missing BOTH dates - drop these")
    
        # Drop
        df = df[~(df['primary_completion_date'].isna() & df['completion_date'].isna())].copy()
        
        print(f"\nDropped {still_both_missing} trials")
    else:
        print("\nNo trials missing both dates")
    
    print(f"Remaining: {len(df)} trials")
    
    return df

def filter_to_labelable_trials(df):
    """
    Keep only trials that have finished and can be labeled
    """
    print("\nFiltering to labelable trials...")
    
    labelable_statuses = ['COMPLETED', 'TERMINATED', 'WITHDRAWN']
    
    before = len(df)
    df = df[df['status'].isin(labelable_statuses)].copy()
    after = len(df)
    
    print(f"Before: {before} trials")
    print(f"After: {after} trials")
    print(f"Dropped: {before - after} trials ({100*(before-after)/before:.1f}%)")
    
    return df


def impute_missing_values(df):
    """
    Step 4: Impute missing values in key design fields
    Using domain expertise for best strategy
    """
    # 4.1 ENROLLMENT - Impute with median by sponsor_type
    missing_enrollment = df['enrollment'].isna().sum()
    print(f"Missing values: {missing_enrollment}")
    
    if missing_enrollment > 0:
        # Create missing flag
        df['enrollment_missing'] = df['enrollment'].isna().astype(int)
        
        # Impute with median by sponsor_type
        df['enrollment'] = df.groupby('sponsor_type')['enrollment'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # If still missing (rare), use overall median
        overall_median = df['enrollment'].median()
        df['enrollment'] = df['enrollment'].fillna(overall_median)
        
        print(f"Imputed {missing_enrollment} values using sponsor_type median")
        print(f"Created 'enrollment_missing' flag")
    else:
        df['enrollment_missing'] = 0
        print("No missing values")
    
    
    # 4.2 ALLOCATION - Fill with "UNKNOWN"
    missing_allocation = df['allocation'].isna().sum()
    print(f"Missing values: {missing_allocation}")
    
    if missing_allocation > 0:
        df['allocation'] = df['allocation'].fillna('UNKNOWN')
        print(f"Filled {missing_allocation} values with 'UNKNOWN' category")
    else:
        print("No missing values")
    
    print(f"Value distribution:\n{df['allocation'].value_counts()}")
    
    
    # 4.3 INTERVENTION_MODEL - Fill with "UNKNOWN"
    missing_model = df['intervention_model'].isna().sum()
    print(f"Missing values: {missing_model}")
    
    if missing_model > 0:
        df['intervention_model'] = df['intervention_model'].fillna('UNKNOWN')
        print(f"Filled {missing_model} values with 'UNKNOWN' category")
    else:
        print("No missing values")
    
    print(f"Value distribution:\n{df['intervention_model'].value_counts()}")
    
    
    # 4.4 PRIMARY_PURPOSE - Fill with "TREATMENT"
    missing_purpose = df['primary_purpose'].isna().sum()
    print(f"Missing values: {missing_purpose}")
    
    if missing_purpose > 0:
        df['primary_purpose'] = df['primary_purpose'].fillna('TREATMENT')
        print(f"Filled {missing_purpose} values with 'TREATMENT' (standard for diabetes drug trials)")
    else:
        print("No missing values")
    
    print(f"Value distribution:\n{df['primary_purpose'].value_counts()}")
    
    
    # 4.5 MASKING - Fill with "NONE"
    missing_masking = df['masking'].isna().sum()
    print(f"Missing values: {missing_masking}")
    
    if missing_masking > 0:
        df['masking'] = df['masking'].fillna('NONE')
        print(f"Filled {missing_masking} values with 'NONE' (conservative assumption)")
    else:
        print("No missing values")
    
    print(f"Value distribution:\n{df['masking'].value_counts()}")
    
    return df


def final_statistics(df, original_count):
    """
    Report final dataset statistics
    """
    print("\n" + "="*80)
    print("FINAL DATASET STATISTICS")
    print("="*80)
    
    print(f"\nOriginal trials: {original_count}")
    print(f"Final trials: {len(df)}")
    print(f"Retention rate: {100 * len(df) / original_count:.1f}%")
    print(f"Dropped: {original_count - len(df)} trials ({100 * (original_count - len(df)) / original_count:.1f}%)")
    
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Check remaining missing values
    print("\n" + "="*80)
    print("REMAINING MISSING VALUES")
    print("="*80)
    
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) > 0:
        print("\nColumns with missing values:")
        for col, count in missing.items():
            pct = 100 * count / len(df)
            print(f"  {col:40s}: {count:4d} ({pct:5.1f}%)")
    else:
        print("\nNO MISSING VALUES!")
    
    return df


def save_cleaned_data(df, output_path):
    """
    Save cleaned dataset
    """
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"{len(df)} trials, {len(df.columns)} columns")
    
    return df


def main():
    """
    Main preprocessing pipeline
    """
    # Configuration
    input_file = 'data/t2d_DRUG_trials_phase1.csv'
    output_file = 'data/t2d_phase1_cleaned.csv'
    
    # Load data
    df = load_data(input_file)
    original_count = len(df)
    
    # Step 1: Drop unnecessary columns
    df = drop_unnecessary_columns(df)

    # drop to labelable trials
    df = filter_to_labelable_trials(df)
    
    # Step 2: Drop missing start_date
    df = drop_missing_start_date(df)
    
    # Step 3: Handle completion dates
    df = handle_completion_dates(df)
    
    # Step 4: Impute missing values
    df = impute_missing_values(df)
    
    # Final statistics
    df = final_statistics(df, original_count)
    
    # Save
    df = save_cleaned_data(df, output_file)
    
    return df


if __name__ == '__main__':
    df_cleaned = main()