"""
this script performs feature engineering on the cleaned clinical trials dataset.

Features created:
1. trial_duration_months - How long the trial took
2. is_top_pharma - Is sponsor a top diabetes pharma company?
3. sponsor_trial_count - How many trials has this sponsor run?
4. has_placebo - Does trial have placebo control?
5. design_quality_score - Overall design quality (0-5)
"""

import pandas as pd
import numpy as np


def parse_incomplete_dates(date_series):
    """
    Parse dates, handling cases where day is missing (YYYY-MM format)
    """
    def parse_single_date(date_str):
        if pd.isna(date_str):
            return pd.NaT
        
        date_str = str(date_str).strip()
        
        # Count hyphens to determine format
        if date_str.count('-') == 1:
            # Format: YYYY-MM (missing day)
            # Add day 15 (middle of month) as approximation
            date_str = date_str + '-15'
        elif date_str.count('-') == 0:
            # Format: YYYY only (very rare)
            date_str = date_str + '-07-01'  # Middle of year
        
        return pd.to_datetime(date_str, errors='coerce')
    
    return date_series.apply(parse_single_date)


def load_cleaned_data(filepath):
    
    df = pd.read_csv(filepath)
    
    return df


def create_features(df):
    """
    Create selected features
    """

    df = df.copy()
    
    # Convert dates to datetime with special handling for incomplete dates
    print("\nParsing dates (handling YYYY-MM format)...")
    df['start_date'] = parse_incomplete_dates(df['start_date'])
    df['completion_date'] = parse_incomplete_dates(df['completion_date'])
    df['primary_completion_date'] = parse_incomplete_dates(df['primary_completion_date'])
    
    
    # ========================================================================
    # FEATURE 1: trial_duration_months
    # ========================================================================
    
    df['trial_duration_months'] = (
        (df['primary_completion_date'] - df['start_date']).dt.days / 30.44
    )
    
    # If primary completion missing, use regular completion
    df['trial_duration_months'] = df['trial_duration_months'].fillna(
        (df['completion_date'] - df['start_date']).dt.days / 30.44
    )

    df['trial_duration_months'] = df['trial_duration_months'].round(0)
    
    # ========================================================================
    # FEATURE 2: is_top_pharma
    # ========================================================================
    
    top_diabetes_pharma = [
        'novo nordisk', 'eli lilly', 'lilly', 'sanofi', 'astrazeneca', 
        'boehringer ingelheim', 'boehringer', 'merck', 'pfizer', 'novartis',
        'glaxosmithkline', 'gsk', 'janssen', 'takeda', 'bristol-myers squibb',
        'bristol-myers', 'bms', 'amgen', 'roche', 'bayer'
    ]
    
    df['is_top_pharma'] = df['sponsor'].str.lower().apply(
        lambda x: int(any(company in str(x).lower() for company in top_diabetes_pharma))
    )
    
    # ========================================================================
    # FEATURE 3: sponsor_trial_count
    # ========================================================================
    
    sponsor_counts = df['sponsor'].value_counts()
    df['sponsor_trial_count'] = df['sponsor'].map(sponsor_counts)
    
    # ========================================================================
    # FEATURE 4: has_placebo
    # ========================================================================
    
    
    df['has_placebo'] = df['intervention_names'].str.lower().str.contains('placebo', na=False).astype(int)
    
    # ========================================================================
    # FEATURE 5: design_quality_score
    # ========================================================================
    design_score = 0
    
    # +1 if randomized
    design_score += (df['allocation'] == 'RANDOMIZED').astype(int)
    
    # +1 if double-blind or better
    design_score += df['masking'].isin(['DOUBLE', 'TRIPLE', 'QUADRUPLE']).astype(int)
    
    # +1 if parallel design
    design_score += (df['intervention_model'] == 'PARALLEL').astype(int)
    
    # +1 if treatment purpose
    design_score += (df['primary_purpose'] == 'TREATMENT').astype(int)
    
    # +1 if has placebo
    design_score += df['has_placebo']
    
    df['design_quality_score'] = design_score

    
    return df



def save_featured_data(df, output_path):
    """
    Save dataset with engineered features
    """
    df.to_csv(output_path, index=False)
    
    return df


def main():
    """
    Main feature engineering pipeline
    """
    # Configuration
    input_file = 'data/t2d_phase1_cleaned.csv'
    output_file = 'data/t2d_phase1_featured.csv'
    
    # Load cleaned data
    df = load_cleaned_data(input_file)

    df.drop(columns=['title'],inplace=True)
    
    # Create features
    df = create_features(df)
    
    # Save
    df = save_featured_data(df, output_file)

    return df


if __name__ == '__main__':
    df_featured = main()