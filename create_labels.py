import pandas as pd
import numpy as np
import json


def load_data(path):
    return pd.read_csv(path)

def show_data(data):
    print(data.head())

    print("Data Structure")
    print("---------------")
    print(f"Dimensions: {data.shape}")
    print(f"Data Types:\n{data.dtypes}")
    print(f"Missing Values:\n{data.isnull().sum()}")

    #prints all unique values in each column and their frequencies
    #for col in ['nct_id','title','official_title','status','start_date','completion_date','primary_completion_date','phase','enrollment','study_type']:
    #    print(f"\n{col}: {data[col].nunique()} unique values")
    #    print(data[col].value_counts())



def find_progressed_trials(phase1, phase2):
    """
    Identifies trials that have progressed from Phase 1 to Phase 2.
    
    Uses multiple methods:
    1. Same sponsor + similar intervention names
    2. Same title (rare, but possible if updated)
    
    Parameters:
    - df: DataFrame with columns including 'nct_id', 'title', 'phase', 
          'sponsor', 'intervention_names', 'start_date'
    
    Returns:
    - DataFrame with likely Phase 1 -> Phase 2 progressions
    """
    
    # Separate Phase 1 and Phase 2 trials
    #phase1 = df[df['phase'].str.contains('PHASE1', case=False, na=False)].copy()
    phase2 = phase2[phase2['phase'].str.contains('PHASE2', case=False, na=False)].copy()
    
    print(f"Phase 1 trials: {len(phase1)}")
    print(f"Phase 2 trials: {len(phase2)}")
    
    matches = []
    success_results = pd.DataFrame(phase1[['nct_id']])
    print(success_results.shape)
    
    # Match by sponsor + intervention_names
    for _, p1_trial in phase1.iterrows():
        p1_sponsor = str(p1_trial.get('sponsor', '')).lower().strip()
        p1_interventions = str(p1_trial.get('intervention_names', '')).lower().strip()
        p1_start = p1_trial.get('start_date', '')
        p1_nct_id = p1_trial.get('nct_id', '')
        
        if not p1_sponsor or not p1_interventions or p1_interventions == 'nan':
            continue
        
        # Find Phase 2 trials with same sponsor and similar interventions
        for _, p2_trial in phase2.iterrows():
            p2_sponsor = str(p2_trial.get('sponsor', '')).lower().strip()
            p2_interventions = str(p2_trial.get('intervention_names', '')).lower().strip()
            p2_start = p2_trial.get('start_date', '')
            
            # Check if sponsor matches
            if p1_sponsor != p2_sponsor:
                continue
            
            # Check if interventions overlap (not exact match due to possible naming variations)
            if p1_interventions in p2_interventions or p2_interventions in p1_interventions:
                # Check that Phase 2 started after Phase 1
                try:
                    p1_date = pd.to_datetime(p1_start, errors='coerce')
                    p2_date = pd.to_datetime(p2_start, errors='coerce')
                    
                    # Skip if dates are invalid or Phase 2 didn't start after Phase 1
                    if pd.isna(p1_date) or pd.isna(p2_date) or p2_date <= p1_date:
                        continue
                except:
                    continue
                
                success_results.loc[success_results['nct_id'] == p1_nct_id, 'success'] = 1

                matches.append({
                    'phase1_nct_id': p1_trial['nct_id'],
                    'phase1_title': p1_trial.get('title', ''),
                    'phase1_start_date': p1_start,
                    'phase2_nct_id': p2_trial['nct_id'],
                    'phase2_title': p2_trial.get('title', ''),
                    'phase2_start_date': p2_start,
                    'sponsor': p1_sponsor,
                    'intervention': p1_interventions,
                    'match_method': 'sponsor + intervention'
                })
    
    results_df = pd.DataFrame(matches)
    
    # Remove duplicates
    if len(results_df) > 0:
        results_df = results_df.drop_duplicates(subset=['phase1_nct_id', 'phase2_nct_id'])
    
    print(f"\nFound {len(results_df)} potential Phase 1 -> Phase 2 progressions")
    
    return results_df, success_results

if __name__ == '__main__':
    data1 = load_data('data/t2d_trials_phase1.csv')
    data23 = load_data('data/t2d_trials_phase2_3.csv')
    data_all = load_data('data/t2d_trials_all_phases.csv')

    data1 = data1[data1['intervention_types'].str.contains('DRUG')]
    data23 = data23[data23['intervention_types'].str.contains('DRUG')]


    show_data(data1)
    progressions, success = find_progressed_trials(data1, data23) # 290 matches

    unique_phase1_trials = progressions['phase1_nct_id'].nunique()
    print(f"Unique Phase 1 trials: {unique_phase1_trials}") #125

    success = success.fillna(0)
    print(success[success['success']==1]) #also 125

    data1.to_csv("data/t2d_DRUGtrials_phase1.csv")
    success.to_csv("data/t2d_DRUGtrials_phase1_labels.csv")

