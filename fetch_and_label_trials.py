'''
This script fetches clinical trials related to Type 2 Diabetes from ClinicalTrials.gov,
filters them to include only those that are completed and have posted results
'''

# imports

import requests
import pandas as pd
import time
from typing import List, Dict

def fetch_trials(condition: str, max_trials: int = 1000) -> List[Dict]:
    """
    Fetch trials from ClinicalTrials.gov API
    Filters: COMPLETED status only
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    all_trials = []
    page_token = None
    page_size = 100
    
    while len(all_trials) < max_trials:
        params = {
            "query.cond": condition,
            "filter.overallStatus": "COMPLETED",  # Only completed trials
            "pageSize": min(page_size, max_trials - len(all_trials)),
            "format": "json"
        }
        
        if page_token:
            params["pageToken"] = page_token
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "studies" not in data:
                break
            
            studies = data["studies"]
            all_trials.extend(studies)
            
            with_results = sum(1 for t in all_trials if "resultsSection" in t)
            print(f"  Fetched {len(all_trials)} trials ({with_results} with results)...")
            
            if "nextPageToken" in data and len(studies) == page_size:
                page_token = data["nextPageToken"]
                time.sleep(0.5)
            else:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
    
    print(f"\nTotal fetched: {len(all_trials)} trials")
    
    # Filter to only trials with results
    trials_with_results = [t for t in all_trials if "resultsSection" in t]
    print(f"Trials with posted results: {len(trials_with_results)}")
    
    return trials_with_results


def extract_trial_info(trial: Dict) -> Dict:
    """
    Extract basic trial information - NO LABELING
    """
    protocol = trial.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    conditions = protocol.get("conditionsModule", {})
    interventions = protocol.get("armsInterventionsModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})
    
    # Extract phase
    phases = design.get("phases", [])
    phase = phases[0] if phases else "NOT_APPLICABLE"
    
    # Extract enrollment
    enrollment_info = design.get("enrollmentInfo", {})
    enrollment = enrollment_info.get("count", None)
    
    # Extract interventions
    intervention_list = interventions.get("interventions", [])
    intervention_types = [i.get("type", "") for i in intervention_list]
    intervention_names = [i.get("name", "") for i in intervention_list]
    
    # Extract design details
    design_info = design.get("designInfo", {})
    masking_info = design_info.get("maskingInfo", {})
    
    # Basic trial info
    trial_info = {
        # Identification
        "nct_id": identification.get("nctId", ""),
        "title": identification.get("briefTitle", ""),
        "official_title": identification.get("officialTitle", ""),
        
        # Status
        "status": status.get("overallStatus", ""),
        "start_date": status.get("startDateStruct", {}).get("date", ""),
        "completion_date": status.get("completionDateStruct", {}).get("date", ""),
        "primary_completion_date": status.get("primaryCompletionDateStruct", {}).get("date", ""),
        
        # Design
        "phase": phase,
        "enrollment": enrollment,
        "study_type": design.get("studyType", ""),
        "allocation": design_info.get("allocation", ""),
        "intervention_model": design_info.get("interventionModel", ""),
        "primary_purpose": design_info.get("primaryPurpose", ""),
        "masking": masking_info.get("masking", ""),
        "masking_description": masking_info.get("maskingDescription", ""),
        
        # Medical info
        "conditions": ", ".join(conditions.get("conditions", [])),
        "intervention_types": ", ".join(intervention_types),
        "intervention_names": ", ".join(intervention_names),
        
        # Sponsor
        "sponsor": sponsor.get("leadSponsor", {}).get("name", ""),
        "sponsor_type": sponsor.get("leadSponsor", {}).get("class", ""),
        
        # Results availability
        "has_results": "resultsSection" in trial,
    }
    
    return trial_info


def main():
    """
    Main function - fetch T2D trials and create two CSV files
    """
    
    # Fetch trials
    condition = "Type 2 Diabetes"
    trials = fetch_trials(condition, max_trials=1000)
    
    if not trials:
        print("\nNo trials found!")
        return
    
    print(f"\nProcessing {len(trials)} trials with results...")
    
    # Extract info from each trial
    trial_data = []
    for i, trial in enumerate(trials):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(trials)}...")
        
        try:
            info = extract_trial_info(trial)
            if info:
                trial_data.append(info)
        except Exception as e:
            nct_id = trial.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "Unknown")
            print(f"  Error processing {nct_id}: {e}")
            continue
    
    print(f"\nSuccessfully processed: {len(trial_data)} trials")
    
    # Create DataFrame
    df = pd.DataFrame(trial_data)
    
    # Save File 1: ALL PHASES
    all_phases_file = "data/t2d_trials_all_phases.csv"
    df.to_csv(all_phases_file, index=False)

    # Save File 2: PHASE 2 and 3 ONLY
    phase_2_3_df = df[df["phase"].isin(["PHASE2", "PHASE3", "PHASE2_PHASE3"])]
    phase_2_3_file = "data/t2d_trials_phase2_3_SIMPLE.csv"
    phase_2_3_df.to_csv(phase_2_3_file, index=False)



if __name__ == "__main__":
    main()