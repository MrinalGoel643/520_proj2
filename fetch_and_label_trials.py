'''
This script fetches clinical trials related to Type 2 Diabetes from ClinicalTrials.gov,
filters them to include only those that are completed and have posted results
'''

# imports

import requests
import pandas as pd
import time
import json
from typing import List, Dict

def fetch_trials(condition: str) -> List[Dict]:
    """
    Fetch ALL trials from ClinicalTrials.gov API (no limit)
    NO FILTERS - gets all trials regardless of status or results
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    all_trials = []
    page_token = None
    page_size = 1000  # Max allowed by API
    
    print(f"Fetching ALL {condition} trials (no filters, no limits)...")
    print("This may take a while...\n")
    
    while True:  # Continue until no more pages
        params = {
            "query.cond": condition,
            # NO status filter - get all trials
            # NO results filter - get all trials
            "pageSize": page_size,
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
            completed = sum(1 for t in all_trials if t.get("protocolSection", {}).get("statusModule", {}).get("overallStatus") == "COMPLETED")
            print(f"  Fetched {len(all_trials)} trials (completed: {completed}, with results: {with_results})...")
            
            # Check if there are more pages
            if "nextPageToken" in data and len(studies) == page_size:
                page_token = data["nextPageToken"]
                time.sleep(0.5)  # Be nice to the API
            else:
                # No more pages
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
    
    print(f"\nTotal fetched: {len(all_trials)} trials")
    
    # Return ALL trials (no filtering)
    return all_trials


def extract_results_data(trial: Dict) -> Dict:
    """
    Extract results data from the trial (outcome measures, analyses, p-values)
    Returns dictionary with results information
    """
    results_data = {
        "has_results": False,
        "num_outcome_measures": 0,
        "num_primary_outcomes": 0,
        "num_secondary_outcomes": 0,
        "primary_outcome_titles": None,
        "primary_outcome_descriptions": None,
        "primary_outcome_timeframes": None,
        "primary_analyses_data": None,  # Will store JSON string of all analyses
        "num_p_values_in_primary": 0,
        "all_p_values_primary": None,
        "all_param_values_primary": None,
        "all_ci_data_primary": None,
        "baseline_participants": None,
    }
    
    if "resultsSection" not in trial:
        return results_data
    
    results = trial["resultsSection"]
    results_data["has_results"] = True
    
    # Extract baseline participant info
    if "baselineCharacteristicsModule" in results:
        baseline = results["baselineCharacteristicsModule"]
        if "denoms" in baseline:
            denoms = baseline["denoms"]
            if denoms and len(denoms) > 0:
                counts = denoms[0].get("counts", [])
                if counts:
                    try:
                        # Convert all values to int before summing
                        total_participants = sum(int(c.get("value", 0)) for c in counts if c.get("value"))
                        results_data["baseline_participants"] = total_participants
                    except (ValueError, TypeError):
                        # If conversion fails, skip
                        pass
    
    # Extract outcome measures
    if "outcomeMeasuresModule" not in results:
        return results_data
    
    outcome_module = results["outcomeMeasuresModule"]
    if "outcomeMeasures" not in outcome_module:
        return results_data
    
    measures = outcome_module["outcomeMeasures"]
    results_data["num_outcome_measures"] = len(measures)
    
    # Separate primary and secondary outcomes
    primary_measures = [m for m in measures if m.get("type") == "PRIMARY"]
    secondary_measures = [m for m in measures if m.get("type") == "SECONDARY"]
    
    results_data["num_primary_outcomes"] = len(primary_measures)
    results_data["num_secondary_outcomes"] = len(secondary_measures)
    
    if not primary_measures:
        return results_data
    
    # Extract PRIMARY outcome information
    primary_titles = []
    primary_descriptions = []
    primary_timeframes = []
    all_primary_analyses = []
    all_p_values = []
    all_param_values = []
    all_ci_data = []
    
    for primary in primary_measures:
        # Basic outcome info
        primary_titles.append(primary.get("title", ""))
        primary_descriptions.append(primary.get("description", ""))
        primary_timeframes.append(primary.get("timeFrame", ""))
        
        # Extract analyses from primary outcome
        analyses = []
        
        # Location 1: primary -> analyses (most common)
        if "analyses" in primary:
            analyses = primary["analyses"]
        
        # Location 2: primary -> classes -> categories -> analyses (alternative structure)
        elif "classes" in primary:
            for class_item in primary["classes"]:
                if "categories" in class_item:
                    for category in class_item["categories"]:
                        if "analyses" in category:
                            analyses.extend(category["analyses"])
        
        # Extract p-values and effect sizes from analyses
        for analysis in analyses:
            all_primary_analyses.append(analysis)
            
            # P-value
            if "pValue" in analysis:
                all_p_values.append(analysis["pValue"])
            
            # Effect size (paramValue)
            if "paramValue" in analysis:
                all_param_values.append(analysis["paramValue"])
            
            # Confidence intervals
            if "ciLowerLimit" in analysis and "ciUpperLimit" in analysis:
                ci_info = f"[{analysis['ciLowerLimit']}, {analysis['ciUpperLimit']}]"
                all_ci_data.append(ci_info)
    
    # Store as comma-separated strings or JSON
    results_data["primary_outcome_titles"] = " | ".join(primary_titles)
    results_data["primary_outcome_descriptions"] = " | ".join(primary_descriptions)
    results_data["primary_outcome_timeframes"] = " | ".join(primary_timeframes)
    
    # Store all analyses as JSON string for detailed inspection
    if all_primary_analyses:
        results_data["primary_analyses_data"] = json.dumps(all_primary_analyses)
    
    results_data["num_p_values_in_primary"] = len(all_p_values)
    
    if all_p_values:
        results_data["all_p_values_primary"] = ", ".join(all_p_values)
    
    if all_param_values:
        results_data["all_param_values_primary"] = ", ".join(all_param_values)
    
    if all_ci_data:
        results_data["all_ci_data_primary"] = " | ".join(all_ci_data)
    
    return results_data


def extract_trial_info(trial: Dict) -> Dict:
    """
    Extract basic trial information + results data
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
        
        # Medical info
        "conditions": ", ".join(conditions.get("conditions", [])),
        "intervention_types": ", ".join(intervention_types),
        "intervention_names": ", ".join(intervention_names),
        
        # Sponsor
        "sponsor": sponsor.get("leadSponsor", {}).get("name", ""),
        "sponsor_type": sponsor.get("leadSponsor", {}).get("class", ""),
    }
    
    # Add results data
    results_data = extract_results_data(trial)
    trial_info.update(results_data)
    
    return trial_info


def main():
    """
    Main function - fetch T2D trials and create THREE CSV files
    """
    
    # Fetch ALL trials (no limit)
    condition = "Type 2 Diabetes"
    trials = fetch_trials(condition)
    
    if not trials:
        print("\nNo trials found!")
        return
    
    print(f"\nProcessing {len(trials)} trials...")
    
    # Extract info from each trial
    trial_data = []
    for i, trial in enumerate(trials):
        if (i + 1) % 100 == 0:
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
    
    # Debug: Check if dataframe is valid
    if df.empty:
        print("ERROR: DataFrame is empty!")
        return
    # Check if phase column exists
    if 'phase' not in df.columns:
        print("ERROR: 'phase' column not found in DataFrame!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Save File 1: ALL PHASES (everything - no filtering)
    all_phases_file = "t2d_trials_all_phases.csv"
    df.to_csv(all_phases_file, index=False)

    # Save File 2: PHASE 1 ONLY
    phase_1_df = df[df["phase"].isin(["PHASE1", "PHASE1_PHASE2"])]
    phase_1_file = "t2d_trials_phase1.csv"
    phase_1_df.to_csv(phase_1_file, index=False)
    
    # Save File 3: PHASE 2 and 3
    phase_2_3_df = df[df["phase"].isin(["PHASE2", "PHASE3", "PHASE2_PHASE3"])]
    phase_2_3_file = "t2d_trials_phase2_3.csv"
    phase_2_3_df.to_csv(phase_2_3_file, index=False)
    
    status_counts = df['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status:20s} {count:4d}")


if __name__ == "__main__":
    main()