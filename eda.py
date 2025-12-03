"""
eda.py
Exploratory Data Analysis on the clinical trial data that we extracted
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_raw_data():
    """Load cleaned data (before feature engineering)"""
    print("="*80)
    print("LOADING RAW DATA FOR EDA")
    print("="*80)
    
    # Load cleaned data (after preprocessing, before feature engineering)
    df = pd.read_csv('data/t2d_phase1_cleaned.csv')
    
    # Load labels
    labels = pd.read_csv('data/t2d_DRUG_trials_phase1_labels.csv')
    
    # Merge
    df = df.merge(labels[['nct_id', 'success']], on='nct_id', how='inner')
    
    # Filter to COMPLETED only
    df = df[df['status'] == 'COMPLETED'].copy()
    
    print(f"\nTotal trials: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"\nSuccess rate: {100 * df['success'].mean():.1f}%")
    print(f"  Success (1): {(df['success'] == 1).sum()}")
    print(f"  Failure (0): {(df['success'] == 0).sum()}")
    
    return df


def plot_target_distribution(df, output_dir='figures'):
    """
    Plot 1: Target variable distribution
    """
    print("\n" + "="*80)
    print("PLOT 1: TARGET VARIABLE DISTRIBUTION")
    print("="*80)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Count plot
    success_counts = df['success'].value_counts().sort_index()
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax.bar(['Did NOT Progress\nto Phase 2', 'Progressed\nto Phase 2'], 
                  success_counts.values, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add values and percentages
    total = len(df)
    for bar, count in zip(bars, success_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count} trials\n({100*count/total:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Trials', fontsize=12, fontweight='bold')
    ax.set_title('Phase 1 Trial Outcomes', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(success_counts.values) * 1.2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_target_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/01_target_distribution.png")
    plt.close()


def plot_top_sponsors_by_volume(df, output_dir='figures'):
    """
    Plot 2: Top 5 sponsors with most trials
    """
    print("\n" + "="*80)
    print("PLOT 2: TOP 5 SPONSORS BY TRIAL COUNT")
    print("="*80)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get top 5 sponsors by trial count
    top_sponsors = df['sponsor'].value_counts().head(5)
    
    bars = ax.barh(range(len(top_sponsors)), top_sponsors.values, 
                   color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add counts
    for i, (sponsor, count) in enumerate(top_sponsors.items()):
        ax.text(count + 0.3, i, f'{count} trials', 
               va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(top_sponsors)))
    ax.set_yticklabels(top_sponsors.index, fontsize=10)
    ax.set_xlabel('Number of Trials', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Sponsors by Number of Trials', fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_top_sponsors_by_volume.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/02_top_sponsors_by_volume.png")
    plt.close()


def plot_top_sponsors_by_success(df, output_dir='figures'):
    """
    Plot 3: Top 5 sponsors with most successful trials
    """
    print("\n" + "="*80)
    print("PLOT 3: TOP 5 SPONSORS BY SUCCESSFUL TRIALS")
    print("="*80)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Count successful trials per sponsor
    successful_by_sponsor = df[df['success'] == 1].groupby('sponsor').size().sort_values(ascending=False).head(5)
    
    bars = ax.barh(range(len(successful_by_sponsor)), successful_by_sponsor.values,
                   color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add counts
    for i, (sponsor, count) in enumerate(successful_by_sponsor.items()):
        ax.text(count + 0.2, i, f'{count} successful', 
               va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(successful_by_sponsor)))
    ax.set_yticklabels(successful_by_sponsor.index, fontsize=10)
    ax.set_xlabel('Number of Successful Trials', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Sponsors by Number of Successful Trials\n(Progressed to Phase 2)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_top_sponsors_by_success.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/03_top_sponsors_by_success.png")
    plt.close()


def plot_enrollment_variation(df, output_dir='figures'):
    """
    Plot 4: Enrollment variation across all trials (no outcome distinction)
    """
    print("\n" + "="*80)
    print("PLOT 4: ENROLLMENT VARIATION")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Histogram
    ax1 = axes[0]
    
    ax1.hist(df['enrollment'], bins=40, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(df['enrollment'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df["enrollment"].mean():.0f}')
    ax1.axvline(df['enrollment'].median(), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {df["enrollment"].median():.0f}')
    
    ax1.set_xlabel('Enrollment Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Trials', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Trial Enrollment', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Box plot with statistics
    ax2 = axes[1]
    
    bp = ax2.boxplot([df['enrollment']], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][0].set_alpha(0.7)
    
    # Add statistics as text
    stats_text = f"Min: {df['enrollment'].min():.0f}\n"
    stats_text += f"Q1: {df['enrollment'].quantile(0.25):.0f}\n"
    stats_text += f"Median: {df['enrollment'].median():.0f}\n"
    stats_text += f"Q3: {df['enrollment'].quantile(0.75):.0f}\n"
    stats_text += f"Max: {df['enrollment'].max():.0f}\n"
    stats_text += f"Std: {df['enrollment'].std():.0f}"
    
    ax2.text(1.25, df['enrollment'].median(), stats_text, 
            fontsize=10, va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax2.set_ylabel('Enrollment Size', fontsize=11, fontweight='bold')
    ax2.set_title('Enrollment Statistics', fontsize=12, fontweight='bold')
    ax2.set_xticks([])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Enrollment Variation Across {len(df)} Phase 1 Trials', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_enrollment_variation.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/04_enrollment_variation.png")
    plt.close()


def calculate_trial_duration(df):
    """Helper: Calculate trial duration for EDA"""
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['completion_date'] = pd.to_datetime(df['completion_date'], errors='coerce')
    df['primary_completion_date'] = pd.to_datetime(df['primary_completion_date'], errors='coerce')
    
    # Use primary completion if available, else regular completion
    df['duration_days'] = (df['primary_completion_date'] - df['start_date']).dt.days
    df['duration_days'] = df['duration_days'].fillna((df['completion_date'] - df['start_date']).dt.days)
    df['duration_months'] = df['duration_days'] / 30.44
    
    return df


def plot_duration_variation(df, output_dir='figures'):
    """
    Plot 5: Trial duration variation across all trials (no outcome distinction)
    """
    print("\n" + "="*80)
    print("PLOT 5: TRIAL DURATION VARIATION")
    print("="*80)
    
    # Calculate duration
    df = calculate_trial_duration(df)
    
    # Remove outliers for better visualization (keep for stats though)
    duration_clean = df['duration_months'].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Histogram
    ax1 = axes[0]
    
    ax1.hist(duration_clean, bins=40, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax1.axvline(duration_clean.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {duration_clean.mean():.1f} mo')
    ax1.axvline(duration_clean.median(), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {duration_clean.median():.1f} mo')
    
    # Shade typical Phase 1 range
    ax1.axvspan(6, 18, alpha=0.2, color='green', label='Typical Phase 1\n(6-18 months)')
    
    ax1.set_xlabel('Trial Duration (months)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Trials', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Trial Duration', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Box plot with statistics
    ax2 = axes[1]
    
    bp = ax2.boxplot([duration_clean], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#9b59b6')
    bp['boxes'][0].set_alpha(0.7)
    
    # Add statistics as text
    stats_text = f"Min: {duration_clean.min():.1f} mo\n"
    stats_text += f"Q1: {duration_clean.quantile(0.25):.1f} mo\n"
    stats_text += f"Median: {duration_clean.median():.1f} mo\n"
    stats_text += f"Q3: {duration_clean.quantile(0.75):.1f} mo\n"
    stats_text += f"Max: {duration_clean.max():.1f} mo\n"
    stats_text += f"Std: {duration_clean.std():.1f} mo"
    
    ax2.text(1.25, duration_clean.median(), stats_text,
            fontsize=10, va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax2.set_ylabel('Trial Duration (months)', fontsize=11, fontweight='bold')
    ax2.set_title('Duration Statistics', fontsize=12, fontweight='bold')
    ax2.set_xticks([])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Trial Duration Variation Across {len(duration_clean)} Phase 1 Trials',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_duration_variation.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/05_duration_variation.png")
    plt.close()


def plot_enrollment_distribution(df, output_dir='figures'):
    """
    Plot 6: Enrollment size distribution by outcome
    Shows: Larger trials might indicate more confidence → feature idea!
    """
    print("\n" + "="*80)
    print("PLOT 6: ENROLLMENT DISTRIBUTION BY OUTCOME")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Histogram by outcome
    ax1 = axes[0]
    
    success = df[df['success'] == 1]['enrollment']
    failure = df[df['success'] == 0]['enrollment']
    
    ax1.hist(failure, bins=30, alpha=0.6, label='Did NOT Progress', color='#e74c3c', edgecolor='black')
    ax1.hist(success, bins=30, alpha=0.6, label='Progressed to Phase 2', color='#2ecc71', edgecolor='black')
    
    ax1.axvline(failure.mean(), color='#c0392b', linestyle='--', linewidth=2, 
               label=f'Failure Mean: {failure.mean():.0f}')
    ax1.axvline(success.mean(), color='#27ae60', linestyle='--', linewidth=2,
               label=f'Success Mean: {success.mean():.0f}')
    
    ax1.set_xlabel('Enrollment Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Trials', fontsize=11, fontweight='bold')
    ax1.set_title('Enrollment Distribution by Outcome', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Box plot
    ax2 = axes[1]
    
    bp = ax2.boxplot([failure, success], labels=['Did NOT\nProgress', 'Progressed\nto Phase 2'],
                     patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], ['#e74c3c', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('Enrollment Size', fontsize=11, fontweight='bold')
    ax2.set_title('Enrollment: Success vs Failure', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Insight: Successful trials have larger enrollment on average\n→ Suggests enrollment size as predictive feature', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_enrollment_by_outcome.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/06_enrollment_by_outcome.png")
    plt.close()


def plot_trial_duration(df, output_dir='figures'):
    """
    Plot 7: Trial duration analysis by outcome
    Shows: Duration patterns differ by outcome → feature idea!
    """
    print("\n" + "="*80)
    print("PLOT 7: TRIAL DURATION BY OUTCOME")
    print("="*80)
    
    # Calculate duration
    df = calculate_trial_duration(df)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Histogram
    ax1 = axes[0]
    
    success = df[df['success'] == 1]['duration_months'].dropna()
    failure = df[df['success'] == 0]['duration_months'].dropna()
    
    ax1.hist(failure, bins=30, alpha=0.6, label='Did NOT Progress', color='#e74c3c', edgecolor='black')
    ax1.hist(success, bins=30, alpha=0.6, label='Progressed to Phase 2', color='#2ecc71', edgecolor='black')
    
    ax1.axvline(failure.mean(), color='#c0392b', linestyle='--', linewidth=2,
               label=f'Failure Mean: {failure.mean():.1f} mo')
    ax1.axvline(success.mean(), color='#27ae60', linestyle='--', linewidth=2,
               label=f'Success Mean: {success.mean():.1f} mo')
    
    # Shade typical Phase 1 range
    ax1.axvspan(6, 18, alpha=0.1, color='blue', label='Typical Phase 1 (6-18 mo)')
    
    ax1.set_xlabel('Trial Duration (months)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Trials', fontsize=11, fontweight='bold')
    ax1.set_title('Duration Distribution by Outcome', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Box plot
    ax2 = axes[1]
    
    bp = ax2.boxplot([failure, success], labels=['Did NOT\nProgress', 'Progressed\nto Phase 2'],
                     patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], ['#e74c3c', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('Trial Duration (months)', fontsize=11, fontweight='bold')
    ax2.set_title('Duration: Success vs Failure', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Insight: Trial duration shows different patterns by outcome\n→ Suggests creating trial_duration_months feature', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_duration_by_outcome.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/07_duration_by_outcome.png")
    plt.close()


def plot_sponsor_analysis(df, output_dir='figures'):
    """
    Plot 8: Sponsor analysis by outcome
    Shows: Certain sponsors have better track records → feature idea!
    """
    print("\n" + "="*80)
    print("PLOT 8: SPONSOR ANALYSIS BY OUTCOME")
    print("="*80)
    
    # Count trials per sponsor
    sponsor_counts = df['sponsor'].value_counts()
    df['sponsor_trial_count'] = df['sponsor'].map(sponsor_counts)
    
    # Identify top diabetes pharma
    top_pharma = [
        'novo nordisk', 'eli lilly', 'sanofi', 'astrazeneca', 
        'boehringer ingelheim', 'merck', 'pfizer'
    ]
    df['is_top_pharma'] = df['sponsor'].str.lower().apply(
        lambda x: int(any(company in str(x).lower() for company in top_pharma))
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Success rate by sponsor type
    ax1 = axes[0]
    
    sponsor_success = df.groupby('sponsor_type')['success'].agg(['mean', 'count'])
    sponsor_success = sponsor_success[sponsor_success['count'] >= 5]  # Min 5 trials
    sponsor_success = sponsor_success.sort_values('mean', ascending=False)
    
    bars = ax1.bar(range(len(sponsor_success)), sponsor_success['mean'] * 100,
                   color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (rate, count) in enumerate(zip(sponsor_success['mean'], sponsor_success['count'])):
        ax1.text(i, rate * 100 + 2, f'n={count}', ha='center', fontsize=10)
    
    ax1.set_xticks(range(len(sponsor_success)))
    ax1.set_xticklabels(sponsor_success.index, rotation=0)
    ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Success Rate by Sponsor Type', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Top pharma vs others
    ax2 = axes[1]
    
    pharma_success = df.groupby('is_top_pharma')['success'].agg(['mean', 'count'])
    pharma_success.index = ['Other Sponsors', 'Top Diabetes Pharma']
    
    bars = ax2.bar(range(len(pharma_success)), pharma_success['mean'] * 100,
                   color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (rate, count) in enumerate(zip(pharma_success['mean'], pharma_success['count'])):
        ax2.text(i, rate * 100 + 2, f'n={count}', ha='center', fontsize=10)
    
    ax2.set_xticks(range(len(pharma_success)))
    ax2.set_xticklabels(pharma_success.index)
    ax2.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Top Pharma vs Others', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Insight: Top pharma companies and sponsor type matter\n→ Suggests creating is_top_pharma and sponsor_trial_count features', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_sponsor_by_outcome.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/08_sponsor_by_outcome.png")
    plt.close()


def plot_trial_design_features(df, output_dir='figures'):
    """
    Plot 9: Trial design features
    Shows: Design quality matters → composite feature idea!
    """
    print("\n" + "="*80)
    print("PLOT 9: TRIAL DESIGN FEATURES")
    print("="*80)
    
    # Check for placebo
    df['has_placebo'] = df['intervention_names'].str.lower().str.contains('placebo', na=False).astype(int)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    design_features = [
        ('allocation', 'Allocation Method', ['UNKNOWN']),  # Exclude UNKNOWN
        ('masking', 'Masking Level', ['NONE']),  # Exclude NONE
        ('intervention_model', 'Study Design', ['UNKNOWN']),  # Exclude UNKNOWN
        ('has_placebo', 'Placebo Control', [])  # No exclusions
    ]
    
    for idx, (feature, title, exclude_values) in enumerate(design_features):
        ax = axes[idx]
        
        # Calculate success rate
        if feature == 'has_placebo':
            success_rate = df.groupby(feature)['success'].agg(['mean', 'count'])
            success_rate.index = ['No Placebo', 'Has Placebo']
        else:
            success_rate = df.groupby(feature)['success'].agg(['mean', 'count'])
            
            # Exclude specified values
            for exclude_val in exclude_values:
                if exclude_val in success_rate.index:
                    success_rate = success_rate.drop(exclude_val)
            
            # Only show categories with at least 5 trials
            success_rate = success_rate[success_rate['count'] >= 5]
        
        success_rate = success_rate.sort_values('mean', ascending=False)
        
        bars = ax.bar(range(len(success_rate)), success_rate['mean'] * 100,
                     color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Color by success rate
        for i, bar in enumerate(bars):
            if success_rate['mean'].iloc[i] > df['success'].mean():
                bar.set_color('#2ecc71')
            else:
                bar.set_color('#e74c3c')
        
        for i, (rate, count) in enumerate(zip(success_rate['mean'], success_rate['count'])):
            ax.text(i, rate * 100 + 2, f'n={count}', ha='center', fontsize=9)
        
        ax.set_xticks(range(len(success_rate)))
        ax.set_xticklabels(success_rate.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Success Rate (%)', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axhline(y=df['success'].mean() * 100, color='black', linestyle='--', 
                  linewidth=1, alpha=0.5, label='Overall Mean')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Insight: Multiple design features impact success\n→ Suggests creating composite design_quality_score feature', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_trial_design_features.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/09_trial_design_features.png")
    plt.close()

def plot_correlation_with_outcome(df, output_dir='figures'):
    """
    Plot 10: Simple correlations with outcome
    """
    print("\n" + "="*80)
    print("PLOT 10: FEATURE CORRELATIONS WITH OUTCOME")
    print("="*80)
    
    # Calculate duration if not done
    if 'duration_months' not in df.columns:
        df = calculate_trial_duration(df)
    
    # Count trials per sponsor
    sponsor_counts = df['sponsor'].value_counts()
    df['sponsor_trial_count'] = df['sponsor'].map(sponsor_counts)
    
    # Select numeric features
    numeric_features = [
        ('enrollment', 'Enrollment Size'),
        ('duration_months', 'Trial Duration (months)'),
        ('sponsor_trial_count', 'Sponsor Trial Count'),
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    correlations = []
    labels = []
    
    for feature, label in numeric_features:
        corr = df[[feature, 'success']].corr().iloc[0, 1]
        correlations.append(corr)
        labels.append(label)
    
    # Bar plot
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in correlations]
    bars = ax.barh(labels, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add values
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        ax.text(corr + 0.01 if corr > 0 else corr - 0.01, i, f'{corr:.3f}',
               va='center', ha='left' if corr > 0 else 'right', fontweight='bold')
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Correlation with Success', fontsize=12, fontweight='bold')
    ax.set_title('Feature Correlations with Trial Success', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.3, 0.3)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_correlations_with_outcome.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/10_correlations_with_outcome.png")
    plt.close()

def main():
    """
    Run complete EDA pipeline
    """
    import os
    
    # Create figures directory
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_raw_data()
    
    # Generate plots
    plot_target_distribution(df, output_dir)
    plot_top_sponsors_by_volume(df, output_dir)
    plot_top_sponsors_by_success(df, output_dir)
    plot_enrollment_variation(df, output_dir)
    plot_duration_variation(df, output_dir)
    plot_enrollment_distribution(df, output_dir)
    plot_trial_duration(df, output_dir)
    plot_sponsor_analysis(df, output_dir)
    plot_trial_design_features(df, output_dir)
    plot_correlation_with_outcome(df, output_dir)


if __name__ == '__main__':
    main()