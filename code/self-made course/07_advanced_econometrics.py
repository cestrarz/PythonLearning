# 09 - Advanced Econometric Methods
# Regression Discontinuity, Difference-in-Differences, Event Studies, and more

"""
ADVANCED ECONOMETRIC METHODS
============================

This module covers advanced causal inference methods:

1. Regression Discontinuity Design (RDD)
2. Difference-in-Differences (DiD)
3. Event Study Analysis
4. Synthetic Control Method
5. Matching Methods
6. Treatment Effect Heterogeneity

Required Libraries:
- statsmodels: Core econometrics
- scipy: Statistical functions
- sklearn: Machine learning for matching
- matplotlib/seaborn: Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Econometrics libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

# Advanced Econometric Methods

# =============================================================================
# 1. REGRESSION DISCONTINUITY DESIGN (RDD)
# =============================================================================

def generate_rdd_data(n=2000, cutoff=0, seed=42):
    """Generate data for RDD example"""
    np.random.seed(seed)
    
    # Running variable (e.g., test score, income, age)
    running_var = np.random.normal(0, 2, n)
    
    # Treatment assignment based on cutoff
    treatment = (running_var >= cutoff).astype(int)
    
    # Outcome with discontinuous jump at cutoff
    # True treatment effect is 2.0
    outcome = (
        1.0 +                           # intercept
        0.5 * running_var +             # smooth function of running variable
        2.0 * treatment +               # treatment effect (discontinuity)
        np.random.normal(0, 1, n)       # random error
    )
    
    return pd.DataFrame({
        'running_var': running_var,
        'treatment': treatment,
        'outcome': outcome
    })

# Generate RDD data
rdd_data = generate_rdd_data(n=2000, cutoff=0)
cutoff = 0

# RDD Data Summary: 2000 observations with running variable, treatment, and outcome
print(rdd_data.describe())
print(f"\nTreatment assignment:")
print(rdd_data['treatment'].value_counts())

# Visualize the discontinuity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Raw data scatter plot
colors = ['red' if t == 0 else 'blue' for t in rdd_data['treatment']]
ax1.scatter(rdd_data['running_var'], rdd_data['outcome'], 
           c=colors, alpha=0.5, s=20)
ax1.axvline(x=cutoff, color='black', linestyle='--', linewidth=2, label='Cutoff')
ax1.set_xlabel('Running Variable')
ax1.set_ylabel('Outcome')
ax1.set_title('RDD: Raw Data')
ax1.legend(['Control', 'Treatment', 'Cutoff'])

# Binned scatter plot for clearer visualization
n_bins = 40
rdd_data['bin'] = pd.cut(rdd_data['running_var'], bins=n_bins, labels=False)
binned_data = rdd_data.groupby('bin').agg({
    'running_var': 'mean',
    'outcome': 'mean',
    'treatment': 'mean'
}).reset_index()

colors_binned = ['red' if t < 0.5 else 'blue' for t in binned_data['treatment']]
ax2.scatter(binned_data['running_var'], binned_data['outcome'], 
           c=colors_binned, s=50, alpha=0.8)
ax2.axvline(x=cutoff, color='black', linestyle='--', linewidth=2)
ax2.set_xlabel('Running Variable')
ax2.set_ylabel('Outcome')
ax2.set_title('RDD: Binned Data')

plt.tight_layout()
plt.show()

# RDD Estimation
print("\nRDD Estimation:")
print("-"*30)

# Local linear regression around cutoff
bandwidth = 1.0  # Bandwidth for local regression
local_data = rdd_data[abs(rdd_data['running_var'] - cutoff) <= bandwidth].copy()

print(f"Using bandwidth: {bandwidth}")
print(f"Observations in bandwidth: {len(local_data)}")

# Center running variable at cutoff
local_data['running_centered'] = local_data['running_var'] - cutoff

# RDD regression: Y = α + β₁*D + β₂*X + β₃*D*X + ε
# where D is treatment, X is centered running variable
rdd_formula = 'outcome ~ treatment + running_centered + treatment:running_centered'
rdd_model = smf.ols(rdd_formula, data=local_data).fit()

print("RDD Regression Results:")
print(rdd_model.summary())

# Extract treatment effect
treatment_effect = rdd_model.params['treatment']
treatment_se = rdd_model.bse['treatment']
treatment_tstat = rdd_model.tvalues['treatment']
treatment_pvalue = rdd_model.pvalues['treatment']

print(f"\nRDD Treatment Effect:")
print(f"Estimate: {treatment_effect:.4f}")
print(f"Standard Error: {treatment_se:.4f}")
print(f"t-statistic: {treatment_tstat:.4f}")
print(f"p-value: {treatment_pvalue:.4f}")
print(f"95% CI: [{treatment_effect - 1.96*treatment_se:.4f}, {treatment_effect + 1.96*treatment_se:.4f}]")

# Bandwidth sensitivity analysis
bandwidths = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
bandwidth_results = []

for bw in bandwidths:
    local_data_bw = rdd_data[abs(rdd_data['running_var'] - cutoff) <= bw].copy()
    local_data_bw['running_centered'] = local_data_bw['running_var'] - cutoff
    
    if len(local_data_bw) > 10:  # Ensure sufficient observations
        model_bw = smf.ols(rdd_formula, data=local_data_bw).fit()
        bandwidth_results.append({
            'bandwidth': bw,
            'treatment_effect': model_bw.params['treatment'],
            'se': model_bw.bse['treatment'],
            'n_obs': len(local_data_bw)
        })

bandwidth_df = pd.DataFrame(bandwidth_results)
print(f"\nBandwidth Sensitivity Analysis:")
print(bandwidth_df.round(4))

# Visualize bandwidth sensitivity
plt.figure(figsize=(10, 6))
plt.errorbar(bandwidth_df['bandwidth'], bandwidth_df['treatment_effect'], 
             yerr=1.96*bandwidth_df['se'], marker='o', capsize=5)
plt.axhline(y=2.0, color='red', linestyle='--', label='True Effect')
plt.xlabel('Bandwidth')
plt.ylabel('Treatment Effect Estimate')
plt.title('RDD: Bandwidth Sensitivity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("✅ Regression Discontinuity Design completed")

# =============================================================================
# 2. DIFFERENCE-IN-DIFFERENCES (DiD)
# =============================================================================

print("\n" + "="*60)
print("2. DIFFERENCE-IN-DIFFERENCES (DiD)")
print("="*60)

def generate_did_data(n_units=100, n_periods=10, treatment_period=6, seed=42):
    """Generate data for DiD analysis"""
    np.random.seed(seed)
    
    # Half of units are treated
    n_treated = n_units // 2
    
    did_data = []
    
    for unit in range(n_units):
        # Unit fixed effect
        unit_fe = np.random.normal(0, 1)
        
        # Treatment group indicator
        treated_unit = 1 if unit < n_treated else 0
        
        for period in range(1, n_periods + 1):
            # Time fixed effect
            time_fe = 0.1 * period + np.random.normal(0, 0.2)
            
            # Treatment indicator (post-treatment for treated units)
            treatment = 1 if (treated_unit == 1 and period >= treatment_period) else 0
            
            # Outcome with parallel trends assumption
            # True treatment effect is 1.5
            outcome = (
                2.0 +                    # intercept
                unit_fe +                # unit fixed effect
                time_fe +                # time fixed effect
                1.5 * treatment +        # treatment effect
                np.random.normal(0, 0.3) # idiosyncratic error
            )
            
            did_data.append({
                'unit_id': unit + 1,
                'period': period,
                'treated_unit': treated_unit,
                'post_treatment': 1 if period >= treatment_period else 0,
                'treatment': treatment,
                'outcome': outcome,
                'unit_fe': unit_fe,
                'time_fe': time_fe
            })
    
    return pd.DataFrame(did_data)

# Generate DiD data
did_data = generate_did_data(n_units=100, n_periods=10, treatment_period=6)
treatment_period = 6

print("DiD Data Summary:")
print(did_data.head())
print(f"\nTreatment timing: Period {treatment_period}")
print(f"Treated units: {did_data['treated_unit'].sum()}")
print(f"Control units: {len(did_data) - did_data['treated_unit'].sum()}")

# Visualize parallel trends
pre_treatment_data = did_data[did_data['period'] < treatment_period]
post_treatment_data = did_data[did_data['period'] >= treatment_period]

# Calculate group means by period
group_means = did_data.groupby(['period', 'treated_unit'])['outcome'].mean().reset_index()
treated_means = group_means[group_means['treated_unit'] == 1]
control_means = group_means[group_means['treated_unit'] == 0]

plt.figure(figsize=(12, 8))
plt.plot(treated_means['period'], treated_means['outcome'], 
         marker='o', linewidth=2, label='Treated Group', color='blue')
plt.plot(control_means['period'], control_means['outcome'], 
         marker='s', linewidth=2, label='Control Group', color='red')
plt.axvline(x=treatment_period - 0.5, color='black', linestyle='--', 
           linewidth=2, label='Treatment Start')
plt.xlabel('Time Period')
plt.ylabel('Average Outcome')
plt.title('Difference-in-Differences: Parallel Trends')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# DiD Estimation
print("\nDiD Estimation:")
print("-"*30)

# Standard DiD regression: Y = α + β₁*Treated + β₂*Post + β₃*(Treated×Post) + ε
did_formula = 'outcome ~ treated_unit + post_treatment + treated_unit:post_treatment'
did_model = smf.ols(did_formula, data=did_data).fit()

print("Basic DiD Regression:")
print(did_model.summary())

# Two-way fixed effects DiD
did_formula_fe = 'outcome ~ treatment + C(unit_id) + C(period)'
did_model_fe = smf.ols(did_formula_fe, data=did_data).fit()

print(f"\nTwo-Way Fixed Effects DiD:")
treatment_effect_fe = did_model_fe.params['treatment']
treatment_se_fe = did_model_fe.bse['treatment']

print(f"Treatment Effect: {treatment_effect_fe:.4f}")
print(f"Standard Error: {treatment_se_fe:.4f}")
print(f"t-statistic: {did_model_fe.tvalues['treatment']:.4f}")
print(f"p-value: {did_model_fe.pvalues['treatment']:.4f}")

# Manual DiD calculation (2x2 table)
print(f"\nManual DiD Calculation:")
print("-"*30)

# Calculate group means
treated_pre = did_data[(did_data['treated_unit'] == 1) & (did_data['post_treatment'] == 0)]['outcome'].mean()
treated_post = did_data[(did_data['treated_unit'] == 1) & (did_data['post_treatment'] == 1)]['outcome'].mean()
control_pre = did_data[(did_data['treated_unit'] == 0) & (did_data['post_treatment'] == 0)]['outcome'].mean()
control_post = did_data[(did_data['treated_unit'] == 0) & (did_data['post_treatment'] == 1)]['outcome'].mean()

# DiD calculation
treated_diff = treated_post - treated_pre
control_diff = control_post - control_pre
did_effect = treated_diff - control_diff

print(f"Treated group change: {treated_diff:.4f}")
print(f"Control group change: {control_diff:.4f}")
print(f"DiD effect: {did_effect:.4f}")

# Create DiD table
did_table = pd.DataFrame({
    'Group': ['Treated', 'Control', 'Difference'],
    'Pre-treatment': [treated_pre, control_pre, treated_pre - control_pre],
    'Post-treatment': [treated_post, control_post, treated_post - control_post],
    'Difference': [treated_diff, control_diff, did_effect]
})

print(f"\nDiD Table:")
print(did_table.round(4))

print("✅ Difference-in-Differences completed")

# =============================================================================
# 3. EVENT STUDY ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("3. EVENT STUDY ANALYSIS")
print("="*60)

def generate_event_study_data(n_units=50, n_periods=20, event_period=11, seed=42):
    """Generate data for event study"""
    np.random.seed(seed)
    
    event_data = []
    
    for unit in range(n_units):
        # Unit fixed effect
        unit_fe = np.random.normal(0, 0.5)
        
        # Random event timing (some units never treated)
        if np.random.random() < 0.7:  # 70% get treated
            unit_event_period = event_period + np.random.randint(-2, 3)  # Vary timing slightly
        else:
            unit_event_period = None  # Never treated
        
        for period in range(1, n_periods + 1):
            # Time fixed effect
            time_fe = 0.05 * period
            
            # Event study effects (dynamic treatment effects)
            if unit_event_period is not None:
                periods_since_event = period - unit_event_period
                
                if periods_since_event < -5:
                    event_effect = 0  # No anticipation effects beyond 5 periods
                elif periods_since_event == -2:
                    event_effect = 0.2  # Small anticipation effect
                elif periods_since_event == -1:
                    event_effect = 0.4  # Larger anticipation effect
                elif periods_since_event == 0:
                    event_effect = 1.0  # Immediate effect
                elif periods_since_event == 1:
                    event_effect = 1.5  # Peak effect
                elif periods_since_event == 2:
                    event_effect = 1.3  # Slight decline
                elif periods_since_event >= 3:
                    event_effect = 1.0  # Long-term effect
                else:
                    event_effect = 0
            else:
                periods_since_event = None
                event_effect = 0
            
            outcome = (
                2.0 +                    # intercept
                unit_fe +                # unit fixed effect
                time_fe +                # time fixed effect
                event_effect +           # dynamic treatment effect
                np.random.normal(0, 0.2) # error
            )
            
            event_data.append({
                'unit_id': unit + 1,
                'period': period,
                'event_period': unit_event_period,
                'periods_since_event': periods_since_event,
                'treated': 1 if unit_event_period is not None else 0,
                'outcome': outcome,
                'event_effect': event_effect
            })
    
    return pd.DataFrame(event_data)

# Generate event study data
event_data = generate_event_study_data(n_units=50, n_periods=20, event_period=11)

print("Event Study Data Summary:")
print(event_data.head())
print(f"\nTreated units: {event_data['treated'].sum() // 20}")  # Divide by periods
print(f"Control units: {(len(event_data) - event_data['treated'].sum()) // 20}")

# Create event time indicators
event_data_clean = event_data[event_data['periods_since_event'].notna()].copy()

# Create event time dummies (relative to event)
event_times = range(-5, 6)  # -5 to +5 periods around event
for t in event_times:
    if t != -1:  # Omit t=-1 as reference period
        event_data_clean[f'event_time_{t}'] = (event_data_clean['periods_since_event'] == t).astype(int)

# Event study regression
event_formula = 'outcome ~ ' + ' + '.join([f'event_time_{t}' for t in event_times if t != -1]) + ' + C(unit_id) + C(period)'
event_model = smf.ols(event_formula, data=event_data_clean).fit()

print(f"\nEvent Study Regression Results:")
print("-"*40)

# Extract event study coefficients
event_coeffs = []
event_ses = []
event_times_plot = []

for t in event_times:
    if t == -1:
        # Reference period (normalized to 0)
        event_coeffs.append(0)
        event_ses.append(0)
        event_times_plot.append(t)
    else:
        var_name = f'event_time_{t}'
        if var_name in event_model.params:
            event_coeffs.append(event_model.params[var_name])
            event_ses.append(event_model.bse[var_name])
            event_times_plot.append(t)
            print(f"Period {t:2d}: {event_model.params[var_name]:6.3f} ({event_model.bse[var_name]:.3f})")

# Plot event study results
plt.figure(figsize=(12, 8))
plt.errorbar(event_times_plot, event_coeffs, yerr=[1.96*se for se in event_ses], 
             marker='o', capsize=5, linewidth=2, markersize=6)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.axvline(x=-0.5, color='black', linestyle='--', alpha=0.7, label='Event Time')
plt.xlabel('Periods Relative to Event')
plt.ylabel('Treatment Effect')
plt.title('Event Study: Dynamic Treatment Effects')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Test for pre-trends
pre_trend_vars = [f'event_time_{t}' for t in range(-5, -1) if f'event_time_{t}' in event_model.params]
if pre_trend_vars:
    # F-test for joint significance of pre-treatment coefficients
    pre_trend_test = event_model.f_test(' = '.join(pre_trend_vars) + ' = 0')
    print(f"\nPre-trend Test:")
    print(f"F-statistic: {pre_trend_test.fvalue:.4f}")
    print(f"p-value: {pre_trend_test.pvalue:.4f}")
    print(f"Conclusion: {'Reject' if pre_trend_test.pvalue < 0.05 else 'Fail to reject'} null of no pre-trends")

print("✅ Event Study Analysis completed")

# =============================================================================
# 4. SYNTHETIC CONTROL METHOD
# =============================================================================

print("\n" + "="*60)
print("4. SYNTHETIC CONTROL METHOD")
print("="*60)

def generate_synthetic_control_data(n_units=20, n_periods=30, treatment_period=21, seed=42):
    """Generate data for synthetic control analysis"""
    np.random.seed(seed)
    
    # One treated unit, rest are potential controls
    treated_unit = 1
    
    sc_data = []
    
    for unit in range(1, n_units + 1):
        # Unit-specific trend
        unit_trend = np.random.normal(0.02, 0.01)
        
        # Unit fixed effect
        unit_fe = np.random.normal(5, 1)
        
        for period in range(1, n_periods + 1):
            # Common time trend
            time_trend = 0.03 * period
            
            # Treatment effect (only for treated unit after treatment)
            if unit == treated_unit and period >= treatment_period:
                treatment_effect = 2.0 + 0.1 * (period - treatment_period)  # Growing effect
            else:
                treatment_effect = 0
            
            # Outcome
            outcome = (
                unit_fe +
                time_trend +
                unit_trend * period +
                treatment_effect +
                np.random.normal(0, 0.3)
            )
            
            sc_data.append({
                'unit_id': unit,
                'period': period,
                'treated': 1 if unit == treated_unit else 0,
                'post_treatment': 1 if period >= treatment_period else 0,
                'treatment': 1 if (unit == treated_unit and period >= treatment_period) else 0,
                'outcome': outcome
            })
    
    return pd.DataFrame(sc_data)

# Generate synthetic control data
sc_data = generate_synthetic_control_data(n_units=20, n_periods=30, treatment_period=21)
treatment_period = 21

print("Synthetic Control Data Summary:")
print(sc_data.head())

# Prepare data for synthetic control
treated_data = sc_data[sc_data['treated'] == 1]['outcome'].values
control_data = sc_data[sc_data['treated'] == 0].pivot(index='period', columns='unit_id', values='outcome')

print(f"\nTreated unit outcome shape: {treated_data.shape}")
print(f"Control units data shape: {control_data.shape}")

# Pre-treatment period for fitting
pre_treatment_periods = range(1, treatment_period)
post_treatment_periods = range(treatment_period, 31)

# Synthetic control optimization
def synthetic_control_objective(weights, treated_pre, control_pre):
    """Objective function for synthetic control"""
    synthetic_pre = control_pre @ weights
    return np.sum((treated_pre - synthetic_pre) ** 2)

# Constraints: weights sum to 1 and are non-negative
from scipy.optimize import minimize

n_controls = control_data.shape[1]
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1) for _ in range(n_controls)]
initial_weights = np.ones(n_controls) / n_controls

# Get pre-treatment data
treated_pre = treated_data[:treatment_period-1]
control_pre = control_data.iloc[:treatment_period-1].values

# Optimize weights
result = minimize(
    synthetic_control_objective,
    initial_weights,
    args=(treated_pre, control_pre),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
print(f"\nSynthetic Control Optimization:")
print(f"Optimization successful: {result.success}")
print(f"Objective value: {result.fun:.6f}")

# Show non-zero weights
significant_weights = [(i+2, w) for i, w in enumerate(optimal_weights) if w > 0.01]  # unit_id starts from 2
print(f"\nSignificant weights (>0.01):")
for unit_id, weight in significant_weights:
    print(f"Unit {unit_id}: {weight:.4f}")

# Create synthetic control
synthetic_outcome = control_data.values @ optimal_weights

# Calculate treatment effects
treatment_effects = treated_data - synthetic_outcome
pre_treatment_effects = treatment_effects[:treatment_period-1]
post_treatment_effects = treatment_effects[treatment_period-1:]

print(f"\nPre-treatment fit:")
print(f"Mean absolute error: {np.mean(np.abs(pre_treatment_effects)):.4f}")
print(f"Root mean squared error: {np.sqrt(np.mean(pre_treatment_effects**2)):.4f}")

print(f"\nPost-treatment effects:")
print(f"Average treatment effect: {np.mean(post_treatment_effects):.4f}")
print(f"Cumulative treatment effect: {np.sum(post_treatment_effects):.4f}")

# Visualize synthetic control
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Outcomes over time
periods = range(1, 31)
ax1.plot(periods, treated_data, 'b-', linewidth=2, label='Treated Unit')
ax1.plot(periods, synthetic_outcome, 'r--', linewidth=2, label='Synthetic Control')
ax1.axvline(x=treatment_period-0.5, color='black', linestyle=':', alpha=0.7, label='Treatment Start')
ax1.set_xlabel('Period')
ax1.set_ylabel('Outcome')
ax1.set_title('Synthetic Control: Treated vs Synthetic')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Treatment effects
ax2.plot(periods, treatment_effects, 'g-', linewidth=2, marker='o', markersize=4)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.axvline(x=treatment_period-0.5, color='black', linestyle=':', alpha=0.7)
ax2.set_xlabel('Period')
ax2.set_ylabel('Treatment Effect')
ax2.set_title('Treatment Effects (Treated - Synthetic)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Synthetic Control Method completed")

# =============================================================================
# 5. PROPENSITY SCORE MATCHING
# =============================================================================

print("\n" + "="*60)
print("5. PROPENSITY SCORE MATCHING")
print("="*60)

def generate_matching_data(n=1000, seed=42):
    """Generate data for propensity score matching"""
    np.random.seed(seed)
    
    # Covariates
    x1 = np.random.normal(0, 1, n)  # Continuous covariate
    x2 = np.random.binomial(1, 0.5, n)  # Binary covariate
    x3 = np.random.normal(2, 0.5, n)  # Another continuous covariate
    
    # Propensity score (probability of treatment)
    propensity_logit = -0.5 + 0.8*x1 + 0.5*x2 + 0.3*x3
    propensity_score = 1 / (1 + np.exp(-propensity_logit))
    
    # Treatment assignment based on propensity score
    treatment = np.random.binomial(1, propensity_score, n)
    
    # Outcome with selection bias
    # True treatment effect is 1.0
    outcome = (
        2.0 +                    # intercept
        1.0 * treatment +        # treatment effect
        0.5 * x1 +              # covariate effects
        0.3 * x2 +
        0.4 * x3 +
        np.random.normal(0, 0.5, n)  # error
    )
    
    return pd.DataFrame({
        'treatment': treatment,
        'outcome': outcome,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'propensity_score': propensity_score
    })

# Generate matching data
matching_data = generate_matching_data(n=1000)

print("Matching Data Summary:")
print(matching_data.describe())
print(f"\nTreatment distribution:")
print(matching_data['treatment'].value_counts())

# Estimate propensity scores
ps_formula = 'treatment ~ x1 + x2 + x3'
ps_model = smf.logit(ps_formula, data=matching_data).fit()

print(f"\nPropensity Score Model:")
print(ps_model.summary())

# Predicted propensity scores
matching_data['ps_predicted'] = ps_model.predict()

# Check overlap (common support)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Propensity score distributions
treated_ps = matching_data[matching_data['treatment'] == 1]['ps_predicted']
control_ps = matching_data[matching_data['treatment'] == 0]['ps_predicted']

ax1.hist(treated_ps, bins=30, alpha=0.7, label='Treated', density=True)
ax1.hist(control_ps, bins=30, alpha=0.7, label='Control', density=True)
ax1.set_xlabel('Propensity Score')
ax1.set_ylabel('Density')
ax1.set_title('Propensity Score Distributions')
ax1.legend()

# Covariate balance before matching
treated_x1 = matching_data[matching_data['treatment'] == 1]['x1'].mean()
control_x1 = matching_data[matching_data['treatment'] == 0]['x1'].mean()
treated_x2 = matching_data[matching_data['treatment'] == 1]['x2'].mean()
control_x2 = matching_data[matching_data['treatment'] == 0]['x2'].mean()

balance_before = pd.DataFrame({
    'Variable': ['x1', 'x2'],
    'Treated_Mean': [treated_x1, treated_x2],
    'Control_Mean': [control_x1, control_x2],
    'Difference': [treated_x1 - control_x1, treated_x2 - control_x2]
})

print(f"\nCovariate Balance Before Matching:")
print(balance_before.round(4))

# Nearest neighbor matching
treated_data = matching_data[matching_data['treatment'] == 1].copy()
control_data = matching_data[matching_data['treatment'] == 0].copy()

# Use sklearn for nearest neighbor matching
nn_matcher = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn_matcher.fit(control_data[['ps_predicted']])

# Find matches for treated units
distances, indices = nn_matcher.kneighbors(treated_data[['ps_predicted']])

# Create matched dataset
matched_control_indices = control_data.iloc[indices.flatten()].index
matched_treated = treated_data.copy()
matched_control = control_data.loc[matched_control_indices].copy()

matched_data = pd.concat([matched_treated, matched_control], ignore_index=True)

print(f"\nMatching Results:")
print(f"Original sample size: {len(matching_data)}")
print(f"Matched sample size: {len(matched_data)}")
print(f"Treated units matched: {len(matched_treated)}")

# Check balance after matching
treated_x1_matched = matched_data[matched_data['treatment'] == 1]['x1'].mean()
control_x1_matched = matched_data[matched_data['treatment'] == 0]['x1'].mean()
treated_x2_matched = matched_data[matched_data['treatment'] == 1]['x2'].mean()
control_x2_matched = matched_data[matched_data['treatment'] == 0]['x2'].mean()

balance_after = pd.DataFrame({
    'Variable': ['x1', 'x2'],
    'Treated_Mean': [treated_x1_matched, treated_x2_matched],
    'Control_Mean': [control_x1_matched, control_x2_matched],
    'Difference': [treated_x1_matched - control_x1_matched, treated_x2_matched - control_x2_matched]
})

print(f"\nCovariate Balance After Matching:")
print(balance_after.round(4))

# Treatment effect estimation
print(f"\nTreatment Effect Estimation:")
print("-"*40)

# Naive comparison (biased)
naive_ate = matching_data[matching_data['treatment'] == 1]['outcome'].mean() - \
           matching_data[matching_data['treatment'] == 0]['outcome'].mean()

# Matched comparison
matched_ate = matched_data[matched_data['treatment'] == 1]['outcome'].mean() - \
             matched_data[matched_data['treatment'] == 0]['outcome'].mean()

# Regression adjustment on matched sample
matched_reg = smf.ols('outcome ~ treatment + x1 + x2 + x3', data=matched_data).fit()

print(f"True treatment effect: 1.000")
print(f"Naive ATE: {naive_ate:.4f}")
print(f"Matched ATE: {matched_ate:.4f}")
print(f"Regression on matched sample: {matched_reg.params['treatment']:.4f}")

# Visualize matching quality
ax2.scatter(matched_treated['ps_predicted'], matched_control['ps_predicted'], alpha=0.6)
ax2.plot([0, 1], [0, 1], 'r--', alpha=0.7)
ax2.set_xlabel('Treated Unit Propensity Score')
ax2.set_ylabel('Matched Control Unit Propensity Score')
ax2.set_title('Matching Quality')

plt.tight_layout()
plt.show()

print("✅ Propensity Score Matching completed")

print("\n" + "="*70)
print("ADVANCED ECONOMETRICS SUMMARY")
print("="*70)
print("✅ Regression Discontinuity Design (RDD)")
print("✅ Difference-in-Differences (DiD)")
print("✅ Event Study Analysis")
print("✅ Synthetic Control Method")
print("✅ Propensity Score Matching")
print("\nThese methods provide robust causal inference tools!")
print("Next: Learn how to create publication-ready tables!")
print("="*70)