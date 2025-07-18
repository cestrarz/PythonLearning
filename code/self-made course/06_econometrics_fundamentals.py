# 08 - Econometrics Fundamentals
# Core econometric methods for causal inference and policy analysis

"""
ECONOMETRICS FUNDAMENTALS
========================

Key Topics:
1. Ordinary Least Squares (OLS)
2. Robust Standard Errors
3. Clustered Standard Errors
4. Fixed Effects Models
5. Random Effects Models
6. Instrumental Variables (IV)
7. Model Diagnostics and Testing

Required Libraries:
- statsmodels: Main econometrics library
- linearmodels: Advanced panel data models
- stargazer: LaTeX table output
- scipy: Statistical tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Core econometrics libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Advanced econometrics
try:
    from linearmodels import PanelOLS, RandomEffects, IV2SLS
    from linearmodels.panel import compare
    LINEARMODELS_AVAILABLE = True
except ImportError:
    print("⚠️  linearmodels not available. Install with: pip install linearmodels")
    LINEARMODELS_AVAILABLE = False

# LaTeX table generation
try:
    from stargazer.stargazer import Stargazer
    STARGAZER_AVAILABLE = True
except ImportError:
    print("⚠️  stargazer not available. Install with: pip install stargazer")
    STARGAZER_AVAILABLE = False

print("Econometrics Fundamentals Module")
print("="*50)

# =============================================================================
# 1. DATA GENERATION FOR ECONOMETRIC EXAMPLES
# =============================================================================

def generate_econometric_data(n=1000, seed=42):
    """Generate realistic econometric dataset"""
    np.random.seed(seed)
    
    # Individual characteristics
    data = pd.DataFrame({
        'id': range(1, n+1),
        'age': np.random.normal(35, 10, n),
        'education': np.random.normal(14, 3, n),  # years of education
        'experience': np.random.normal(10, 8, n),  # years of experience
        'female': np.random.binomial(1, 0.5, n),
        'married': np.random.binomial(1, 0.6, n),
        'urban': np.random.binomial(1, 0.7, n),
    })
    
    # Ensure realistic bounds
    data['age'] = np.clip(data['age'], 18, 65)
    data['education'] = np.clip(data['education'], 8, 20)
    data['experience'] = np.clip(data['experience'], 0, data['age'] - data['education'] - 6)
    
    # Generate correlated error terms
    epsilon = np.random.normal(0, 0.3, n)
    
    # True wage equation with realistic coefficients
    data['log_wage'] = (
        1.5 +                           # intercept
        0.08 * data['education'] +      # returns to education
        0.03 * data['experience'] +     # returns to experience
        -0.0005 * data['experience']**2 + # diminishing returns
        -0.15 * data['female'] +        # gender wage gap
        0.10 * data['married'] +        # marriage premium
        0.12 * data['urban'] +          # urban premium
        epsilon                         # random error
    )
    
    # Convert to wage levels
    data['wage'] = np.exp(data['log_wage'])
    
    # Add some additional variables
    data['age_squared'] = data['age']**2
    data['experience_squared'] = data['experience']**2
    
    return data

def generate_panel_data(n_individuals=200, n_periods=5, seed=42):
    """Generate panel dataset for fixed effects examples"""
    np.random.seed(seed)
    
    # Individual fixed effects (unobserved heterogeneity)
    individual_effects = np.random.normal(0, 0.5, n_individuals)
    
    # Time fixed effects
    time_effects = np.random.normal(0, 0.2, n_periods)
    
    panel_data = []
    
    for i in range(n_individuals):
        for t in range(n_periods):
            # Time-varying characteristics
            x1 = np.random.normal(2, 1)  # Some policy variable
            x2 = np.random.normal(1, 0.5)  # Control variable
            
            # Outcome with individual and time fixed effects
            y = (
                2.0 +                    # intercept
                1.5 * x1 +              # treatment effect
                0.8 * x2 +              # control effect
                individual_effects[i] +  # individual FE
                time_effects[t] +        # time FE
                np.random.normal(0, 0.3) # idiosyncratic error
            )
            
            panel_data.append({
                'individual_id': i + 1,
                'time_period': t + 1,
                'y': y,
                'x1': x1,
                'x2': x2,
                'individual_fe': individual_effects[i],
                'time_fe': time_effects[t]
            })
    
    return pd.DataFrame(panel_data)

print("✅ Data generation functions created")

# =============================================================================
# 2. ORDINARY LEAST SQUARES (OLS) REGRESSION
# =============================================================================

print("\n" + "="*60)
print("2. ORDINARY LEAST SQUARES (OLS) REGRESSION")
print("="*60)

# Generate data
wage_data = generate_econometric_data(n=1000)

print("Wage Data Summary:")
print(wage_data.describe())

# Basic OLS regression
print("\n" + "-"*40)
print("BASIC OLS REGRESSION")
print("-"*40)

# Simple regression: log wage on education
X_simple = sm.add_constant(wage_data['education'])
y_simple = wage_data['log_wage']

model_simple = sm.OLS(y_simple, X_simple).fit()
print("Simple Regression: log(wage) = β₀ + β₁*education + ε")
print(model_simple.summary())

# Multiple regression
print("\n" + "-"*40)
print("MULTIPLE REGRESSION")
print("-"*40)

# Full wage equation
formula = 'log_wage ~ education + experience + I(experience**2) + female + married + urban'
model_full = smf.ols(formula, data=wage_data).fit()

print("Full Wage Equation:")
print(model_full.summary())

# Extract and interpret coefficients
print("\n" + "-"*40)
print("COEFFICIENT INTERPRETATION")
print("-"*40)

coeffs = model_full.params
print(f"Returns to Education: {coeffs['education']:.4f}")
print(f"  → One additional year of education increases wages by {coeffs['education']*100:.2f}%")

print(f"Gender Wage Gap: {coeffs['female']:.4f}")
print(f"  → Women earn {abs(coeffs['female'])*100:.2f}% less than men, ceteris paribus")

print(f"Marriage Premium: {coeffs['married']:.4f}")
print(f"  → Married individuals earn {coeffs['married']*100:.2f}% more, ceteris paribus")

print(f"Urban Premium: {coeffs['urban']:.4f}")
print(f"  → Urban workers earn {coeffs['urban']*100:.2f}% more, ceteris paribus")

# Model fit statistics
print(f"\nModel Fit:")
print(f"R-squared: {model_full.rsquared:.4f}")
print(f"Adjusted R-squared: {model_full.rsquared_adj:.4f}")
print(f"F-statistic: {model_full.fvalue:.2f}")
print(f"F-statistic p-value: {model_full.f_pvalue:.2e}")

print("✅ OLS regression analysis completed")

# =============================================================================
# 3. REGRESSION DIAGNOSTICS
# =============================================================================

print("\n" + "="*60)
print("3. REGRESSION DIAGNOSTICS")
print("="*60)

# Residual analysis
residuals = model_full.resid
fitted_values = model_full.fittedvalues

# Create diagnostic plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Regression Diagnostics', fontsize=16, fontweight='bold')

# 1. Residuals vs Fitted
axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# 2. Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (Normality)')

# 3. Scale-Location plot
standardized_residuals = residuals / np.std(residuals)
axes[0, 2].scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
axes[0, 2].set_xlabel('Fitted Values')
axes[0, 2].set_ylabel('√|Standardized Residuals|')
axes[0, 2].set_title('Scale-Location Plot')

# 4. Histogram of residuals
axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Distribution of Residuals')

# Add normal curve
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
axes[1, 0].plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal')
axes[1, 0].legend()

# 5. Leverage plot
influence = model_full.get_influence()
leverage = influence.hat_matrix_diag
axes[1, 1].scatter(range(len(leverage)), leverage, alpha=0.6)
axes[1, 1].axhline(y=2*len(model_full.params)/len(wage_data), color='red', linestyle='--', 
                   label='2p/n threshold')
axes[1, 1].set_xlabel('Observation')
axes[1, 1].set_ylabel('Leverage')
axes[1, 1].set_title('Leverage Plot')
axes[1, 1].legend()

# 6. Cook's Distance
cooks_d = influence.cooks_distance[0]
axes[1, 2].scatter(range(len(cooks_d)), cooks_d, alpha=0.6)
axes[1, 2].axhline(y=4/len(wage_data), color='red', linestyle='--', 
                   label="Cook's D = 4/n")
axes[1, 2].set_xlabel('Observation')
axes[1, 2].set_ylabel("Cook's Distance")
axes[1, 2].set_title("Cook's Distance")
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# Statistical tests for assumptions
print("\n" + "-"*40)
print("STATISTICAL TESTS")
print("-"*40)

# Test for heteroscedasticity
bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, model_full.model.exog)
white_stat, white_pvalue, _, _ = het_white(residuals, model_full.model.exog)

print(f"Breusch-Pagan Test for Heteroscedasticity:")
print(f"  Statistic: {bp_stat:.4f}, p-value: {bp_pvalue:.4f}")
print(f"  {'Reject' if bp_pvalue < 0.05 else 'Fail to reject'} null of homoscedasticity")

print(f"\nWhite Test for Heteroscedasticity:")
print(f"  Statistic: {white_stat:.4f}, p-value: {white_pvalue:.4f}")
print(f"  {'Reject' if white_pvalue < 0.05 else 'Fail to reject'} null of homoscedasticity")

# Durbin-Watson test for autocorrelation
dw_stat = durbin_watson(residuals)
print(f"\nDurbin-Watson Test for Autocorrelation:")
print(f"  Statistic: {dw_stat:.4f}")
print(f"  Interpretation: {'No autocorrelation' if 1.5 < dw_stat < 2.5 else 'Possible autocorrelation'}")

# Jarque-Bera test for normality
jb_stat, jb_pvalue = stats.jarque_bera(residuals)
print(f"\nJarque-Bera Test for Normality:")
print(f"  Statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}")
print(f"  {'Reject' if jb_pvalue < 0.05 else 'Fail to reject'} null of normality")

# Multicollinearity check (VIF)
print(f"\nVariance Inflation Factors (VIF):")
X_vif = wage_data[['education', 'experience', 'female', 'married', 'urban']].copy()
X_vif = sm.add_constant(X_vif)

vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data)
print("Note: VIF > 10 indicates high multicollinearity")

print("✅ Regression diagnostics completed")

# =============================================================================
# 4. ROBUST STANDARD ERRORS
# =============================================================================

print("\n" + "="*60)
print("4. ROBUST STANDARD ERRORS")
print("="*60)

# Compare different standard error types
print("Comparison of Standard Error Types:")
print("-"*50)

# Standard OLS (homoscedastic assumption)
model_ols = smf.ols(formula, data=wage_data).fit()

# Heteroscedasticity-robust (White) standard errors
model_hc1 = smf.ols(formula, data=wage_data).fit(cov_type='HC1')
model_hc3 = smf.ols(formula, data=wage_data).fit(cov_type='HC3')

# Create comparison table
comparison_data = []
variables = ['education', 'experience', 'female', 'married', 'urban']

for var in variables:
    comparison_data.append({
        'Variable': var,
        'Coefficient': model_ols.params[var],
        'SE_OLS': model_ols.bse[var],
        'SE_HC1': model_hc1.bse[var],
        'SE_HC3': model_hc3.bse[var],
        't_OLS': model_ols.tvalues[var],
        't_HC1': model_hc1.tvalues[var],
        't_HC3': model_hc3.tvalues[var]
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.round(4))

# Show the impact on significance
print(f"\nSignificance at 5% level:")
print("-"*30)
for var in variables:
    ols_sig = abs(model_ols.tvalues[var]) > 1.96
    hc1_sig = abs(model_hc1.tvalues[var]) > 1.96
    hc3_sig = abs(model_hc3.tvalues[var]) > 1.96
    
    print(f"{var:12}: OLS={ols_sig}, HC1={hc1_sig}, HC3={hc3_sig}")

# When to use robust standard errors
print(f"\nWhen to use Robust Standard Errors:")
print("="*40)
guidance = [
    "✅ Always use when you suspect heteroscedasticity",
    "✅ Default choice in many applied economics papers",
    "✅ HC1: Good general purpose robust SE",
    "✅ HC3: Better for small samples",
    "✅ Cost: Slightly less efficient if homoscedasticity holds",
    "✅ Benefit: Valid inference even with heteroscedasticity"
]

for item in guidance:
    print(item)

print("✅ Robust standard errors analysis completed")

# =============================================================================
# 5. CLUSTERED STANDARD ERRORS
# =============================================================================

print("\n" + "="*60)
print("5. CLUSTERED STANDARD ERRORS")
print("="*60)

# Generate data with clustering structure
np.random.seed(42)
n_clusters = 50
n_per_cluster = 20
n_total = n_clusters * n_per_cluster

# Create clustered data
clustered_data = []
for cluster_id in range(1, n_clusters + 1):
    # Cluster-specific effect
    cluster_effect = np.random.normal(0, 0.5)
    
    for individual in range(n_per_cluster):
        x = np.random.normal(2, 1)
        # Outcome with cluster correlation
        y = 1 + 0.5 * x + cluster_effect + np.random.normal(0, 0.3)
        
        clustered_data.append({
            'cluster_id': cluster_id,
            'individual_id': individual + 1,
            'y': y,
            'x': x,
            'cluster_effect': cluster_effect
        })

clustered_df = pd.DataFrame(clustered_data)

print(f"Clustered Data Structure:")
print(f"Number of clusters: {n_clusters}")
print(f"Observations per cluster: {n_per_cluster}")
print(f"Total observations: {n_total}")

# Compare standard errors
formula_cluster = 'y ~ x'

# Standard OLS
model_standard = smf.ols(formula_cluster, data=clustered_df).fit()

# Clustered standard errors
model_clustered = smf.ols(formula_cluster, data=clustered_df).fit(
    cov_type='cluster', cov_kwds={'groups': clustered_df['cluster_id']}
)

print(f"\nComparison of Standard Errors:")
print("-"*40)
print(f"{'Variable':<12} {'Coef':<8} {'SE_OLS':<8} {'SE_Cluster':<10} {'t_OLS':<8} {'t_Cluster':<10}")
print("-"*60)

for var in ['Intercept', 'x']:
    coef = model_standard.params[var]
    se_ols = model_standard.bse[var]
    se_cluster = model_clustered.bse[var]
    t_ols = model_standard.tvalues[var]
    t_cluster = model_clustered.tvalues[var]
    
    print(f"{var:<12} {coef:<8.4f} {se_ols:<8.4f} {se_cluster:<10.4f} {t_ols:<8.2f} {t_cluster:<10.2f}")

# Show the impact of clustering
se_ratio = model_clustered.bse['x'] / model_standard.bse['x']
print(f"\nClustered SE is {se_ratio:.2f}x larger than OLS SE")
print(f"This suggests {'significant' if se_ratio > 1.5 else 'moderate'} within-cluster correlation")

# Visualize cluster effects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot by cluster
for cluster in range(1, min(11, n_clusters + 1)):  # Show first 10 clusters
    cluster_data = clustered_df[clustered_df['cluster_id'] == cluster]
    ax1.scatter(cluster_data['x'], cluster_data['y'], alpha=0.6, label=f'Cluster {cluster}')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Data by Cluster (First 10 Clusters)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Cluster effects distribution
cluster_effects = clustered_df.groupby('cluster_id')['cluster_effect'].first()
ax2.hist(cluster_effects, bins=15, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Cluster Effect')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Cluster Effects')

plt.tight_layout()
plt.show()

print("✅ Clustered standard errors analysis completed")

# =============================================================================
# 6. FIXED EFFECTS MODELS
# =============================================================================

print("\n" + "="*60)
print("6. FIXED EFFECTS MODELS")
print("="*60)

# Generate panel data
panel_df = generate_panel_data(n_individuals=100, n_periods=5)

print("Panel Data Structure:")
print(panel_df.head(10))
print(f"\nPanel dimensions: {panel_df['individual_id'].nunique()} individuals, "
      f"{panel_df['time_period'].nunique()} time periods")

# Pooled OLS (ignoring panel structure)
pooled_model = smf.ols('y ~ x1 + x2', data=panel_df).fit()

# Fixed Effects using statsmodels
# Create dummy variables for individuals (LSDV approach)
individual_dummies = pd.get_dummies(panel_df['individual_id'], prefix='ind')
time_dummies = pd.get_dummies(panel_df['time_period'], prefix='time')

# Individual Fixed Effects
panel_with_dummies = pd.concat([panel_df, individual_dummies.iloc[:, 1:]], axis=1)  # Drop first dummy
fe_individual_formula = 'y ~ x1 + x2 + ' + ' + '.join([f'ind_{i}' for i in range(2, panel_df['individual_id'].nunique() + 1)])
fe_individual_model = smf.ols(fe_individual_formula, data=panel_with_dummies).fit()

# Two-way Fixed Effects (individual + time)
panel_with_all_dummies = pd.concat([panel_with_dummies, time_dummies.iloc[:, 1:]], axis=1)
fe_twoway_formula = fe_individual_formula + ' + ' + ' + '.join([f'time_{i}' for i in range(2, panel_df['time_period'].nunique() + 1)])
fe_twoway_model = smf.ols(fe_twoway_formula, data=panel_with_all_dummies).fit()

# Compare models
print("\nModel Comparison:")
print("-"*50)
models_comparison = pd.DataFrame({
    'Model': ['Pooled OLS', 'Individual FE', 'Two-way FE'],
    'x1_coeff': [pooled_model.params['x1'], fe_individual_model.params['x1'], fe_twoway_model.params['x1']],
    'x1_se': [pooled_model.bse['x1'], fe_individual_model.bse['x1'], fe_twoway_model.bse['x1']],
    'x2_coeff': [pooled_model.params['x2'], fe_individual_model.params['x2'], fe_twoway_model.params['x2']],
    'x2_se': [pooled_model.bse['x2'], fe_individual_model.bse['x2'], fe_twoway_model.bse['x2']],
    'R_squared': [pooled_model.rsquared, fe_individual_model.rsquared, fe_twoway_model.rsquared]
})

print(models_comparison.round(4))

# True vs estimated effects
print(f"\nTrue vs Estimated Treatment Effects:")
print(f"True x1 coefficient: 1.500")
print(f"Pooled OLS estimate: {pooled_model.params['x1']:.4f}")
print(f"Individual FE estimate: {fe_individual_model.params['x1']:.4f}")
print(f"Two-way FE estimate: {fe_twoway_model.params['x1']:.4f}")

# Advanced Fixed Effects with linearmodels (if available)
if LINEARMODELS_AVAILABLE:
    print(f"\nAdvanced Fixed Effects (using linearmodels):")
    print("-"*50)
    
    # Set up panel data
    panel_df_indexed = panel_df.set_index(['individual_id', 'time_period'])
    
    # Individual Fixed Effects
    fe_model = PanelOLS(panel_df_indexed['y'], 
                       panel_df_indexed[['x1', 'x2']], 
                       entity_effects=True).fit()
    
    # Two-way Fixed Effects
    fe_twoway_model_lm = PanelOLS(panel_df_indexed['y'], 
                                 panel_df_indexed[['x1', 'x2']], 
                                 entity_effects=True, 
                                 time_effects=True).fit()
    
    print("Individual Fixed Effects (linearmodels):")
    print(fe_model.summary.tables[1])
    
    print("\nTwo-way Fixed Effects (linearmodels):")
    print(fe_twoway_model_lm.summary.tables[1])

# Visualize fixed effects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Individual effects
individual_effects = panel_df.groupby('individual_id')['individual_fe'].first()
ax1.hist(individual_effects, bins=20, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Individual Fixed Effect')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Individual Fixed Effects')

# Time effects
time_effects = panel_df.groupby('time_period')['time_fe'].first()
ax2.bar(range(1, len(time_effects) + 1), time_effects, alpha=0.7)
ax2.set_xlabel('Time Period')
ax2.set_ylabel('Time Fixed Effect')
ax2.set_title('Time Fixed Effects')

plt.tight_layout()
plt.show()

print("✅ Fixed effects models analysis completed")

# =============================================================================
# 7. INSTRUMENTAL VARIABLES (IV) ESTIMATION
# =============================================================================

print("\n" + "="*60)
print("7. INSTRUMENTAL VARIABLES (IV) ESTIMATION")
print("="*60)

# Generate IV data with endogeneity
np.random.seed(42)
n_iv = 1000

# Generate instrument (must be correlated with endogenous variable but not error)
z = np.random.normal(0, 1, n_iv)  # Instrument

# Generate unobserved confounder
u = np.random.normal(0, 1, n_iv)  # Unobserved factor

# Generate endogenous variable (correlated with error through u)
x_endo = 1 + 0.8 * z + 0.5 * u + np.random.normal(0, 0.5, n_iv)

# Generate outcome (true causal effect is 2.0)
y_iv = 1 + 2.0 * x_endo + 0.7 * u + np.random.normal(0, 0.3, n_iv)

iv_data = pd.DataFrame({
    'y': y_iv,
    'x_endo': x_endo,
    'z': z,
    'u': u  # Unobserved in practice
})

print("IV Data Summary:")
print(iv_data.describe())

# Check instrument relevance
first_stage = smf.ols('x_endo ~ z', data=iv_data).fit()
print(f"\nFirst Stage Regression (Instrument Relevance):")
print(f"F-statistic: {first_stage.fvalue:.2f}")
print(f"R-squared: {first_stage.rsquared:.4f}")
print(f"Coefficient on instrument: {first_stage.params['z']:.4f}")

# Weak instrument test
f_stat_threshold = 10
print(f"Weak instrument test: F > {f_stat_threshold}? {'Yes' if first_stage.fvalue > f_stat_threshold else 'No'}")

# Compare OLS vs IV estimates
print(f"\nOLS vs IV Comparison:")
print("-"*30)

# Biased OLS (ignores endogeneity)
ols_biased = smf.ols('y ~ x_endo', data=iv_data).fit()

# IV estimation using 2SLS
if LINEARMODELS_AVAILABLE:
    # Using linearmodels
    iv_model = IV2SLS(iv_data['y'], iv_data[['x_endo']], None, iv_data[['z']]).fit()
    
    print(f"True causal effect: 2.000")
    print(f"OLS estimate: {ols_biased.params['x_endo']:.4f} (SE: {ols_biased.bse['x_endo']:.4f})")
    print(f"IV estimate: {iv_model.params['x_endo']:.4f} (SE: {iv_model.std_errors['x_endo']:.4f})")
    
    print(f"\nIV Results Summary:")
    print(iv_model.summary.tables[1])
else:
    # Manual 2SLS
    # First stage: regress endogenous variable on instrument
    x_endo_hat = first_stage.fittedvalues
    
    # Second stage: regress y on predicted x
    second_stage_data = pd.DataFrame({'y': y_iv, 'x_endo_hat': x_endo_hat})
    second_stage = smf.ols('y ~ x_endo_hat', data=second_stage_data).fit()
    
    print(f"True causal effect: 2.000")
    print(f"OLS estimate: {ols_biased.params['x_endo']:.4f} (SE: {ols_biased.bse['x_endo']:.4f})")
    print(f"IV estimate: {second_stage.params['x_endo_hat']:.4f}")
    print("Note: Manual 2SLS standard errors are incorrect. Use proper IV software.")

# Visualize the IV strategy
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Instrument vs Endogenous Variable
axes[0].scatter(iv_data['z'], iv_data['x_endo'], alpha=0.6)
axes[0].set_xlabel('Instrument (Z)')
axes[0].set_ylabel('Endogenous Variable (X)')
axes[0].set_title('First Stage: Z → X')

# Add regression line
z_range = np.linspace(iv_data['z'].min(), iv_data['z'].max(), 100)
x_pred = first_stage.params['Intercept'] + first_stage.params['z'] * z_range
axes[0].plot(z_range, x_pred, 'r-', linewidth=2)

# Endogenous Variable vs Outcome
axes[1].scatter(iv_data['x_endo'], iv_data['y'], alpha=0.6)
axes[1].set_xlabel('Endogenous Variable (X)')
axes[1].set_ylabel('Outcome (Y)')
axes[1].set_title('Reduced Form: X → Y')

# Instrument vs Outcome
axes[2].scatter(iv_data['z'], iv_data['y'], alpha=0.6)
axes[2].set_xlabel('Instrument (Z)')
axes[2].set_ylabel('Outcome (Y)')
axes[2].set_title('Reduced Form: Z → Y')

plt.tight_layout()
plt.show()

print("✅ Instrumental variables analysis completed")

# =============================================================================
# 8. MODEL COMPARISON AND TESTING
# =============================================================================

print("\n" + "="*60)
print("8. MODEL COMPARISON AND TESTING")
print("="*60)

# F-tests for joint significance
print("F-Tests for Joint Significance:")
print("-"*40)

# Test joint significance of education and experience
wage_data['experience_sq'] = wage_data['experience']**2
full_model = smf.ols('log_wage ~ education + experience + experience_sq + female + married + urban', 
                    data=wage_data).fit()
restricted_model = smf.ols('log_wage ~ female + married + urban', data=wage_data).fit()

# F-test for joint significance of education and experience variables
f_stat = ((restricted_model.ssr - full_model.ssr) / 3) / (full_model.ssr / (len(wage_data) - len(full_model.params)))
f_pvalue = 1 - stats.f.cdf(f_stat, 3, len(wage_data) - len(full_model.params))

print(f"Joint test: education + experience + experience² = 0")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {f_pvalue:.2e}")
print(f"Conclusion: {'Reject' if f_pvalue < 0.05 else 'Fail to reject'} null hypothesis")

# Chow test for structural break
print(f"\nChow Test for Structural Break:")
print("-"*40)

# Split sample by gender
male_data = wage_data[wage_data['female'] == 0]
female_data = wage_data[wage_data['female'] == 1]

# Separate regressions
male_model = smf.ols('log_wage ~ education + experience + married + urban', data=male_data).fit()
female_model = smf.ols('log_wage ~ education + experience + married + urban', data=female_data).fit()

# Pooled regression (without gender dummy)
pooled_model_chow = smf.ols('log_wage ~ education + experience + married + urban', data=wage_data).fit()

# Chow test statistic
ssr_pooled = pooled_model_chow.ssr
ssr_separate = male_model.ssr + female_model.ssr
k = len(male_model.params)  # Number of parameters
n1, n2 = len(male_data), len(female_data)

chow_stat = ((ssr_pooled - ssr_separate) / k) / (ssr_separate / (n1 + n2 - 2*k))
chow_pvalue = 1 - stats.f.cdf(chow_stat, k, n1 + n2 - 2*k)

print(f"Chow test for gender-based structural break:")
print(f"F-statistic: {chow_stat:.4f}")
print(f"p-value: {chow_pvalue:.4f}")
print(f"Conclusion: {'Reject' if chow_pvalue < 0.05 else 'Fail to reject'} null of parameter stability")

# Information criteria comparison
print(f"\nModel Selection Criteria:")
print("-"*40)

models_ic = {
    'Simple (education only)': smf.ols('log_wage ~ education', data=wage_data).fit(),
    'Basic (+ experience)': smf.ols('log_wage ~ education + experience', data=wage_data).fit(),
    'Extended (+ demographics)': smf.ols('log_wage ~ education + experience + female + married', data=wage_data).fit(),
    'Full model': full_model
}

ic_comparison = []
for name, model in models_ic.items():
    ic_comparison.append({
        'Model': name,
        'AIC': model.aic,
        'BIC': model.bic,
        'R²': model.rsquared,
        'Adj R²': model.rsquared_adj
    })

ic_df = pd.DataFrame(ic_comparison)
print(ic_df.round(4))

print(f"\nModel Selection Guidelines:")
print("- Lower AIC/BIC indicates better model")
print("- AIC tends to select more complex models")
print("- BIC penalizes complexity more heavily")

print("✅ Model comparison and testing completed")

print("\n" + "="*70)
print("ECONOMETRICS FUNDAMENTALS SUMMARY")
print("="*70)
print("✅ OLS regression and interpretation")
print("✅ Regression diagnostics and assumption testing")
print("✅ Robust standard errors for heteroscedasticity")
print("✅ Clustered standard errors for grouped data")
print("✅ Fixed effects models for panel data")
print("✅ Instrumental variables for endogeneity")
print("✅ Model comparison and statistical testing")
print("\nNext: Move to advanced econometric methods!")
print("="*70)