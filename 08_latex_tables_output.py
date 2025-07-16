# 10 - LaTeX Tables and Publication-Ready Output
# Creating professional tables for academic papers and reports

"""
LATEX TABLES AND PUBLICATION-READY OUTPUT
=========================================

This module covers creating professional tables for:
- Academic papers
- Policy reports
- Business presentations
- Research documentation

Key Topics:
1. Basic LaTeX table generation
2. Regression tables with multiple models
3. Summary statistics tables
4. Custom formatting and styling
5. Exporting to different formats
6. Integration with econometric results

Required Libraries:
- stargazer: LaTeX regression tables
- tabulate: General table formatting
- pandas: Data manipulation
- statsmodels: Econometric models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Table generation libraries
try:
    from stargazer.stargazer import Stargazer
    STARGAZER_AVAILABLE = True
    print("✅ Stargazer available for LaTeX tables")
except ImportError:
    STARGAZER_AVAILABLE = False
    print("⚠️  Stargazer not available. Install with: pip install stargazer")

from tabulate import tabulate
import re

print("LaTeX Tables and Publication-Ready Output")
print("="*50)

# =============================================================================
# 1. SAMPLE DATA GENERATION
# =============================================================================

def generate_publication_data(n=1000, seed=42):
    """Generate realistic data for publication tables"""
    np.random.seed(seed)
    
    # Demographics
    age = np.random.normal(35, 10, n)
    age = np.clip(age, 18, 65)
    
    education = np.random.normal(14, 3, n)
    education = np.clip(education, 8, 20)
    
    experience = np.random.normal(10, 8, n)
    experience = np.clip(experience, 0, age - education - 6)
    
    female = np.random.binomial(1, 0.5, n)
    married = np.random.binomial(1, 0.6, n)
    urban = np.random.binomial(1, 0.7, n)
    
    # Industry categories
    industries = ['Manufacturing', 'Services', 'Technology', 'Healthcare', 'Finance']
    industry = np.random.choice(industries, n)
    
    # Generate correlated error
    epsilon = np.random.normal(0, 0.3, n)
    
    # Wage equation
    log_wage = (
        1.2 +
        0.08 * education +
        0.03 * experience +
        -0.0005 * experience**2 +
        -0.15 * female +
        0.10 * married +
        0.12 * urban +
        epsilon
    )
    
    wage = np.exp(log_wage)
    
    # Additional variables
    hours_worked = np.random.normal(40, 8, n)
    hours_worked = np.clip(hours_worked, 20, 60)
    
    job_satisfaction = np.random.randint(1, 8, n)  # 1-7 scale
    
    return pd.DataFrame({
        'wage': wage,
        'log_wage': log_wage,
        'age': age,
        'education': education,
        'experience': experience,
        'female': female,
        'married': married,
        'urban': urban,
        'industry': industry,
        'hours_worked': hours_worked,
        'job_satisfaction': job_satisfaction
    })

# Generate data
pub_data = generate_publication_data(n=1000)

print("Publication Data Generated:")
print(pub_data.head())
print(f"Dataset shape: {pub_data.shape}")

# =============================================================================
# 2. SUMMARY STATISTICS TABLES
# =============================================================================

print("\n" + "="*60)
print("2. SUMMARY STATISTICS TABLES")
print("="*60)

def create_summary_stats_table(data, variables, by_group=None, latex=True):
    """Create professional summary statistics table"""
    
    if by_group is None:
        # Overall summary statistics
        summary_stats = []
        
        for var in variables:
            if var in data.columns:
                series = data[var]
                if series.dtype in ['int64', 'float64']:
                    stats_dict = {
                        'Variable': var,
                        'N': len(series.dropna()),
                        'Mean': series.mean(),
                        'Std Dev': series.std(),
                        'Min': series.min(),
                        'Max': series.max(),
                        'P25': series.quantile(0.25),
                        'P50': series.quantile(0.50),
                        'P75': series.quantile(0.75)
                    }
                else:
                    # Categorical variable
                    stats_dict = {
                        'Variable': var,
                        'N': len(series.dropna()),
                        'Mean': series.value_counts().iloc[0] / len(series) if len(series) > 0 else 0,
                        'Std Dev': np.nan,
                        'Min': np.nan,
                        'Max': np.nan,
                        'P25': np.nan,
                        'P50': np.nan,
                        'P75': np.nan
                    }
                summary_stats.append(stats_dict)
        
        summary_df = pd.DataFrame(summary_stats)
        
    else:
        # Summary statistics by group
        groups = data[by_group].unique()
        summary_stats = []
        
        for var in variables:
            if var in data.columns and var != by_group:
                row_data = {'Variable': var}
                
                for group in groups:
                    group_data = data[data[by_group] == group][var]
                    if group_data.dtype in ['int64', 'float64']:
                        row_data[f'{group}_N'] = len(group_data.dropna())
                        row_data[f'{group}_Mean'] = group_data.mean()
                        row_data[f'{group}_SD'] = group_data.std()
                    else:
                        row_data[f'{group}_N'] = len(group_data.dropna())
                        row_data[f'{group}_Mean'] = group_data.value_counts().iloc[0] / len(group_data) if len(group_data) > 0 else 0
                        row_data[f'{group}_SD'] = np.nan
                
                summary_stats.append(row_data)
        
        summary_df = pd.DataFrame(summary_stats)
    
    # Format for display
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(3)
    
    if latex:
        return summary_df.to_latex(index=False, float_format='%.3f', na_rep='--')
    else:
        return summary_df

# Overall summary statistics
variables_for_summary = ['wage', 'age', 'education', 'experience', 'hours_worked', 'job_satisfaction']
summary_table = create_summary_stats_table(pub_data, variables_for_summary, latex=False)

print("Summary Statistics Table:")
print(tabulate(summary_table, headers='keys', tablefmt='grid', floatfmt='.3f'))

# Summary by gender
summary_by_gender = create_summary_stats_table(pub_data, variables_for_summary, by_group='female', latex=False)

print(f"\nSummary Statistics by Gender:")
print(tabulate(summary_by_gender, headers='keys', tablefmt='grid', floatfmt='.3f'))

# LaTeX output
if True:  # Set to True to see LaTeX code
    print(f"\nLaTeX Summary Statistics Table:")
    print("-"*50)
    latex_summary = create_summary_stats_table(pub_data, variables_for_summary, latex=True)
    print(latex_summary)

print("✅ Summary statistics tables created")

# =============================================================================
# 3. REGRESSION TABLES WITH STARGAZER
# =============================================================================

print("\n" + "="*60)
print("3. REGRESSION TABLES WITH STARGAZER")
print("="*60)

# Estimate multiple regression models
model1 = smf.ols('log_wage ~ education', data=pub_data).fit()
model2 = smf.ols('log_wage ~ education + experience', data=pub_data).fit()
model3 = smf.ols('log_wage ~ education + experience + I(experience**2)', data=pub_data).fit()
model4 = smf.ols('log_wage ~ education + experience + I(experience**2) + female + married + urban', data=pub_data).fit()

models = [model1, model2, model3, model4]
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4']

print("Regression Models Estimated:")
for i, model in enumerate(models):
    print(f"{model_names[i]}: R² = {model.rsquared:.4f}, N = {model.nobs}")

if STARGAZER_AVAILABLE:
    print(f"\nCreating Stargazer Regression Table:")
    print("-"*50)
    
    # Create stargazer table
    stargazer = Stargazer(models)
    
    # Customize the table
    stargazer.title('Wage Regression Results')
    stargazer.custom_columns(model_names, [1, 1, 1, 1])
    
    # Add notes
    stargazer.add_line('Sample', ['Full', 'Full', 'Full', 'Full'])
    stargazer.add_line('Controls', ['No', 'No', 'No', 'Yes'])
    
    # Show different output formats
    print("HTML Table:")
    print(stargazer.render_html())
    
    print(f"\nLaTeX Table:")
    print(stargazer.render_latex())
    
else:
    print("Creating manual regression table...")
    
    # Manual regression table creation
    reg_table_data = []
    
    for i, model in enumerate(models):
        model_data = {
            'Model': model_names[i],
            'Education_Coef': model.params.get('education', np.nan),
            'Education_SE': model.bse.get('education', np.nan),
            'Experience_Coef': model.params.get('experience', np.nan),
            'Experience_SE': model.bse.get('experience', np.nan),
            'Female_Coef': model.params.get('female', np.nan),
            'Female_SE': model.bse.get('female', np.nan),
            'R_squared': model.rsquared,
            'N': int(model.nobs)
        }
        reg_table_data.append(model_data)
    
    reg_table_df = pd.DataFrame(reg_table_data)
    print("Manual Regression Table:")
    print(tabulate(reg_table_df, headers='keys', tablefmt='grid', floatfmt='.4f'))

print("✅ Regression tables created")

# =============================================================================
# 4. CUSTOM TABLE FORMATTING FUNCTIONS
# =============================================================================

print("\n" + "="*60)
print("4. CUSTOM TABLE FORMATTING FUNCTIONS")
print("="*60)

def format_regression_table(models, model_names=None, dependent_var="Dependent Variable", 
                          title="Regression Results", latex=True):
    """Create a custom formatted regression table"""
    
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]
    
    # Extract coefficients and standard errors
    all_vars = set()
    for model in models:
        all_vars.update(model.params.index)
    
    # Remove intercept for cleaner display
    all_vars.discard('Intercept')
    all_vars = sorted(list(all_vars))
    
    # Build table
    table_data = []
    
    # Header
    header = ['Variable'] + model_names
    table_data.append(header)
    
    # Coefficients and standard errors
    for var in all_vars:
        # Coefficient row
        coef_row = [var]
        se_row = ['']
        
        for model in models:
            if var in model.params:
                coef = model.params[var]
                se = model.bse[var]
                pval = model.pvalues[var]
                
                # Add significance stars
                stars = ''
                if pval < 0.01:
                    stars = '***'
                elif pval < 0.05:
                    stars = '**'
                elif pval < 0.1:
                    stars = '*'
                
                coef_row.append(f"{coef:.4f}{stars}")
                se_row.append(f"({se:.4f})")
            else:
                coef_row.append('')
                se_row.append('')
        
        table_data.append(coef_row)
        table_data.append(se_row)
    
    # Model statistics
    table_data.append([''] * len(header))  # Empty row
    
    # R-squared
    r2_row = ['R²']
    for model in models:
        r2_row.append(f"{model.rsquared:.4f}")
    table_data.append(r2_row)
    
    # Observations
    n_row = ['Observations']
    for model in models:
        n_row.append(f"{int(model.nobs)}")
    table_data.append(n_row)
    
    if latex:
        # Convert to LaTeX
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "\\centering\n"
        latex_table += f"\\caption{{{title}}}\n"
        latex_table += "\\begin{tabular}{l" + "c" * len(model_names) + "}\n"
        latex_table += "\\hline\\hline\n"
        
        for i, row in enumerate(table_data):
            if i == 0:  # Header
                latex_table += " & ".join(row) + " \\\\\n"
                latex_table += "\\hline\n"
            elif i == len(table_data) - 3:  # Before statistics
                latex_table += "\\hline\n"
                latex_table += " & ".join(row) + " \\\\\n"
            else:
                latex_table += " & ".join(row) + " \\\\\n"
        
        latex_table += "\\hline\\hline\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\begin{tablenotes}\n"
        latex_table += "\\small\n"
        latex_table += "\\item Notes: Standard errors in parentheses. "
        latex_table += "*** p$<$0.01, ** p$<$0.05, * p$<$0.1\n"
        latex_table += "\\end{tablenotes}\n"
        latex_table += "\\end{table}\n"
        
        return latex_table
    else:
        return tabulate(table_data[1:], headers=table_data[0], tablefmt='grid')

# Create custom formatted table
print("Custom Formatted Regression Table:")
custom_table = format_regression_table(models, model_names, 
                                     dependent_var="Log Wage", 
                                     title="Determinants of Wages",
                                     latex=False)
print(custom_table)

print(f"\nLaTeX Version:")
print("-"*30)
latex_custom = format_regression_table(models, model_names, 
                                     dependent_var="Log Wage", 
                                     title="Determinants of Wages",
                                     latex=True)
print(latex_custom)

print("✅ Custom table formatting completed")

# =============================================================================
# 5. CORRELATION AND DESCRIPTIVE TABLES
# =============================================================================

print("\n" + "="*60)
print("5. CORRELATION AND DESCRIPTIVE TABLES")
print("="*60)

def create_correlation_table(data, variables, latex=True):
    """Create a professional correlation table"""
    
    # Calculate correlation matrix
    corr_data = data[variables].corr()
    
    # Create lower triangular matrix (more professional)
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    corr_data_lower = corr_data.mask(mask)
    
    if latex:
        # Convert to LaTeX
        n_vars = len(variables)
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{Correlation Matrix}\n"
        latex_table += f"\\begin{tabular}{{l{'c' * n_vars}}}\n"
        latex_table += "\\hline\\hline\n"
        
        # Header
        header = "Variable & " + " & ".join([f"({i+1})" for i in range(n_vars)]) + " \\\\\n"
        latex_table += header
        latex_table += "\\hline\n"
        
        # Rows
        for i, var in enumerate(variables):
            row = f"({i+1}) {var}"
            for j in range(n_vars):
                if j <= i:
                    if i == j:
                        row += " & 1.000"
                    else:
                        corr_val = corr_data.iloc[i, j]
                        row += f" & {corr_val:.3f}"
                else:
                    row += " & "
            row += " \\\\\n"
            latex_table += row
        
        latex_table += "\\hline\\hline\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        
        return latex_table
    else:
        return corr_data_lower.round(3)

# Create correlation table
corr_vars = ['wage', 'age', 'education', 'experience', 'hours_worked']
correlation_table = create_correlation_table(pub_data, corr_vars, latex=False)

print("Correlation Table:")
print(correlation_table)

print(f"\nLaTeX Correlation Table:")
print("-"*30)
latex_corr = create_correlation_table(pub_data, corr_vars, latex=True)
print(latex_corr)

# Industry distribution table
def create_frequency_table(data, variable, latex=True):
    """Create frequency table for categorical variables"""
    
    freq_table = data[variable].value_counts().reset_index()
    freq_table.columns = [variable, 'Frequency']
    freq_table['Percentage'] = (freq_table['Frequency'] / freq_table['Frequency'].sum() * 100).round(2)
    freq_table['Cumulative %'] = freq_table['Percentage'].cumsum().round(2)
    
    if latex:
        return freq_table.to_latex(index=False, float_format='%.2f')
    else:
        return freq_table

industry_table = create_frequency_table(pub_data, 'industry', latex=False)
print(f"\nIndustry Distribution:")
print(tabulate(industry_table, headers='keys', tablefmt='grid'))

print("✅ Correlation and descriptive tables completed")

# =============================================================================
# 6. ROBUSTNESS AND SENSITIVITY TABLES
# =============================================================================

print("\n" + "="*60)
print("6. ROBUSTNESS AND SENSITIVITY TABLES")
print("="*60)

def create_robustness_table(data, dependent_var, main_var, controls, 
                          subsamples=None, latex=True):
    """Create robustness check table"""
    
    results = []
    
    # Base specification
    base_formula = f"{dependent_var} ~ {main_var}"
    if controls:
        base_formula += " + " + " + ".join(controls)
    
    base_model = smf.ols(base_formula, data=data).fit()
    results.append({
        'Specification': 'Full Sample',
        'Coefficient': base_model.params[main_var],
        'Std_Error': base_model.bse[main_var],
        'P_Value': base_model.pvalues[main_var],
        'N': int(base_model.nobs),
        'R_Squared': base_model.rsquared
    })
    
    # Subsample analyses
    if subsamples:
        for subsample_name, condition in subsamples.items():
            subsample_data = data[condition]
            if len(subsample_data) > 50:  # Ensure sufficient observations
                try:
                    subsample_model = smf.ols(base_formula, data=subsample_data).fit()
                    results.append({
                        'Specification': subsample_name,
                        'Coefficient': subsample_model.params[main_var],
                        'Std_Error': subsample_model.bse[main_var],
                        'P_Value': subsample_model.pvalues[main_var],
                        'N': int(subsample_model.nobs),
                        'R_Squared': subsample_model.rsquared
                    })
                except:
                    pass  # Skip if model fails
    
    # Different specifications
    # Without controls
    simple_formula = f"{dependent_var} ~ {main_var}"
    simple_model = smf.ols(simple_formula, data=data).fit()
    results.append({
        'Specification': 'No Controls',
        'Coefficient': simple_model.params[main_var],
        'Std_Error': simple_model.bse[main_var],
        'P_Value': simple_model.pvalues[main_var],
        'N': int(simple_model.nobs),
        'R_Squared': simple_model.rsquared
    })
    
    # With additional controls
    if 'age' not in controls:
        extended_formula = base_formula + " + age"
        try:
            extended_model = smf.ols(extended_formula, data=data).fit()
            results.append({
                'Specification': 'Extended Controls',
                'Coefficient': extended_model.params[main_var],
                'Std_Error': extended_model.bse[main_var],
                'P_Value': extended_model.pvalues[main_var],
                'N': int(extended_model.nobs),
                'R_Squared': extended_model.rsquared
            })
        except:
            pass
    
    results_df = pd.DataFrame(results)
    
    # Add significance indicators
    results_df['Significance'] = results_df['P_Value'].apply(
        lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    )
    
    if latex:
        # Format for LaTeX
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{Robustness Checks}\n"
        latex_table += "\\begin{tabular}{lcccc}\n"
        latex_table += "\\hline\\hline\n"
        latex_table += "Specification & Coefficient & Std. Error & N & R² \\\\\n"
        latex_table += "\\hline\n"
        
        for _, row in results_df.iterrows():
            latex_table += f"{row['Specification']} & "
            latex_table += f"{row['Coefficient']:.4f}{row['Significance']} & "
            latex_table += f"({row['Std_Error']:.4f}) & "
            latex_table += f"{row['N']} & "
            latex_table += f"{row['R_Squared']:.4f} \\\\\n"
        
        latex_table += "\\hline\\hline\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\begin{tablenotes}\n"
        latex_table += "\\small\n"
        latex_table += "\\item Notes: *** p$<$0.01, ** p$<$0.05, * p$<$0.1\n"
        latex_table += "\\end{tablenotes}\n"
        latex_table += "\\end{table}\n"
        
        return latex_table
    else:
        return results_df

# Create robustness table
subsamples = {
    'Male Only': pub_data['female'] == 0,
    'Female Only': pub_data['female'] == 1,
    'High Education': pub_data['education'] >= pub_data['education'].median(),
    'Urban Only': pub_data['urban'] == 1
}

robustness_table = create_robustness_table(
    pub_data, 
    'log_wage', 
    'education', 
    ['experience', 'female', 'married'],
    subsamples,
    latex=False
)

print("Robustness Checks Table:")
print(tabulate(robustness_table[['Specification', 'Coefficient', 'Std_Error', 'N', 'R_Squared']], 
               headers='keys', tablefmt='grid', floatfmt='.4f'))

print("✅ Robustness tables completed")

# =============================================================================
# 7. EXPORT FUNCTIONS AND UTILITIES
# =============================================================================

print("\n" + "="*60)
print("7. EXPORT FUNCTIONS AND UTILITIES")
print("="*60)

def save_table_to_file(table_content, filename, format_type='latex'):
    """Save table to file"""
    
    if format_type == 'latex':
        with open(f"{filename}.tex", 'w') as f:
            f.write(table_content)
        print(f"✅ LaTeX table saved to {filename}.tex")
    
    elif format_type == 'html':
        with open(f"{filename}.html", 'w') as f:
            f.write(table_content)
        print(f"✅ HTML table saved to {filename}.html")

def create_table_template():
    """Create a template for manual table creation"""
    
    template = """
% LaTeX Table Template
\\begin{table}[htbp]
\\centering
\\caption{Your Table Title}
\\label{tab:your_label}
\\begin{tabular}{lcccc}
\\hline\\hline
Variable & Model 1 & Model 2 & Model 3 & Model 4 \\\\
\\hline
Education & 0.0800*** & 0.0750*** & 0.0720*** & 0.0680*** \\\\
          & (0.0050)  & (0.0048)  & (0.0047)  & (0.0045)  \\\\
Experience &          & 0.0300*** & 0.0350*** & 0.0320*** \\\\
           &          & (0.0030)  & (0.0035)  & (0.0033)  \\\\
\\hline
R² & 0.250 & 0.280 & 0.290 & 0.320 \\\\
N  & 1000  & 1000  & 1000  & 1000  \\\\
\\hline\\hline
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Notes: Standard errors in parentheses. 
*** p$<$0.01, ** p$<$0.05, * p$<$0.1
\\end{tablenotes}
\\end{table}
"""
    return template

# Save example tables
print("Saving example tables to files...")

# Save regression table
if STARGAZER_AVAILABLE:
    stargazer_output = Stargazer(models).render_latex()
    save_table_to_file(stargazer_output, "regression_table", "latex")

# Save custom table
custom_latex = format_regression_table(models, model_names, latex=True)
save_table_to_file(custom_latex, "custom_regression_table", "latex")

# Save correlation table
corr_latex = create_correlation_table(pub_data, corr_vars, latex=True)
save_table_to_file(corr_latex, "correlation_table", "latex")

# Create and save template
template = create_table_template()
save_table_to_file(template, "table_template", "latex")

print("✅ Tables saved to files")

# =============================================================================
# 8. BEST PRACTICES AND STYLE GUIDE
# =============================================================================

print("\n" + "="*60)
print("8. BEST PRACTICES AND STYLE GUIDE")
print("="*60)

best_practices = """
LATEX TABLE BEST PRACTICES
=========================

1. STRUCTURE AND FORMATTING:
   ✅ Use consistent decimal places (usually 3-4)
   ✅ Align numbers by decimal point
   ✅ Use horizontal lines sparingly (top, bottom, after header)
   ✅ Group related variables together
   ✅ Put standard errors in parentheses below coefficients

2. SIGNIFICANCE INDICATORS:
   ✅ Use *** for p<0.01, ** for p<0.05, * for p<0.1
   ✅ Place stars immediately after coefficients
   ✅ Explain significance levels in table notes

3. MODEL INFORMATION:
   ✅ Always include R², N (observations)
   ✅ Consider including F-statistic for joint tests
   ✅ Add model specification details in notes
   ✅ Include fixed effects indicators when relevant

4. VARIABLE NAMES:
   ✅ Use descriptive but concise variable names
   ✅ Group dummy variables logically
   ✅ Consider using variable labels instead of raw names

5. TABLE NOTES:
   ✅ Explain data source and sample restrictions
   ✅ Define key variables if not obvious
   ✅ Mention robust/clustered standard errors if used
   ✅ Include any important methodological details

6. PRESENTATION:
   ✅ Tables should be self-contained
   ✅ Use informative captions
   ✅ Number tables consistently
   ✅ Reference tables in text

7. COMMON MISTAKES TO AVOID:
   ❌ Too many decimal places
   ❌ Inconsistent formatting across tables
   ❌ Missing standard errors or significance tests
   ❌ Unclear variable definitions
   ❌ Tables that are too wide for the page
   ❌ Missing sample size information

8. JOURNAL-SPECIFIC REQUIREMENTS:
   ✅ Check journal style guides
   ✅ Some prefer t-statistics instead of standard errors
   ✅ Some require specific table formatting
   ✅ Consider word/page limits when designing tables
"""

print(best_practices)

# Example of a well-formatted table
print(f"\nEXAMPLE: Well-Formatted Regression Table")
print("="*50)

well_formatted_table = """
\\begin{table}[htbp]
\\centering
\\caption{Determinants of Log Wages}
\\label{tab:wage_regression}
\\begin{tabular}{lcccc}
\\hline\\hline
                    & (1)      & (2)      & (3)      & (4)      \\\\
                    & Basic    & + Exp.   & + Exp²   & Full     \\\\
\\hline
Education           & 0.0823*** & 0.0756*** & 0.0751*** & 0.0679*** \\\\
                    & (0.0051)  & (0.0049)  & (0.0049)  & (0.0047)  \\\\
Experience          &          & 0.0287*** & 0.0341*** & 0.0298*** \\\\
                    &          & (0.0031)  & (0.0055)  & (0.0052)  \\\\
Experience²         &          &          & -0.0005   & -0.0004   \\\\
                    &          &          & (0.0005)  & (0.0005)  \\\\
Female              &          &          &          & -0.1489*** \\\\
                    &          &          &          & (0.0188)  \\\\
Married             &          &          &          & 0.0967*** \\\\
                    &          &          &          & (0.0195)  \\\\
Urban               &          &          &          & 0.1156*** \\\\
                    &          &          &          & (0.0208)  \\\\
Constant            & 1.0459*** & 1.1287*** & 1.1063*** & 1.2456*** \\\\
                    & (0.0728)  & (0.0731)  & (0.0738)  & (0.0751)  \\\\
\\hline
Observations        & 1,000    & 1,000    & 1,000    & 1,000    \\\\
R²                  & 0.251    & 0.281    & 0.282    & 0.321    \\\\
Adjusted R²         & 0.250    & 0.279    & 0.279    & 0.317    \\\\
\\hline\\hline
\\end{tabular}
\\begin{tablenotes}
\\small
\\item \\textit{Notes:} Dependent variable is log hourly wage. 
Standard errors in parentheses. Sample includes full-time workers 
aged 18-65. *** p$<$0.01, ** p$<$0.05, * p$<$0.1.
\\end{tablenotes}
\\end{table}
"""

print(well_formatted_table)

print("✅ Best practices guide completed")

print("\n" + "="*70)
print("LATEX TABLES AND PUBLICATION OUTPUT SUMMARY")
print("="*70)
print("✅ Summary statistics tables")
print("✅ Regression tables with Stargazer")
print("✅ Custom table formatting functions")
print("✅ Correlation and descriptive tables")
print("✅ Robustness and sensitivity tables")
print("✅ Export functions and file saving")
print("✅ Best practices and style guide")
print("✅ Professional LaTeX table templates")
print("\nYou're now ready to create publication-quality tables!")
print("="*70)