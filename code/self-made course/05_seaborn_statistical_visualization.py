# 05 - Seaborn Statistical Visualization
# Seaborn is built on matplotlib and provides beautiful statistical plots

"""
SEABORN
=======

Seaborn provides:
- Beautiful default styles and color palettes
- High-level statistical plotting functions
- Built-in themes and customization options
- Excellent integration with Pandas DataFrames
- Statistical estimation and visualization
- Complex multi-plot grids

Key Advantages over Matplotlib:
- Less code for complex statistical plots
- Better default aesthetics
- Automatic statistical calculations
- Easy categorical data handling
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

print(f"Seaborn version: {sns.__version__}")

# =============================================================================
# 1. SEABORN BASICS AND STYLING
# =============================================================================

print("="*60)
print("1. SEABORN BASICS AND STYLING")
print("="*60)

# Available styles and palettes
print("Available styles:", sns.axes_style().keys())
print("Available contexts:", ['paper', 'notebook', 'talk', 'poster'])

# Create sample data
np.random.seed(42)
tips = sns.load_dataset("tips")  # Built-in dataset
print("Tips dataset sample:")
print(tips.head())
print(f"Dataset shape: {tips.shape}")

# Basic plot comparison: matplotlib vs seaborn
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Matplotlib style
ax1.scatter(tips['total_bill'], tips['tip'])
ax1.set_title('Matplotlib Default Style')
ax1.set_xlabel('Total Bill')
ax1.set_ylabel('Tip')

# Seaborn style
sns.scatterplot(data=tips, x='total_bill', y='tip', ax=ax2)
ax2.set_title('Seaborn Default Style')

plt.tight_layout()
plt.show()

print("✅ Basic styling comparison completed")

# =============================================================================
# 2. DISTRIBUTION PLOTS
# =============================================================================

print("\n" + "="*60)
print("2. DISTRIBUTION PLOTS")
print("="*60)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Distribution Plots in Seaborn', fontsize=16, fontweight='bold')

# Histogram with KDE
sns.histplot(data=tips, x='total_bill', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histogram with KDE')

# Distribution plot (deprecated but still useful to know)
sns.histplot(data=tips, x='total_bill', stat='density', ax=axes[0, 1])
sns.kdeplot(data=tips, x='total_bill', ax=axes[0, 1])
axes[0, 1].set_title('Density + KDE')

# Box plot
sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[0, 2])
axes[0, 2].set_title('Box Plot by Day')

# Violin plot
sns.violinplot(data=tips, x='day', y='total_bill', ax=axes[1, 0])
axes[1, 0].set_title('Violin Plot by Day')

# Strip plot
sns.stripplot(data=tips, x='day', y='total_bill', ax=axes[1, 1])
axes[1, 1].set_title('Strip Plot by Day')

# Swarm plot
sns.swarmplot(data=tips, x='day', y='total_bill', ax=axes[1, 2])
axes[1, 2].set_title('Swarm Plot by Day')

plt.tight_layout()
plt.show()

print("✅ Distribution plots created")

# =============================================================================
# 3. RELATIONSHIP PLOTS
# =============================================================================

print("\n" + "="*60)
print("3. RELATIONSHIP PLOTS")
print("="*60)

# Create comprehensive relationship analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Relationship Plots', fontsize=16, fontweight='bold')

# Scatter plot with regression line
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time', ax=axes[0, 0])
sns.regplot(data=tips, x='total_bill', y='tip', scatter=False, ax=axes[0, 0], color='red')
axes[0, 0].set_title('Scatter + Regression Line')

# Regression plot with confidence interval
sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[0, 1])
axes[0, 1].set_title('Regression Plot with CI')

# Residual plot
sns.residplot(data=tips, x='total_bill', y='tip', ax=axes[1, 0])
axes[1, 0].set_title('Residual Plot')

# Joint plot (this creates its own figure, so we'll use a different approach)
# Instead, let's create a correlation heatmap
correlation_data = tips.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.show()

# Joint plot (separate figure)
g = sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg', height=8)
g.fig.suptitle('Joint Plot with Regression', y=1.02)
plt.show()

print("✅ Relationship plots created")

# =============================================================================
# 4. CATEGORICAL PLOTS
# =============================================================================

print("\n" + "="*60)
print("4. CATEGORICAL PLOTS")
print("="*60)

# Create comprehensive categorical analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Categorical Plots', fontsize=16, fontweight='bold')

# Bar plot
sns.barplot(data=tips, x='day', y='total_bill', ax=axes[0, 0])
axes[0, 0].set_title('Bar Plot (Mean with CI)')

# Count plot
sns.countplot(data=tips, x='day', ax=axes[0, 1])
axes[0, 1].set_title('Count Plot')

# Point plot
sns.pointplot(data=tips, x='day', y='total_bill', hue='time', ax=axes[0, 2])
axes[0, 2].set_title('Point Plot')
axes[0, 2].legend(title='Time', bbox_to_anchor=(1.05, 1), loc='upper left')

# Box plot with hue
sns.boxplot(data=tips, x='day', y='total_bill', hue='smoker', ax=axes[1, 0])
axes[1, 0].set_title('Box Plot with Hue')

# Violin plot with split
sns.violinplot(data=tips, x='day', y='total_bill', hue='smoker', split=True, ax=axes[1, 1])
axes[1, 1].set_title('Split Violin Plot')

# Bar plot with multiple categories
sns.barplot(data=tips, x='day', y='tip', hue='time', ax=axes[1, 2])
axes[1, 2].set_title('Grouped Bar Plot')

plt.tight_layout()
plt.show()

print("✅ Categorical plots created")

# =============================================================================
# 5. MULTI-PLOT GRIDS
# =============================================================================

print("\n" + "="*60)
print("5. MULTI-PLOT GRIDS")
print("="*60)

# FacetGrid for custom plots
g = sns.FacetGrid(tips, col='time', row='smoker', height=4, aspect=1.2)
g.map(sns.scatterplot, 'total_bill', 'tip', alpha=0.7)
g.add_legend()
g.fig.suptitle('FacetGrid: Tips by Time and Smoker Status', y=1.02)
plt.show()

# PairGrid for pairwise relationships
numeric_tips = tips.select_dtypes(include=[np.number])
g = sns.PairGrid(numeric_tips, height=2.5)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot)
g.fig.suptitle('PairGrid: Pairwise Relationships', y=1.02)
plt.show()

# Pairplot (simpler version of PairGrid)
g = sns.pairplot(tips, hue='time', height=2.5)
g.fig.suptitle('Pairplot with Hue', y=1.02)
plt.show()

print("✅ Multi-plot grids created")

# =============================================================================
# 6. ADVANCED STATISTICAL PLOTS
# =============================================================================

print("\n" + "="*60)
print("6. ADVANCED STATISTICAL PLOTS")
print("="*60)

# Create sample time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
np.random.seed(42)

# Generate realistic business data with trends and seasonality
trend = np.linspace(1000, 1500, 365)
seasonal = 200 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 50, 365)
sales = trend + seasonal + noise

# Create DataFrame
ts_data = pd.DataFrame({
    'Date': dates,
    'Sales': sales,
    'Month': dates.month,
    'Quarter': dates.quarter,
    'Weekday': dates.day_name(),
    'Is_Weekend': dates.weekday >= 5
})

# Add categorical variables
ts_data['Season'] = ts_data['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Advanced Statistical Visualizations', fontsize=16, fontweight='bold')

# Time series plot
sns.lineplot(data=ts_data, x='Date', y='Sales', ax=axes[0, 0])
axes[0, 0].set_title('Time Series Plot')
axes[0, 0].tick_params(axis='x', rotation=45)

# Seasonal decomposition visualization
monthly_avg = ts_data.groupby('Month')['Sales'].mean().reset_index()
sns.barplot(data=monthly_avg, x='Month', y='Sales', ax=axes[0, 1])
axes[0, 1].set_title('Seasonal Pattern (Monthly Averages)')

# Box plot by season
sns.boxplot(data=ts_data, x='Season', y='Sales', ax=axes[1, 0])
axes[1, 0].set_title('Sales Distribution by Season')

# Weekend vs Weekday analysis
sns.violinplot(data=ts_data, x='Is_Weekend', y='Sales', ax=axes[1, 1])
axes[1, 1].set_title('Weekend vs Weekday Sales')
axes[1, 1].set_xticklabels(['Weekday', 'Weekend'])

plt.tight_layout()
plt.show()

# Clustermap (hierarchical clustering heatmap)
# Create correlation matrix for different time periods
quarterly_data = ts_data.groupby(['Quarter', 'Month'])['Sales'].mean().unstack()
g = sns.clustermap(quarterly_data, annot=True, cmap='viridis', figsize=(10, 6))
g.fig.suptitle('Hierarchical Clustering of Monthly Sales by Quarter', y=1.02)
plt.show()

print("✅ Advanced statistical plots created")

# =============================================================================
# 7. CUSTOMIZATION AND THEMES
# =============================================================================

print("\n" + "="*60)
print("7. CUSTOMIZATION AND THEMES")
print("="*60)

# Demonstrate different themes
themes = ['whitegrid', 'darkgrid', 'white', 'dark', 'ticks']
fig, axes = plt.subplots(1, len(themes), figsize=(20, 4))
fig.suptitle('Seaborn Themes Comparison', fontsize=16, fontweight='bold')

for i, theme in enumerate(themes):
    sns.set_style(theme)
    sns.scatterplot(data=tips, x='total_bill', y='tip', ax=axes[i])
    axes[i].set_title(f'Theme: {theme}')

plt.tight_layout()
plt.show()

# Reset to default
sns.set_style("whitegrid")

# Color palettes
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Color Palette Examples', fontsize=16, fontweight='bold')

palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']

for i, palette in enumerate(palettes):
    row, col = i // 3, i % 3
    sns.set_palette(palette)
    sns.barplot(data=tips, x='day', y='total_bill', hue='time', ax=axes[row, col])
    axes[row, col].set_title(f'Palette: {palette}')
    axes[row, col].legend().remove()

plt.tight_layout()
plt.show()

# Custom color palette
custom_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
sns.set_palette(custom_colors)

plt.figure(figsize=(10, 6))
sns.barplot(data=tips, x='day', y='total_bill', hue='time')
plt.title('Custom Color Palette', fontsize=14, fontweight='bold')
plt.show()

print("✅ Themes and customization demonstrated")

# =============================================================================
# 8. PRACTICAL EXAMPLE: COMPREHENSIVE DATA ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("8. PRACTICAL EXAMPLE: EMPLOYEE PERFORMANCE ANALYSIS")
print("="*60)

# Create comprehensive employee dataset
np.random.seed(42)
n_employees = 500

departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
levels = ['Junior', 'Mid', 'Senior', 'Lead']
locations = ['New York', 'San Francisco', 'Chicago', 'Austin', 'Boston']

employee_data = pd.DataFrame({
    'Employee_ID': range(1, n_employees + 1),
    'Department': np.random.choice(departments, n_employees),
    'Level': np.random.choice(levels, n_employees),
    'Location': np.random.choice(locations, n_employees),
    'Years_Experience': np.random.randint(0, 20, n_employees),
    'Age': np.random.randint(22, 65, n_employees),
    'Performance_Score': np.random.normal(75, 15, n_employees),
    'Salary': np.random.normal(80000, 25000, n_employees),
    'Training_Hours': np.random.randint(0, 100, n_employees),
    'Projects_Completed': np.random.randint(1, 20, n_employees)
})

# Ensure realistic relationships
employee_data['Salary'] = (
    employee_data['Salary'] + 
    employee_data['Years_Experience'] * 2000 + 
    employee_data['Performance_Score'] * 500
)

# Ensure positive values
employee_data['Performance_Score'] = np.abs(employee_data['Performance_Score'])
employee_data['Salary'] = np.abs(employee_data['Salary'])

print("Employee dataset sample:")
print(employee_data.head())
print(f"Dataset shape: {employee_data.shape}")

# Create comprehensive analysis dashboard
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# 1. Salary distribution by department
ax1 = fig.add_subplot(gs[0, :2])
sns.boxplot(data=employee_data, x='Department', y='Salary', ax=ax1)
ax1.set_title('Salary Distribution by Department', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# 2. Performance vs Experience
ax2 = fig.add_subplot(gs[0, 2:])
sns.scatterplot(data=employee_data, x='Years_Experience', y='Performance_Score', 
                hue='Level', size='Salary', sizes=(50, 200), ax=ax2)
ax2.set_title('Performance vs Experience', fontsize=14, fontweight='bold')

# 3. Department distribution
ax3 = fig.add_subplot(gs[1, 0])
dept_counts = employee_data['Department'].value_counts()
ax3.pie(dept_counts.values, labels=dept_counts.index, autopct='%1.1f%%')
ax3.set_title('Department Distribution', fontsize=12, fontweight='bold')

# 4. Level distribution
ax4 = fig.add_subplot(gs[1, 1])
sns.countplot(data=employee_data, x='Level', ax=ax4)
ax4.set_title('Level Distribution', fontsize=12, fontweight='bold')

# 5. Training hours by performance
ax5 = fig.add_subplot(gs[1, 2:])
sns.regplot(data=employee_data, x='Training_Hours', y='Performance_Score', ax=ax5)
ax5.set_title('Training Hours vs Performance', fontsize=14, fontweight='bold')

# 6. Salary by level and department
ax6 = fig.add_subplot(gs[2, :2])
sns.barplot(data=employee_data, x='Level', y='Salary', hue='Department', ax=ax6)
ax6.set_title('Average Salary by Level and Department', fontsize=14, fontweight='bold')
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 7. Performance distribution
ax7 = fig.add_subplot(gs[2, 2:])
sns.histplot(data=employee_data, x='Performance_Score', hue='Level', 
             multiple='stack', ax=ax7)
ax7.set_title('Performance Score Distribution by Level', fontsize=14, fontweight='bold')

# 8. Correlation heatmap
ax8 = fig.add_subplot(gs[3, :2])
numeric_cols = ['Years_Experience', 'Age', 'Performance_Score', 'Salary', 
                'Training_Hours', 'Projects_Completed']
correlation_matrix = employee_data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax8)
ax8.set_title('Correlation Matrix', fontsize=14, fontweight='bold')

# 9. Location analysis
ax9 = fig.add_subplot(gs[3, 2:])
location_salary = employee_data.groupby('Location')['Salary'].mean().sort_values(ascending=False)
sns.barplot(x=location_salary.values, y=location_salary.index, ax=ax9)
ax9.set_title('Average Salary by Location', fontsize=14, fontweight='bold')

plt.suptitle('Employee Performance Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
plt.savefig('employee_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical summary
print("\nKEY INSIGHTS:")
print("="*50)
print(f"Total Employees: {len(employee_data)}")
print(f"Average Salary: ${employee_data['Salary'].mean():,.0f}")
print(f"Average Performance Score: {employee_data['Performance_Score'].mean():.1f}")
print(f"Highest Paying Department: {employee_data.groupby('Department')['Salary'].mean().idxmax()}")
print(f"Best Performing Department: {employee_data.groupby('Department')['Performance_Score'].mean().idxmax()}")

# Correlation insights
salary_perf_corr = employee_data['Salary'].corr(employee_data['Performance_Score'])
training_perf_corr = employee_data['Training_Hours'].corr(employee_data['Performance_Score'])
print(f"Salary-Performance Correlation: {salary_perf_corr:.3f}")
print(f"Training-Performance Correlation: {training_perf_corr:.3f}")

print("✅ Comprehensive employee analysis completed")

print("\n" + "="*70)
print("SEABORN SUMMARY")
print("="*70)
print("✅ Beautiful default styles and themes")
print("✅ High-level statistical plotting functions")
print("✅ Excellent categorical data visualization")
print("✅ Built-in statistical calculations and confidence intervals")
print("✅ Multi-plot grids for complex analyses")
print("✅ Easy customization with color palettes")
print("✅ Perfect for exploratory data analysis")
print("✅ Seamless integration with Pandas DataFrames")
print("\nNext: Move to file 06 - Machine Learning with Scikit-learn!")
print("="*70)