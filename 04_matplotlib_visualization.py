# 04 - Data Visualization with Matplotlib
# Matplotlib is the foundational plotting library for Python

"""
MATPLOTLIB
==========

Matplotlib provides:
- Static, animated, and interactive visualizations
- Publication-quality figures
- Wide variety of plot types
- Fine-grained control over plot appearance
- Foundation for other plotting libraries (Seaborn, Plotly, etc.)

Key Components:
- Figure: The entire figure (can contain multiple plots)
- Axes: A single plot within a figure
- Artist: Everything you can see (text, lines, etc.)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set style for better-looking plots
plt.style.use('default')  # You can try 'seaborn', 'ggplot', etc.

print(f"Matplotlib version: {plt.__version__}")

# =============================================================================
# 1. BASIC PLOTTING
# =============================================================================

print("="*60)
print("1. BASIC PLOTTING")
print("="*60)

# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Simple Sine Wave')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True)
plt.show()

# Multiple lines on same plot
plt.figure(figsize=(10, 6))
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2, linestyle='--')
plt.plot(x, y3, label='sin(x)*cos(x)', linewidth=2, linestyle=':')

plt.title('Multiple Functions')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("✅ Basic line plots created")

# =============================================================================
# 2. DIFFERENT PLOT TYPES
# =============================================================================

print("\n" + "="*60)
print("2. DIFFERENT PLOT TYPES")
print("="*60)

# Create sample data
np.random.seed(42)
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100)

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Different Plot Types', fontsize=16)

# Bar plot
axes[0, 0].bar(categories, values, color='skyblue')
axes[0, 0].set_title('Bar Plot')
axes[0, 0].set_ylabel('Values')

# Horizontal bar plot
axes[0, 1].barh(categories, values, color='lightcoral')
axes[0, 1].set_title('Horizontal Bar Plot')
axes[0, 1].set_xlabel('Values')

# Scatter plot
axes[0, 2].scatter(x_scatter, y_scatter, alpha=0.6, color='green')
axes[0, 2].set_title('Scatter Plot')
axes[0, 2].set_xlabel('X values')
axes[0, 2].set_ylabel('Y values')

# Histogram
data_hist = np.random.normal(100, 15, 1000)
axes[1, 0].hist(data_hist, bins=30, color='purple', alpha=0.7)
axes[1, 0].set_title('Histogram')
axes[1, 0].set_xlabel('Values')
axes[1, 0].set_ylabel('Frequency')

# Pie chart
axes[1, 1].pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Pie Chart')

# Box plot
box_data = [np.random.normal(0, std, 100) for std in range(1, 4)]
axes[1, 2].boxplot(box_data, labels=['Group 1', 'Group 2', 'Group 3'])
axes[1, 2].set_title('Box Plot')
axes[1, 2].set_ylabel('Values')

plt.tight_layout()
plt.show()

print("✅ Different plot types demonstrated")

# =============================================================================
# 3. CUSTOMIZING PLOTS
# =============================================================================

print("\n" + "="*60)
print("3. CUSTOMIZING PLOTS")
print("="*60)

# Highly customized plot
fig, ax = plt.subplots(figsize=(12, 8))

# Data
x = np.linspace(0, 4*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Plot with custom styling
line1 = ax.plot(x, y1, color='#FF6B6B', linewidth=3, linestyle='-', 
                marker='o', markersize=4, markevery=10, label='sin(x)')
line2 = ax.plot(x, y2, color='#4ECDC4', linewidth=3, linestyle='--', 
                marker='s', markersize=4, markevery=10, label='cos(x)')

# Customize axes
ax.set_xlim(0, 4*np.pi)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('X values', fontsize=14, fontweight='bold')
ax.set_ylabel('Y values', fontsize=14, fontweight='bold')
ax.set_title('Customized Trigonometric Functions', fontsize=16, fontweight='bold', pad=20)

# Customize grid
ax.grid(True, linestyle=':', alpha=0.6, color='gray')

# Customize legend
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)

# Add annotations
ax.annotate('Maximum', xy=(np.pi/2, 1), xytext=(np.pi/2, 1.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, ha='center')

# Customize spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.tight_layout()
plt.show()

print("✅ Customized plot created")

# =============================================================================
# 4. WORKING WITH PANDAS DATA
# =============================================================================

print("\n" + "="*60)
print("4. WORKING WITH PANDAS DATA")
print("="*60)

# Create sample sales data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
np.random.seed(42)

sales_data = pd.DataFrame({
    'Date': dates,
    'Sales': np.random.normal(1000, 200, 365) + 
             50 * np.sin(2 * np.pi * np.arange(365) / 365),  # Seasonal pattern
    'Marketing_Spend': np.random.normal(100, 20, 365),
    'Temperature': 20 + 15 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 5, 365)
})

# Ensure positive values
sales_data['Sales'] = np.abs(sales_data['Sales'])
sales_data['Marketing_Spend'] = np.abs(sales_data['Marketing_Spend'])

# Add monthly aggregation
sales_data['Month'] = sales_data['Date'].dt.month
monthly_sales = sales_data.groupby('Month').agg({
    'Sales': 'mean',
    'Marketing_Spend': 'mean',
    'Temperature': 'mean'
}).reset_index()

# Plot 1: Time series
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Daily sales over time
ax1.plot(sales_data['Date'], sales_data['Sales'], alpha=0.7, color='blue')
ax1.set_title('Daily Sales Over Time', fontsize=14, fontweight='bold')
ax1.set_ylabel('Sales ($)', fontsize=12)
ax1.grid(True, alpha=0.3)

# Monthly averages
ax2.bar(monthly_sales['Month'], monthly_sales['Sales'], color='lightblue', alpha=0.8)
ax2.set_title('Average Monthly Sales', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Average Sales ($)', fontsize=12)
ax2.set_xticks(range(1, 13))
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Plot 2: Correlation analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Sales vs Marketing Spend
ax1.scatter(sales_data['Marketing_Spend'], sales_data['Sales'], alpha=0.6, color='green')
ax1.set_xlabel('Marketing Spend ($)', fontsize=12)
ax1.set_ylabel('Sales ($)', fontsize=12)
ax1.set_title('Sales vs Marketing Spend', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(sales_data['Marketing_Spend'], sales_data['Sales'], 1)
p = np.poly1d(z)
ax1.plot(sales_data['Marketing_Spend'], p(sales_data['Marketing_Spend']), "r--", alpha=0.8)

# Sales vs Temperature
ax2.scatter(sales_data['Temperature'], sales_data['Sales'], alpha=0.6, color='orange')
ax2.set_xlabel('Temperature (°C)', fontsize=12)
ax2.set_ylabel('Sales ($)', fontsize=12)
ax2.set_title('Sales vs Temperature', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Pandas data visualization completed")

# =============================================================================
# 5. STATISTICAL PLOTS
# =============================================================================

print("\n" + "="*60)
print("5. STATISTICAL PLOTS")
print("="*60)

# Generate sample data for different groups
np.random.seed(42)
group_a = np.random.normal(100, 15, 200)
group_b = np.random.normal(110, 20, 200)
group_c = np.random.normal(95, 12, 200)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Statistical Visualizations', fontsize=16, fontweight='bold')

# Histogram comparison
axes[0, 0].hist(group_a, bins=20, alpha=0.7, label='Group A', color='red')
axes[0, 0].hist(group_b, bins=20, alpha=0.7, label='Group B', color='blue')
axes[0, 0].hist(group_c, bins=20, alpha=0.7, label='Group C', color='green')
axes[0, 0].set_title('Distribution Comparison')
axes[0, 0].set_xlabel('Values')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot comparison
box_data = [group_a, group_b, group_c]
axes[0, 1].boxplot(box_data, labels=['Group A', 'Group B', 'Group C'])
axes[0, 1].set_title('Box Plot Comparison')
axes[0, 1].set_ylabel('Values')
axes[0, 1].grid(True, alpha=0.3)

# Violin plot (requires seaborn-style data)
positions = [1, 2, 3]
parts = axes[1, 0].violinplot(box_data, positions=positions, showmeans=True)
axes[1, 0].set_title('Violin Plot Comparison')
axes[1, 0].set_xlabel('Groups')
axes[1, 0].set_ylabel('Values')
axes[1, 0].set_xticks(positions)
axes[1, 0].set_xticklabels(['Group A', 'Group B', 'Group C'])
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot (quantile-quantile)
from scipy import stats
stats.probplot(group_a, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Group A vs Normal)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Statistical plots created")

# =============================================================================
# 6. HEATMAPS AND CORRELATION MATRICES
# =============================================================================

print("\n" + "="*60)
print("6. HEATMAPS AND CORRELATION MATRICES")
print("="*60)

# Create correlation matrix from sales data
correlation_data = sales_data[['Sales', 'Marketing_Spend', 'Temperature']].corr()

# Create heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Simple heatmap
im1 = ax1.imshow(correlation_data, cmap='coolwarm', aspect='auto')
ax1.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(correlation_data.columns)))
ax1.set_yticks(range(len(correlation_data.columns)))
ax1.set_xticklabels(correlation_data.columns, rotation=45)
ax1.set_yticklabels(correlation_data.columns)

# Add correlation values as text
for i in range(len(correlation_data.columns)):
    for j in range(len(correlation_data.columns)):
        text = ax1.text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im1, ax=ax1)

# 2D histogram (heatmap of scatter data)
ax2.hist2d(sales_data['Marketing_Spend'], sales_data['Sales'], bins=20, cmap='Blues')
ax2.set_xlabel('Marketing Spend ($)', fontsize=12)
ax2.set_ylabel('Sales ($)', fontsize=12)
ax2.set_title('2D Histogram: Sales vs Marketing', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Heatmaps and correlation matrices created")

# =============================================================================
# 7. ADVANCED PLOTTING TECHNIQUES
# =============================================================================

print("\n" + "="*60)
print("7. ADVANCED PLOTTING TECHNIQUES")
print("="*60)

# Multiple y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# First y-axis (Sales)
color = 'tab:blue'
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Sales ($)', color=color, fontsize=12)
line1 = ax1.plot(monthly_sales['Month'], monthly_sales['Sales'], 
                 color=color, marker='o', linewidth=2, label='Sales')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Second y-axis (Temperature)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Temperature (°C)', color=color, fontsize=12)
line2 = ax2.plot(monthly_sales['Month'], monthly_sales['Temperature'], 
                 color=color, marker='s', linewidth=2, linestyle='--', label='Temperature')
ax2.tick_params(axis='y', labelcolor=color)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Sales and Temperature by Month (Dual Y-Axis)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Subplots with shared axes
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
fig.suptitle('Shared Axes Example', fontsize=16, fontweight='bold')

# Generate different datasets
datasets = []
for i in range(4):
    np.random.seed(i)
    x = np.linspace(0, 10, 50)
    y = np.sin(x + i) + np.random.normal(0, 0.1, 50)
    datasets.append((x, y))

# Plot in each subplot
for i, (ax, (x, y)) in enumerate(zip(axes.flat, datasets)):
    ax.plot(x, y, marker='o', markersize=3)
    ax.set_title(f'Dataset {i+1}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Advanced plotting techniques demonstrated")

# =============================================================================
# 8. SAVING PLOTS
# =============================================================================

print("\n" + "="*60)
print("8. SAVING PLOTS")
print("="*60)

# Create a sample plot to save
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/10)

ax.plot(x, y, linewidth=3, color='purple')
ax.set_title('Damped Sine Wave', fontsize=16, fontweight='bold')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.grid(True, alpha=0.3)

# Save in different formats
plt.savefig('damped_sine_wave.png', dpi=300, bbox_inches='tight')
plt.savefig('damped_sine_wave.pdf', bbox_inches='tight')
plt.savefig('damped_sine_wave.svg', bbox_inches='tight')

plt.show()

print("✅ Plot saved in multiple formats:")
print("   - damped_sine_wave.png (high resolution)")
print("   - damped_sine_wave.pdf (vector format)")
print("   - damped_sine_wave.svg (web-friendly vector)")

# =============================================================================
# 9. PRACTICAL EXAMPLE: COMPREHENSIVE BUSINESS DASHBOARD
# =============================================================================

print("\n" + "="*60)
print("9. PRACTICAL EXAMPLE: BUSINESS DASHBOARD")
print("="*60)

# Create comprehensive business data
np.random.seed(42)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Business metrics
revenue = [45000, 52000, 48000, 61000, 58000, 67000,
           72000, 69000, 74000, 71000, 78000, 82000]
costs = [35000, 38000, 36000, 42000, 41000, 45000,
         48000, 46000, 49000, 47000, 51000, 53000]
profit = [r - c for r, c in zip(revenue, costs)]

customers = [1200, 1350, 1280, 1480, 1420, 1580,
             1650, 1590, 1720, 1680, 1820, 1900]

# Create dashboard
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main revenue trend (spans 2 columns)
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(months, revenue, marker='o', linewidth=3, color='green', label='Revenue')
ax1.plot(months, costs, marker='s', linewidth=3, color='red', label='Costs')
ax1.plot(months, profit, marker='^', linewidth=3, color='blue', label='Profit')
ax1.set_title('Financial Performance 2023', fontsize=16, fontweight='bold')
ax1.set_ylabel('Amount ($)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Profit margin pie chart
ax2 = fig.add_subplot(gs[0, 2])
total_revenue = sum(revenue)
total_costs = sum(costs)
total_profit = sum(profit)
sizes = [total_costs, total_profit]
labels = ['Costs', 'Profit']
colors = ['lightcoral', 'lightgreen']
ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax2.set_title('Annual Cost vs Profit', fontsize=12, fontweight='bold')

# Customer growth
ax3 = fig.add_subplot(gs[1, :2])
ax3.bar(months, customers, color='skyblue', alpha=0.8)
ax3.set_title('Customer Growth 2023', fontsize=16, fontweight='bold')
ax3.set_ylabel('Number of Customers', fontsize=12)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# Monthly profit distribution
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(profit, bins=8, color='gold', alpha=0.7, edgecolor='black')
ax4.set_title('Profit Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Profit ($)', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)

# Revenue vs Customers correlation
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(customers, revenue, color='purple', s=100, alpha=0.7)
ax5.set_title('Revenue vs Customers', fontsize=12, fontweight='bold')
ax5.set_xlabel('Customers', fontsize=10)
ax5.set_ylabel('Revenue ($)', fontsize=10)
ax5.grid(True, alpha=0.3)

# Quarterly performance
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
q_revenue = [sum(revenue[i:i+3]) for i in range(0, 12, 3)]
q_profit = [sum(profit[i:i+3]) for i in range(0, 12, 3)]

ax6 = fig.add_subplot(gs[2, 1])
x_pos = np.arange(len(quarters))
width = 0.35
ax6.bar(x_pos - width/2, q_revenue, width, label='Revenue', color='lightblue')
ax6.bar(x_pos + width/2, q_profit, width, label='Profit', color='lightgreen')
ax6.set_title('Quarterly Performance', fontsize=12, fontweight='bold')
ax6.set_xlabel('Quarter', fontsize=10)
ax6.set_ylabel('Amount ($)', fontsize=10)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(quarters)
ax6.legend()

# Key metrics summary
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
metrics_text = f"""
KEY METRICS 2023

Total Revenue: ${total_revenue:,}
Total Costs: ${total_costs:,}
Total Profit: ${total_profit:,}

Profit Margin: {(total_profit/total_revenue)*100:.1f}%
Avg Monthly Revenue: ${total_revenue/12:,.0f}
Customer Growth: {((customers[-1]/customers[0])-1)*100:.1f}%

Best Month: {months[revenue.index(max(revenue))]}
Worst Month: {months[revenue.index(min(revenue))]}
"""
ax7.text(0.1, 0.9, metrics_text, transform=ax7.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.suptitle('Business Performance Dashboard 2023', fontsize=20, fontweight='bold', y=0.98)
plt.savefig('business_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Comprehensive business dashboard created and saved")

print("\n" + "="*70)
print("MATPLOTLIB SUMMARY")
print("="*70)
print("✅ Wide variety of plot types (line, bar, scatter, histogram, etc.)")
print("✅ Highly customizable appearance and styling")
print("✅ Excellent integration with NumPy and Pandas")
print("✅ Support for multiple subplots and complex layouts")
print("✅ Statistical plotting capabilities")
print("✅ Professional publication-quality output")
print("✅ Multiple export formats (PNG, PDF, SVG, etc.)")
print("\nNext: Move to file 05 - Seaborn for statistical visualization!")
print("="*70)