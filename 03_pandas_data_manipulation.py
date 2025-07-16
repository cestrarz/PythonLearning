# 03 - Pandas Data Manipulation
# Pandas is the go-to library for data analysis and manipulation in Python

"""
PANDAS (Python Data Analysis Library)
=====================================

Pandas provides:
- DataFrame and Series data structures
- Data cleaning and transformation tools
- File I/O (CSV, Excel, JSON, SQL, etc.)
- Data aggregation and grouping
- Time series analysis
- Missing data handling

Key Data Structures:
- Series: 1D labeled array
- DataFrame: 2D labeled data structure (like Excel spreadsheet)
"""

import pandas as pd
import numpy as np

print(f"Pandas version: {pd.__version__}")

# =============================================================================
# 1. CREATING DATAFRAMES AND SERIES
# =============================================================================

print("="*60)
print("1. CREATING DATAFRAMES AND SERIES")
print("="*60)

# Creating a Series
series_data = pd.Series([10, 20, 30, 40, 50], 
                       index=['A', 'B', 'C', 'D', 'E'])
print("Series:")
print(series_data)
print(f"Series type: {type(series_data)}")

# Creating DataFrame from dictionary
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['New York', 'Boston', 'Chicago', 'Denver', 'Seattle'],
    'Salary': [50000, 60000, 55000, 52000, 58000],
    'Department': ['IT', 'Finance', 'IT', 'HR', 'Finance']
}

df = pd.DataFrame(data_dict)
print(f"\nDataFrame from dictionary:")
print(df)
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame columns: {list(df.columns)}")
print(f"DataFrame index: {list(df.index)}")

# Creating DataFrame from lists
data_lists = [
    ['Frank', 29, 'Miami', 54000, 'Marketing'],
    ['Grace', 31, 'Austin', 56000, 'IT'],
    ['Henry', 27, 'Portland', 51000, 'HR']
]
df_from_lists = pd.DataFrame(data_lists, 
                            columns=['Name', 'Age', 'City', 'Salary', 'Department'])
print(f"\nDataFrame from lists:")
print(df_from_lists)

# =============================================================================
# 2. BASIC DATAFRAME OPERATIONS
# =============================================================================

print("\n" + "="*60)
print("2. BASIC DATAFRAME OPERATIONS")
print("="*60)

# Basic info about the DataFrame
print("DataFrame Info:")
print(f"Shape: {df.shape}")
print(f"Size: {df.size}")
print(f"Data types:\n{df.dtypes}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nLast 2 rows:")
print(df.tail(2))

# Statistical summary
print(f"\nStatistical summary:")
print(df.describe())

# Info about the DataFrame
print(f"\nDataFrame info:")
df.info()

# =============================================================================
# 3. SELECTING AND INDEXING DATA
# =============================================================================

print("\n" + "="*60)
print("3. SELECTING AND INDEXING DATA")
print("="*60)

# Selecting columns
print("Single column (Series):")
print(df['Name'])
print(f"Type: {type(df['Name'])}")

print(f"\nMultiple columns (DataFrame):")
print(df[['Name', 'Salary']])

# Selecting rows by index
print(f"\nFirst row:")
print(df.iloc[0])  # By position
print(f"\nFirst row by label:")
print(df.loc[0])   # By label (same as iloc when index is numeric)

# Selecting specific rows and columns
print(f"\nSpecific rows and columns:")
print(df.loc[0:2, ['Name', 'Age']])  # Rows 0-2, specific columns
print(f"\nUsing iloc:")
print(df.iloc[0:3, 0:2])  # First 3 rows, first 2 columns

# Boolean indexing (filtering)
print(f"\nFiltering: Employees with salary > 55000:")
high_salary = df[df['Salary'] > 55000]
print(high_salary)

print(f"\nMultiple conditions: IT department with salary > 50000:")
it_high_salary = df[(df['Department'] == 'IT') & (df['Salary'] > 50000)]
print(it_high_salary)

# =============================================================================
# 4. DATA MANIPULATION
# =============================================================================

print("\n" + "="*60)
print("4. DATA MANIPULATION")
print("="*60)

# Adding new columns
df_copy = df.copy()  # Work with a copy
df_copy['Salary_K'] = df_copy['Salary'] / 1000  # Salary in thousands
df_copy['Age_Group'] = df_copy['Age'].apply(lambda x: 'Young' if x < 30 else 'Senior')

print("DataFrame with new columns:")
print(df_copy[['Name', 'Age', 'Salary', 'Salary_K', 'Age_Group']])

# Modifying existing data
df_copy.loc[df_copy['City'] == 'New York', 'City'] = 'NYC'
print(f"\nAfter modifying New York to NYC:")
print(df_copy[['Name', 'City']])

# Dropping columns and rows
df_dropped = df_copy.drop(['Salary_K'], axis=1)  # Drop column
print(f"\nAfter dropping Salary_K column:")
print(df_dropped.columns.tolist())

df_dropped_rows = df_copy.drop([0, 1])  # Drop rows by index
print(f"\nAfter dropping first 2 rows:")
print(df_dropped_rows[['Name', 'Age']])

# =============================================================================
# 5. SORTING AND RANKING
# =============================================================================

print("\n" + "="*60)
print("5. SORTING AND RANKING")
print("="*60)

# Sort by single column
sorted_by_age = df.sort_values('Age')
print("Sorted by Age:")
print(sorted_by_age[['Name', 'Age']])

# Sort by multiple columns
sorted_multi = df.sort_values(['Department', 'Salary'], ascending=[True, False])
print(f"\nSorted by Department (asc) then Salary (desc):")
print(sorted_multi[['Name', 'Department', 'Salary']])

# Ranking
df_with_rank = df.copy()
df_with_rank['Salary_Rank'] = df['Salary'].rank(ascending=False)
print(f"\nWith salary ranking:")
print(df_with_rank[['Name', 'Salary', 'Salary_Rank']])

# =============================================================================
# 6. GROUPING AND AGGREGATION
# =============================================================================

print("\n" + "="*60)
print("6. GROUPING AND AGGREGATION")
print("="*60)

# Group by single column
dept_groups = df.groupby('Department')
print("Average salary by department:")
print(dept_groups['Salary'].mean())

print(f"\nMultiple aggregations by department:")
dept_agg = dept_groups.agg({
    'Salary': ['mean', 'min', 'max', 'count'],
    'Age': ['mean', 'min', 'max']
})
print(dept_agg)

# Group by multiple columns (add more data first)
extended_data = pd.concat([df, df_from_lists], ignore_index=True)
print(f"\nExtended dataset:")
print(extended_data)

city_dept_groups = extended_data.groupby(['Department', 'City'])
print(f"\nAverage salary by Department and City:")
print(city_dept_groups['Salary'].mean())

# =============================================================================
# 7. HANDLING MISSING DATA
# =============================================================================

print("\n" + "="*60)
print("7. HANDLING MISSING DATA")
print("="*60)

# Create data with missing values
data_with_na = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, np.nan, 35, 28, 32],
    'Salary': [50000, 60000, np.nan, 52000, 58000],
    'City': ['New York', 'Boston', 'Chicago', None, 'Seattle']
}

df_na = pd.DataFrame(data_with_na)
print("DataFrame with missing values:")
print(df_na)

# Check for missing values
print(f"\nMissing values per column:")
print(df_na.isnull().sum())

print(f"\nRows with any missing values:")
print(df_na[df_na.isnull().any(axis=1)])

# Handle missing values
# Fill with specific values
df_filled = df_na.fillna({
    'Age': df_na['Age'].mean(),
    'Salary': df_na['Salary'].median(),
    'City': 'Unknown'
})
print(f"\nAfter filling missing values:")
print(df_filled)

# Drop rows with missing values
df_dropped_na = df_na.dropna()
print(f"\nAfter dropping rows with missing values:")
print(df_dropped_na)

# =============================================================================
# 8. STRING OPERATIONS
# =============================================================================

print("\n" + "="*60)
print("8. STRING OPERATIONS")
print("="*60)

# String methods
df_str = df.copy()
print("Original names:")
print(df_str['Name'])

# String transformations
df_str['Name_Upper'] = df_str['Name'].str.upper()
df_str['Name_Lower'] = df_str['Name'].str.lower()
df_str['Name_Length'] = df_str['Name'].str.len()

print(f"\nString transformations:")
print(df_str[['Name', 'Name_Upper', 'Name_Lower', 'Name_Length']])

# String filtering
names_with_a = df_str[df_str['Name'].str.contains('a', case=False)]
print(f"\nNames containing 'a' (case insensitive):")
print(names_with_a['Name'])

# =============================================================================
# 9. FILE I/O OPERATIONS
# =============================================================================

print("\n" + "="*60)
print("9. FILE I/O OPERATIONS")
print("="*60)

# Save to CSV
df.to_csv('employee_data.csv', index=False)
print("✅ Data saved to 'employee_data.csv'")

# Read from CSV
df_from_csv = pd.read_csv('employee_data.csv')
print("Data read from CSV:")
print(df_from_csv.head())

# Save to Excel (requires openpyxl: pip install openpyxl)
try:
    df.to_excel('employee_data.xlsx', index=False, sheet_name='Employees')
    print("✅ Data saved to 'employee_data.xlsx'")
except ImportError:
    print("⚠️  Excel export requires openpyxl: pip install openpyxl")

# Save to JSON
df.to_json('employee_data.json', orient='records', indent=2)
print("✅ Data saved to 'employee_data.json'")

# =============================================================================
# 10. PRACTICAL EXAMPLE: SALES ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("10. PRACTICAL EXAMPLE: COMPREHENSIVE SALES ANALYSIS")
print("="*60)

# Create comprehensive sales dataset
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=365, freq='D')
products = ['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones']
regions = ['North', 'South', 'East', 'West']

# Generate sales data
sales_records = []
for _ in range(1000):
    record = {
        'Date': np.random.choice(dates),
        'Product': np.random.choice(products),
        'Region': np.random.choice(regions),
        'Quantity': np.random.randint(1, 10),
        'Unit_Price': np.random.randint(100, 1000),
        'Customer_Age': np.random.randint(18, 70),
        'Customer_Type': np.random.choice(['New', 'Returning'])
    }
    record['Total_Sales'] = record['Quantity'] * record['Unit_Price']
    sales_records.append(record)

sales_df = pd.DataFrame(sales_records)
sales_df['Date'] = pd.to_datetime(sales_df['Date'])
sales_df['Month'] = sales_df['Date'].dt.month
sales_df['Quarter'] = sales_df['Date'].dt.quarter

print("Sales dataset sample:")
print(sales_df.head())
print(f"\nDataset shape: {sales_df.shape}")

# Analysis 1: Top products by total sales
print(f"\n1. Top Products by Total Sales:")
product_sales = sales_df.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False)
print(product_sales)

# Analysis 2: Regional performance
print(f"\n2. Regional Performance:")
regional_stats = sales_df.groupby('Region').agg({
    'Total_Sales': ['sum', 'mean', 'count'],
    'Quantity': 'sum'
}).round(2)
print(regional_stats)

# Analysis 3: Monthly trends
print(f"\n3. Monthly Sales Trends:")
monthly_sales = sales_df.groupby('Month')['Total_Sales'].sum()
print(monthly_sales)

# Analysis 4: Customer analysis
print(f"\n4. Customer Analysis:")
customer_analysis = sales_df.groupby('Customer_Type').agg({
    'Total_Sales': ['sum', 'mean'],
    'Quantity': 'sum',
    'Customer_Age': 'mean'
}).round(2)
print(customer_analysis)

# Analysis 5: Product performance by region
print(f"\n5. Product Performance by Region:")
product_region = sales_df.pivot_table(
    values='Total_Sales', 
    index='Product', 
    columns='Region', 
    aggfunc='sum',
    fill_value=0
)
print(product_region)

# Analysis 6: Top 10 sales days
print(f"\n6. Top 10 Sales Days:")
daily_sales = sales_df.groupby('Date')['Total_Sales'].sum().sort_values(ascending=False)
print(daily_sales.head(10))

print("\n" + "="*70)
print("PANDAS SUMMARY")
print("="*70)
print("✅ DataFrames provide powerful 2D data structures")
print("✅ Easy data selection, filtering, and manipulation")
print("✅ Built-in statistical and aggregation functions")
print("✅ Excellent file I/O capabilities (CSV, Excel, JSON, SQL)")
print("✅ Powerful grouping and pivot table functionality")
print("✅ Robust missing data handling")
print("✅ String and datetime operations")
print("\nNext: Move to file 04 - Data Visualization with Matplotlib!")
print("="*70)