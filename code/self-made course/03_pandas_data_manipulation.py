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
"""

import pandas as pd
import numpy as np

# Pandas version check
pd.__version__

# =============================================================================
# 1. CREATING DATAFRAMES AND SERIES
# =============================================================================

'''
Key Data Structures:
- Series: 1D labeled array
- DataFrame: 2D labeled data structure (like Excel spreadsheet)
'''

# Creating a Series
series_data = pd.Series([10, 20, 30, 40, 50],
                        index=['A', 'B', 'C', 'D', 'E'])
# Series: A 10, B 20, C 30, D 40, E 50
# Series type: <class 'pandas.core.series.Series'>

# Creating DataFrame from dictionary
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['New York', 'Boston', 'Chicago', 'Denver', 'Seattle'],
    'Salary': [50000, 60000, 55000, 52000, 58000],
    'Department': ['IT', 'Finance', 'IT', 'HR', 'Finance']
}

df = pd.DataFrame(data_dict)
# DataFrame from dictionary: 5 rows × 5 columns
# DataFrame shape: (5, 5)
# DataFrame columns: ['Name', 'Age', 'City', 'Salary', 'Department']
# DataFrame index: [0, 1, 2, 3, 4] - returns row labels of the dictionary data frame (0 to n-1)
# note: you cannot hash (change the indices) for a dictionary


# Creating DataFrame from lists
data_lists = [
    ['Frank', 29, 'Miami', 54000, 'Marketing'],
    ['Grace', 31, 'Austin', 56000, 'IT'],
    ['Henry', 27, 'Portland', 51000, 'HR']
]
df_from_lists = pd.DataFrame(data_lists,
                             columns=['Name', 'Age', 'City', 'Salary', 'Department'])
# DataFrame from lists: 3 rows × 5 columns (Frank, Grace, Henry)

# =============================================================================
# 2. BASIC DATAFRAME OPERATIONS
# =============================================================================

# Basic info about the DataFrame
# Shape: (5, 5), Size: 25
df.shape
df.size

# Data types: Name(object), Age(int64), City(object), Salary(int64), Department(object)
df.dtypes

# First 3 rows
df.head(3)

# Last 2 rows
df.tail(2)

# Statistical summary
df.describe()

# DataFrame info
df.info()

# =============================================================================
# 3. SELECTING AND INDEXING DATA
# =============================================================================

# Selecting columns
single_column = df['Name']  # returns pandas Series
multiple_columns = df[['Name', 'Salary']]  # returns DataFrame
# Single column (Series): returns pandas Series
# Multiple columns (DataFrame): returns DataFrame

# Selecting rows by index
# First row by position (iloc)
df.iloc[0]

# First row by label (loc)
df.loc[0]

# Selecting specific rows and columns
# Rows 0-2, specific columns
df.loc[0:2, ['Name', 'Age']]

# First 3 rows, first 2 columns
df.iloc[0:3, 0:2]

# Boolean indexing (filtering)
high_salary = df[df['Salary'] > 55000]  # Employees with salary > 55000
it_high_salary = df[(df['Department'] == 'IT') & (
    df['Salary'] > 50000)]  # IT dept with salary > 50000
# High salary employees: filtered DataFrame
# IT employees with high salary: filtered DataFrame

# =============================================================================
# 4. DATA MANIPULATION
# =============================================================================

# Adding new columns
df_copy = df.copy()  # Work with a copy
df_copy['Salary_K'] = df_copy['Salary'] / 1000  # Salary in thousands
df_copy['Age_Group'] = df_copy['Age'].apply(
    lambda x: 'Young' if x < 30 else 'Senior')
# DataFrame with new columns: includes Salary_K and Age_Group

# Modifying existing data
df_copy.loc[df_copy['City'] == 'New York', 'City'] = 'NYC'
# After modifying NYC: New York changed to NYC

# Dropping columns and rows
df_dropped = df_copy.drop(['Salary_K'], axis=1)  # Drop column
df_dropped_rows = df_copy.drop([0, 1])  # Drop rows by index
# After dropping column: Salary_K column removed
# After dropping rows: first 2 rows removed

# =============================================================================
# 5. SORTING AND RANKING
# =============================================================================

# Sort by single column
sorted_by_age = df.sort_values('Age')
# Sorted by age: DataFrame sorted by Age column

# Sort by multiple columns
sorted_multi = df.sort_values(
    ['Department', 'Salary'], ascending=[True, False])
# Sorted by department and salary: Department ascending, Salary descending

# Ranking
df_with_rank = df.copy()
df_with_rank['Salary_Rank'] = df['Salary'].rank(ascending=False)
# DataFrame with salary ranking: includes Salary_Rank column

# =============================================================================
# 6. GROUPING AND AGGREGATION
# =============================================================================

# Group by single column
dept_groups = df.groupby('Department')
avg_salary_by_dept = dept_groups['Salary'].mean()
# Average salary by department: Finance, HR, IT departments

# Multiple aggregations by department
dept_agg = dept_groups.agg({
    'Salary': ['mean', 'min', 'max', 'count'],
    'Age': ['mean', 'min', 'max']
})
# Department aggregations: multiple statistics per department

# Group by multiple columns (add more data first)
extended_data = pd.concat([df, df_from_lists], ignore_index=True)
city_dept_groups = extended_data.groupby(['Department', 'City'])
avg_salary_by_city_dept = city_dept_groups['Salary'].mean()
# Average salary by Department and City: grouped by both dimensions

# =============================================================================
# 7. HANDLING MISSING DATA
# =============================================================================

# Create data with missing values
data_with_na = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, np.nan, 35, 28, 32],
    'Salary': [50000, 60000, np.nan, 52000, 58000],
    'City': ['New York', 'Boston', 'Chicago', None, 'Seattle']
}

df_na = pd.DataFrame(data_with_na)
# DataFrame with missing values: contains NaN and None values

# Check for missing values
missing_counts = df_na.isnull().sum()
rows_with_missing = df_na[df_na.isnull().any(axis=1)]
# Missing values per column: count of NaN/None per column
# Rows with any missing values: rows containing any missing data

# Handle missing values
# Fill with specific values
df_filled = df_na.fillna({
    'Age': df_na['Age'].mean(),
    'Salary': df_na['Salary'].median(),
    'City': 'Unknown'
})
# DataFrame after filling missing values: NaN replaced with mean/median/default

# Drop rows with missing values
df_dropped_na = df_na.dropna()
# DataFrame after dropping missing values: rows with NaN removed

# =============================================================================
# 8. STRING OPERATIONS
# =============================================================================

# String methods
df_str = df.copy()

# String transformations
df_str['Name_Upper'] = df_str['Name'].str.upper()
df_str['Name_Lower'] = df_str['Name'].str.lower()
df_str['Name_Length'] = df_str['Name'].str.len()
# DataFrame with string operations: includes upper, lower, length columns

# String filtering
names_with_a = df_str[df_str['Name'].str.contains(
    'a', case=False)]  # Names containing 'a'
# Names containing 'a': filtered DataFrame with names containing letter 'a'

# =============================================================================
# 9. FILE I/O OPERATIONS
# =============================================================================

# Save to CSV
df.to_csv('employee_data.csv', index=False)
# Data saved to employee_data.csv

# Read from CSV
df_from_csv = pd.read_csv('employee_data.csv')
# Data read from CSV: DataFrame loaded from CSV file

# Save to Excel (requires openpyxl: pip install openpyxl)
try:
    df.to_excel('employee_data.xlsx', index=False, sheet_name='Employees')
    # Data saved to employee_data.xlsx
except ImportError:
    # Excel export requires openpyxl: pip install openpyxl
    pass

# Save to JSON
df.to_json('employee_data.json', orient='records', indent=2)
# Data saved to employee_data.json

# =============================================================================
# 10. PRACTICAL EXAMPLE: SALES ANALYSIS
# =============================================================================

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

# Sales dataset shape: (1000, 9) - 1000 sales records with 9 columns
# Sales dataset sample: first 5 rows of sales data

# Analysis 1: Top products by total sales
product_sales = sales_df.groupby(
    'Product')['Total_Sales'].sum().sort_values(ascending=False)
# Top products by total sales: products ranked by revenue

# Analysis 2: Regional performance
regional_stats = sales_df.groupby('Region').agg({
    'Total_Sales': ['sum', 'mean', 'count'],
    'Quantity': 'sum'
}).round(2)
# Regional performance: sales statistics by region

# Analysis 3: Monthly trends
monthly_sales = sales_df.groupby('Month')['Total_Sales'].sum()
# Monthly sales trends: total sales per month

# Analysis 4: Customer analysis
customer_analysis = sales_df.groupby('Customer_Type').agg({
    'Total_Sales': ['sum', 'mean'],
    'Quantity': 'sum',
    'Customer_Age': 'mean'
}).round(2)
# Customer analysis: new vs returning customer metrics

# Analysis 5: Product performance by region
product_region = sales_df.pivot_table(
    values='Total_Sales',
    index='Product',
    columns='Region',
    aggfunc='sum',
    fill_value=0
)
# Product performance by region: cross-tabulation of products and regions

# Analysis 6: Top 10 sales days
daily_sales = sales_df.groupby(
    'Date')['Total_Sales'].sum().sort_values(ascending=False)
top_sales_days = daily_sales.head(10)
# Top 10 sales days: highest revenue days
