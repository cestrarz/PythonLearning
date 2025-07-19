# 02 - NumPy Fundamentals
# NumPy is the foundation of the Python data science ecosystem

"""
NUMPY (Numerical Python)
========================

NumPy provides:
- Fast, efficient arrays (ndarray)
- Mathematical functions
- Broadcasting capabilities
- Foundation for other libraries (Pandas, Scikit-learn, etc.)

Why NumPy?
- 10-100x faster than pure Python lists for numerical operations
- Memory efficient
- Vectorized operations (no explicit loops needed)
"""

# change test for git

import numpy as np
# NumPy version check
np.__version__

# =============================================================================
# 1. CREATING ARRAYS
# =============================================================================

# From Python lists
list_1d = [1, 2, 3, 4, 5]
array_1d = np.array(list_1d)
# 1D array: [1 2 3 4 5]
# Type: <class 'numpy.ndarray'>
# Data type: int64

# 2D array (matrix)
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# you don't have to first create another list object first!
# 2D array: 3x3 matrix
# Shape: (3, 3)
# Dimensions: 2

# Common array creation functions
zeros = np.zeros((3, 4))  # 3x4 array of zeros
ones = np.ones((2, 3))    # 2x3 array of ones
empty = np.empty((2, 2))  # Uninitialized array
full = np.full((2, 3), 7)  # Array filled with 7

# Zeros array: 3x4 array of zeros
# Ones array: 2x3 array of ones
# Empty array: 2x2 uninitialized array
# Full array (7s): 2x3 array filled with 7

# Range arrays
range_array = np.arange(0, 10, 2)  # Start, stop, step
linspace_array = np.linspace(0, 1, 5)  # Start, stop, num_points

# Range array (0 to 10, step 2): [0 2 4 6 8]
# Linspace array (0 to 1, 5 points): [0. 0.25 0.5 0.75 1.]

# Random arrays
np.random.seed(42)  # For reproducible results
random_array = np.random.random((2, 3))  # Random floats 0-1
random_int = np.random.randint(1, 10, size=(2, 3))  # Random integers between 1 and 10 -- does NOT include 10, but includes 1

# Random floats: 2x3 array of random floats between 0-1
# Random integers: 2x3 array of random integers between 1-9

# =============================================================================
# 2. ARRAY PROPERTIES AND INDEXING
# =============================================================================

data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# Sample array: 3x4 matrix with values 1-12
# Shape: (3, 4)
# Size (total elements): 12
# Data type: int64
# Memory usage: 96 bytes

# Indexing (similar to Python lists but more powerful)
first_element = data[0, 0]      # First element: 1
last_element = data[-1, -1]     # Last element: 12
first_row = data[0]             # First row: [1 2 3 4]
first_column = data[:, 0]       # First column: [1 5 9]
subarray = data[:2, :2]         # Subarray (first 2 rows, first 2 cols): 2x2

# Boolean indexing (very powerful!)
mask = data > 6
# Boolean mask (elements > 6): True/False array
elements_gt_6 = data[mask]      # Elements > 6: [7 8 9 10 11 12]

# Fancy indexing
indices = [0, 2]  # Select rows 0 and 2
selected_rows = data[indices]   # Rows 0 and 2: first and third rows

# =============================================================================
# 3. ARRAY OPERATIONS
# =============================================================================

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Array a: [1 2 3 4]
# Array b: [5 6 7 8]

# Element-wise operations
addition = a + b        # Addition: [6 8 10 12]
subtraction = a - b     # Subtraction: [-4 -4 -4 -4]
multiplication = a * b  # Multiplication: [5 12 21 32]
division = a / b        # Division: [0.2 0.33 0.43 0.5]
power = a ** 2          # Power: [1 4 9 16]

# Operations with scalars (broadcasting)
# Scalar operations
scalar_add = a + 10     # a + 10 = [11 12 13 14]
scalar_mult = a * 2     # a * 2 = [2 4 6 8]

# Mathematical functions
# Math functions
sqrt_a = np.sqrt(a)     # Square root: [1. 1.41 1.73 2.]
exp_a = np.exp(a)       # Exponential: [2.72 7.39 20.09 54.6]
log_a = np.log(a)       # Logarithm: [0. 0.69 1.1 1.39]
sin_a = np.sin(a)       # Sine: [0.84 0.91 0.14 -0.76]

# =============================================================================
# 4. STATISTICAL OPERATIONS
# =============================================================================

# Sample data for statistics
np.random.seed(42)
# Normal distribution, mean=100, std=15, 1000 observations
sample_data = np.random.normal(100, 15, 1000)

# Sample data (first 10): array of 10 random values around 100
# Basic statistics
mean_val = np.mean(sample_data)         # Mean: ~100.0
median_val = np.median(sample_data)     # Median: ~100.0
std_val = np.std(sample_data)           # Standard deviation: ~15.0
var_val = np.var(sample_data)           # Variance: ~225.0
min_val = np.min(sample_data)           # Min: ~50-60
max_val = np.max(sample_data)           # Max: ~140-150

# Percentiles
percentile_25 = np.percentile(sample_data, 25)  # 25th percentile: ~90
percentile_75 = np.percentile(sample_data, 75)  # 75th percentile: ~110

# 2D array statistics
matrix = np.random.randint(1, 10, (3, 4))
# Matrix for axis operations: 3x4 matrix with random integers 1-9
sum_all = np.sum(matrix)                # Sum of all elements
sum_cols = np.sum(matrix, axis=0)       # Sum along axis 0 (columns) - its so weird that columns are axis 0
sum_rows = np.sum(matrix, axis=1)       # Sum along axis 1 (rows) - think of axis as the dimension that gets collapsed!

# =============================================================================
# 5. ARRAY MANIPULATION
# =============================================================================

original = np.arange(12)
# Original array: [0 1 2 3 4 5 6 7 8 9 10 11]

# Reshaping
reshaped = original.reshape(3, 4)
# Reshaped to 3x4: 3x4 matrix with values 0-11

# Flattening
flattened = reshaped.flatten()
# Flattened: [0 1 2 3 4 5 6 7 8 9 10 11]

# Transposing
transposed = reshaped.T
# Transposed: 4x3 matrix (transposed version)

# Concatenation
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated = np.concatenate([arr1, arr2])
# Concatenated arrays: [1 2 3 4 5 6]

# Stacking
stacked_v = np.vstack([arr1, arr2])  # Vertical stack
stacked_h = np.hstack([arr1, arr2])  # Horizontal stack
# Vertical stack: 2x3 matrix (rows stacked)
# Horizontal stack: [1 2 3 4 5 6]

# Splitting
split_arrays = np.split(concatenated, 2)
# Split array: [array([1, 2, 3]), array([4, 5, 6])]

# =============================================================================
# 6. BROADCASTING
# =============================================================================

# Broadcasting allows operations between arrays of different shapes
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector = np.array([10, 20, 30])

# Matrix: 3x3 matrix
# Vector: [10 20 30]

# Add vector to each row of matrix
result = matrix + vector
# Matrix + vector (broadcast): vector added to each row of the matrix

# Broadcasting with different shapes
col_vector = np.array([[1], [2], [3]])
# Column vector: 3x1 column vector
result2 = matrix + col_vector
# Matrix + column vector: vector added to each column of the matrix

# Horizontal + Vertical vector operation
what_happens = vector + col_vector
# Horizontal + Vertical vector operation: element i,j of the resulting matrix is the sum of elements i and j of vector and col_vector

# =============================================================================
# 7. PRACTICAL DATA ANALYSIS EXAMPLE
# =============================================================================

# Simulate monthly sales data for 4 products over 12 months
np.random.seed(42)
products = ['Product A', 'Product B', 'Product C', 'Product D']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Generate sales data (4 products x 12 months)
sales_data = np.random.randint(1000, 5000, (4, 12))

print("Sales Data (Products x Months):")
print(f"{'Product':<12}", end='')
for month in months:
    print(f"{month:>6}", end='')
print()

for i, product in enumerate(products):
    print(f"{product:<12}", end='')
    for j in range(12):
        print(f"{sales_data[i, j]:>6}", end='')
    print()

# Analysis
print(f"\nSales Analysis:")
print("-" * 50)

# Total sales per product
product_totals = np.sum(sales_data, axis=1)
for i, product in enumerate(products):
    print(f"{product}: ${product_totals[i]:,}")

# Total sales per month
monthly_totals = np.sum(sales_data, axis=0)
print(f"\nMonthly totals:")
for i, month in enumerate(months):
    print(f"{month}: ${monthly_totals[i]:,}")

# Best and worst performing
best_product_idx = np.argmax(product_totals)
worst_product_idx = np.argmin(product_totals)
best_month_idx = np.argmax(monthly_totals)
worst_month_idx = np.argmin(monthly_totals)

print(f"\nPerformance Summary:")
print(
    f"Best product: {products[best_product_idx]} (${product_totals[best_product_idx]:,})")
print(
    f"Worst product: {products[worst_product_idx]} (${product_totals[worst_product_idx]:,})")
print(
    f"Best month: {months[best_month_idx]} (${monthly_totals[best_month_idx]:,})")
print(
    f"Worst month: {months[worst_month_idx]} (${monthly_totals[worst_month_idx]:,})")

# Statistics
print(f"\nOverall Statistics:")
print(f"Total sales: ${np.sum(sales_data):,}")
print(f"Average monthly sales: ${np.mean(monthly_totals):,.0f}")
print(f"Standard deviation: ${np.std(sales_data):,.0f}")
