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

import numpy as np
print(f"NumPy version: {np.__version__}")

# =============================================================================
# 1. CREATING ARRAYS
# =============================================================================

# From Python lists
list_1d = [1, 2, 3, 4, 5]
array_1d = np.array(list_1d)
print(f"1D array: {array_1d}")
print(f"Type: {type(array_1d)}")
print(f"Data type: {array_1d.dtype}")

# 2D array (matrix)
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# you don't have to first create another list object first!
print(f"\n2D array:\n{array_2d}")
print(f"Shape: {array_2d.shape}")
print(f"Dimensions: {array_2d.ndim}")

# Common array creation functions
zeros = np.zeros((3, 4))  # 3x4 array of zeros
ones = np.ones((2, 3))    # 2x3 array of ones
empty = np.empty((2, 2))  # Uninitialized array
full = np.full((2, 3), 7)  # Array filled with 7

print(f"\nZeros array:\n{zeros}")
print(f"\nOnes array:\n{ones}")
print(f"\nFull array (7s):\n{full}")

# Range arrays
range_array = np.arange(0, 10, 2)  # Start, stop, step
linspace_array = np.linspace(0, 1, 5)  # Start, stop, num_points

print(f"\nRange array (0 to 10, step 2): {range_array}")
print(f"Linspace array (0 to 1, 5 points): {linspace_array}")

# Random arrays
np.random.seed(42)  # For reproducible results
random_array = np.random.random((2, 3))  # Random floats 0-1
random_int = np.random.randint(1, 10, size=(2, 3))  # Random integers

print(f"\nRandom floats:\n{random_array}")
print(f"\nRandom integers:\n{random_int}")

# =============================================================================
# 2. ARRAY PROPERTIES AND INDEXING
# =============================================================================

print("\n" + "="*50)
print("2. ARRAY PROPERTIES AND INDEXING")
print("="*50)

data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Sample array:\n{data}")
print(f"Shape: {data.shape}")
print(f"Size (total elements): {data.size}")
print(f"Data type: {data.dtype}")
print(f"Memory usage: {data.nbytes} bytes")

# Indexing (similar to Python lists but more powerful)
print(f"\nFirst element: {data[0, 0]}")
print(f"Last element: {data[-1, -1]}")
print(f"First row: {data[0]}")
print(f"First column: {data[:, 0]}")
print(f"Subarray (first 2 rows, first 2 cols):\n{data[:2, :2]}")

# Boolean indexing (very powerful!)
mask = data > 6
print(f"\nBoolean mask (elements > 6):\n{mask}")
print(f"Elements > 6: {data[mask]}")

# Fancy indexing
indices = [0, 2]  # Select rows 0 and 2
print(f"\nRows 0 and 2:\n{data[indices]}")

# =============================================================================
# 3. ARRAY OPERATIONS
# =============================================================================

print("\n" + "="*50)
print("3. ARRAY OPERATIONS")
print("="*50)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"Array a: {a}")
print(f"Array b: {b}")

# Element-wise operations
print(f"\nAddition: a + b = {a + b}")
print(f"Subtraction: a - b = {a - b}")
print(f"Multiplication: a * b = {a * b}")
print(f"Division: a / b = {a / b}")
print(f"Power: a ** 2 = {a ** 2}")

# Operations with scalars (broadcasting)
print(f"\nScalar operations:")
print(f"a + 10 = {a + 10}")
print(f"a * 2 = {a * 2}")

# Mathematical functions
print(f"\nMath functions:")
print(f"Square root: np.sqrt(a) = {np.sqrt(a)}")
print(f"Exponential: np.exp(a) = {np.exp(a)}")
print(f"Logarithm: np.log(a) = {np.log(a)}")
print(f"Sine: np.sin(a) = {np.sin(a)}")

# =============================================================================
# 4. STATISTICAL OPERATIONS
# =============================================================================

print("\n" + "="*50)
print("4. STATISTICAL OPERATIONS")
print("="*50)

# Sample data for statistics
np.random.seed(42)
# Normal distribution, mean=100, std=15
sample_data = np.random.normal(100, 15, 1000)

print(f"Sample data (first 10): {sample_data[:10]}")
print(f"\nBasic statistics:")
print(f"Mean: {np.mean(sample_data):.2f}")
print(f"Median: {np.median(sample_data):.2f}")
print(f"Standard deviation: {np.std(sample_data):.2f}")
print(f"Variance: {np.var(sample_data):.2f}")
print(f"Min: {np.min(sample_data):.2f}")
print(f"Max: {np.max(sample_data):.2f}")

# Percentiles
print(f"\nPercentiles:")
print(f"25th percentile: {np.percentile(sample_data, 25):.2f}")
print(f"75th percentile: {np.percentile(sample_data, 75):.2f}")

# 2D array statistics
matrix = np.random.randint(1, 10, (3, 4))
print(f"\nMatrix for axis operations:\n{matrix}")
print(f"Sum of all elements: {np.sum(matrix)}")
print(f"Sum along axis 0 (columns): {np.sum(matrix, axis=0)}")
print(f"Sum along axis 1 (rows): {np.sum(matrix, axis=1)}")

# =============================================================================
# 5. ARRAY MANIPULATION
# =============================================================================

print("\n" + "="*50)
print("5. ARRAY MANIPULATION")
print("="*50)

original = np.arange(12)
print(f"Original array: {original}")

# Reshaping
reshaped = original.reshape(3, 4)
print(f"Reshaped to 3x4:\n{reshaped}")

# Flattening
flattened = reshaped.flatten()
print(f"Flattened: {flattened}")

# Transposing
transposed = reshaped.T
print(f"Transposed:\n{transposed}")

# Concatenation
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated = np.concatenate([arr1, arr2])
print(f"\nConcatenated arrays: {concatenated}")

# Stacking
stacked_v = np.vstack([arr1, arr2])  # Vertical stack
stacked_h = np.hstack([arr1, arr2])  # Horizontal stack
print(f"Vertical stack:\n{stacked_v}")
print(f"Horizontal stack: {stacked_h}")

# Splitting
split_arrays = np.split(concatenated, 2)
print(f"Split array: {split_arrays}")

# =============================================================================
# 6. BROADCASTING
# =============================================================================

print("\n" + "="*50)
print("6. BROADCASTING")
print("="*50)

# Broadcasting allows operations between arrays of different shapes
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector = np.array([10, 20, 30])

print(f"Matrix:\n{matrix}")
print(f"Vector: {vector}")

# Add vector to each row of matrix
result = matrix + vector
print(f"Matrix + vector (broadcast):\n{result}")

# Broadcasting with different shapes
col_vector = np.array([[1], [2], [3]])
print(f"\nColumn vector:\n{col_vector}")
result2 = matrix + col_vector
print(f"Matrix + column vector:\n{result2}")

# =============================================================================
# 7. PRACTICAL DATA ANALYSIS EXAMPLE
# =============================================================================

print("\n" + "="*50)
print("7. PRACTICAL EXAMPLE: SALES DATA ANALYSIS")
print("="*50)

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

print("\n" + "="*70)
print("NUMPY SUMMARY")
print("="*70)
print("✅ Arrays are faster and more memory efficient than Python lists")
print("✅ Vectorized operations eliminate the need for explicit loops")
print("✅ Broadcasting allows operations between different shaped arrays")
print("✅ Rich set of mathematical and statistical functions")
print("✅ Foundation for Pandas, Scikit-learn, and other libraries")
print("\nNext: Move to file 03 - Pandas for data manipulation!")
print("="*70)
