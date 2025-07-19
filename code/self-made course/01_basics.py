# Python Basics
# This file contains fundamental Python concepts you'll need for data analysis

# =============================================================================
# 1. VARIABLES AND DATA TYPES
# =============================================================================

# Numbers
age = 25
price = 19.99
temperature = -5.2

# Strings
name = "Alice"
city = 'New York'

# Booleans
is_student = True
has_discount = False

# Lists (ordered, mutable)
numbers = [1, 2, 3, 4, 5]
fruits = ["apple", "banana", "orange"]
mixed = [1, "hello", True, 3.14]

# Dictionaries (key-value pairs)
person = {
    "name": "John",
    "age": 30,
    "city": "Boston"
}

# Variables and Data Types
# Age: 25, Price: 19.99
# Name: Alice, City: New York
# Numbers: [1, 2, 3, 4, 5]
# Person: {'name': 'John', 'age': 30, 'city': 'Boston'}

# =============================================================================
# 2. BASIC OPERATIONS
# =============================================================================

# Arithmetic
result_add = 10 + 5    # 15
result_sub = 10 - 3    # 7
result_mul = 4 * 6     # 24
result_div = 15 / 3    # 5.0
result_floor = 17 // 5 # 3 (floor division)
result_mod = 17 % 5    # 2 (remainder)

# Arithmetic results: 15, 7, 24, 5.0, 3, 2

# String operations
greeting = "Hello" + " " + "World"  # "Hello World"
repeated = "Python " * 3            # "Python Python Python "
# Greeting: Hello World
# Repeated: Python Python Python 

# List operations
numbers.append(6)           # Add to end
numbers.insert(0, 0)        # Insert at position
first_item = numbers[0]     # Access by index
last_item = numbers[-1]     # Negative indexing
# Modified numbers: [0, 1, 2, 3, 4, 5, 6]
# First item: 0, Last item: 6

# =============================================================================
# 3. CONTROL FLOW
# =============================================================================

# If statements
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

# Score: 85, Grade: B

# For loops
# Fruits I like: apple, banana, orange
for fruit in fruits:
    pass  # Loop through fruits

# Counting 0 to 4
for i in range(5):  # 0, 1, 2, 3, 4
    pass  # Count from 0 to 4

# While loops
# While loop example: count from 0 to 2
count = 0
while count < 3:
    count += 1

# =============================================================================
# 4. FUNCTIONS
# =============================================================================

def greet(name):
    """Function to greet someone"""
    return f"Hello, {name}!"

def calculate_average(numbers):
    """Calculate the average of a list of numbers"""
    if len(numbers) == 0:
        return 0
    return sum(numbers) / len(numbers)

def analyze_data(data_list):
    """Analyze a list of numbers and return statistics"""
    if not data_list:
        return {"error": "Empty list"}
    
    return {
        "count": len(data_list),
        "sum": sum(data_list),
        "average": sum(data_list) / len(data_list),
        "min": min(data_list),
        "max": max(data_list)
    }

# Using functions
message = greet("Alice")
avg = calculate_average([10, 20, 30, 40])
# Greeting: Hello, Alice!
# Average: 25.0

# Analyze some data
sample_data = [1, 5, 3, 9, 2, 8, 4]
stats = analyze_data(sample_data)
# Data analysis: {'count': 7, 'sum': 32, 'average': 4.57, 'min': 1, 'max': 9}

# =============================================================================
# 5. WORKING WITH LISTS (Important for Data Analysis)
# =============================================================================

data = [1, 5, 3, 9, 2, 8, 4]
# Original data: [1, 5, 3, 9, 2, 8, 4]

# List comprehensions (very useful!)
squared = [x**2 for x in data]           # [1, 25, 9, 81, 4, 64, 16]
evens = [x for x in data if x % 2 == 0]  # [2, 8, 4]
# Squared: [1, 25, 9, 81, 4, 64, 16]
# Even numbers: [2, 8, 4]

# Built-in functions
total = sum(data)        # 32
maximum = max(data)      # 9
minimum = min(data)      # 1
length = len(data)       # 7
# Sum: 32, Max: 9, Min: 1, Length: 7

# Sorting
sorted_data = sorted(data)  # [1, 2, 3, 4, 5, 8, 9]
# Sorted data: [1, 2, 3, 4, 5, 8, 9]

# =============================================================================
# 6. WORKING WITH DICTIONARIES
# =============================================================================
# These look like they are 1:1 aliases?

sales_data = {
    "January": 1000,
    "February": 1200,
    "March": 900
}

# Original sales data: {'January': 1000, 'February': 1200, 'March': 900}

# Accessing values
jan_sales = sales_data["January"]
feb_sales = sales_data.get("February", 0)  # Safer way
# January sales: 1000
# February sales: 1200

# Adding/updating
sales_data["April"] = 1100
sales_data.update({"May": 1300, "June": 1150})
# Updated sales data includes April, May, June

# Iterating
# Monthly sales: iterate through all months and values
for month, sales in sales_data.items():
    pass  # Process each month's sales

# =============================================================================
# 7. FILE HANDLING 
# =============================================================================


# Writing a sample data file
sample_csv_data = """Name,Age,City,Salary
Alice,25,New York,50000
Bob,30,Boston,60000
Charlie,35,Chicago,55000
Diana,28,Denver,52000"""

# Write sample data to file
with open("sample_data.csv", "w") as file:
    file.write(sample_csv_data)
# Created sample_data.csv

# Reading the file back
with open("sample_data.csv", "r") as file:
    content = file.read()
    # File content: CSV data with headers and 4 rows

# Reading line by line
with open("sample_data.csv", "r") as file:
    lines = file.readlines()
    for i, line in enumerate(lines):
        pass  # Process each line

# =============================================================================
# 8. ERROR HANDLING
# =============================================================================

def safe_divide(a, b):
    """Safely divide two numbers with error handling"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    except TypeError:
        print("Error: Invalid input types!")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Test error handling
result1 = safe_divide(10, 2)    # Returns 5.0
result2 = safe_divide(10, 0)    # Returns None, prints error
result3 = safe_divide('10', 2)  # Returns None, prints error

# =============================================================================
# 9. PRACTICAL DATA ANALYSIS EXAMPLE
# =============================================================================

# Sample dataset: student grades
students_data = [
    {"name": "Alice", "math": 85, "science": 92, "english": 78},
    {"name": "Bob", "math": 78, "science": 85, "english": 88},
    {"name": "Charlie", "math": 92, "science": 89, "english": 94},
    {"name": "Diana", "math": 88, "science": 91, "english": 85},
    {"name": "Eve", "math": 76, "science": 83, "english": 90}
]

def analyze_student_data(data):
    """Analyze student performance data"""
    # Calculate averages for each student
    for student in data:
        grades = [student["math"], student["science"], student["english"]]
        student["average"] = sum(grades) / len(grades)
    
    # Calculate subject averages
    math_scores = [s["math"] for s in data]
    science_scores = [s["science"] for s in data]
    english_scores = [s["english"] for s in data]
    
    subject_averages = {
        "math": sum(math_scores) / len(math_scores),
        "science": sum(science_scores) / len(science_scores),
        "english": sum(english_scores) / len(english_scores)
    }
    
    return data, subject_averages

# Analyze the data
analyzed_data, subject_avgs = analyze_student_data(students_data)

# Student Performance Analysis
# Display each student's grades and average
for student in analyzed_data:
    pass  # Show student performance data

# Subject Averages
# Display average for each subject
for subject, avg in subject_avgs.items():
    pass  # Show subject averages

# Find top performer
top_student = max(analyzed_data, key=lambda x: x["average"])
# lambda is a Python keyword that creates anonymous functions.
# lambda x: x["average"] creates an anonymous function that takes each student record (x) and returns their "average" field
# equivalent to...
# def get_average(x):
#     return x["average"]
# But lambda allows for this to be writing inline without defining a separate function
# So instead of comparing the entire dictionary objects, max() compares just the average values
# Top performer: student with highest average

# =============================================================================
# 9. Managing Environment Variables
# =============================================================================

# shows all variable names in current scope
all_variables = dir()

# returns dictionary of all local variables
local_vars = locals()
