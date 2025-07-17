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

print("=== Variables and Data Types ===")
print(f"Age: {age}, Price: {price}")
print(f"Name: {name}, City: {city}")
print(f"Numbers: {numbers}")
print(f"Person: {person}")
print()

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

print(f"10 + 5 = {result_add}")
print(f"10 - 3 = {result_sub}")
print(f"4 * 6 = {result_mul}")
print(f"15 / 3 = {result_div}")
print(f"17 // 5 = {result_floor}")
print(f"17 % 5 = {result_mod}")

# String operations
greeting = "Hello" + " " + "World"  # "Hello World"
repeated = "Python " * 3            # "Python Python Python "
print(f"Greeting: {greeting}")
print(f"Repeated: {repeated}")

# List operations
numbers.append(6)           # Add to end
numbers.insert(0, 0)        # Insert at position
first_item = numbers[0]     # Access by index
last_item = numbers[-1]     # Negative indexing
print(f"Modified numbers: {numbers}")
print(f"First item: {first_item}, Last item: {last_item}")
print()

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

print(f"Score: {score}, Grade: {grade}")

# For loops
print("Fruits I like:")
for fruit in fruits:
    print(f"  - {fruit}")

print("Counting 0 to 4:")
for i in range(5):  # 0, 1, 2, 3, 4
    print(f"  {i}")

# While loops
print("While loop example:")
count = 0
while count < 3:
    print(f"  Count: {count}")
    count += 1
print()

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
print(f"Greeting: {message}")
print(f"Average: {avg}")

# Analyze some data
sample_data = [1, 5, 3, 9, 2, 8, 4]
stats = analyze_data(sample_data)
print(f"Data analysis: {stats}")
print()

# =============================================================================
# 5. WORKING WITH LISTS (Important for Data Analysis)
# =============================================================================

data = [1, 5, 3, 9, 2, 8, 4]
print(f"Original data: {data}")

# List comprehensions (very useful!)
squared = [x**2 for x in data]           # [1, 25, 9, 81, 4, 64, 16]
evens = [x for x in data if x % 2 == 0]  # [2, 8, 4]
print(f"Squared: {squared}")
print(f"Even numbers: {evens}")

# Built-in functions
total = sum(data)        # 32
maximum = max(data)      # 9
minimum = min(data)      # 1
length = len(data)       # 7
print(f"Sum: {total}, Max: {maximum}, Min: {minimum}, Length: {length}")

# Sorting
sorted_data = sorted(data)  # [1, 2, 3, 4, 5, 8, 9]
print(f"Sorted data: {sorted_data}")
print()

# =============================================================================
# 6. WORKING WITH DICTIONARIES
# =============================================================================
# These look like they are 1:1 aliases?

sales_data = {
    "January": 1000,
    "February": 1200,
    "March": 900
}

print(f"Original sales data: {sales_data}")

# Accessing values
jan_sales = sales_data["January"]
feb_sales = sales_data.get("February", 0)  # Safer way
print(f"January sales: {jan_sales}")
print(f"February sales: {feb_sales}")

# Adding/updating
sales_data["April"] = 1100
sales_data.update({"May": 1300, "June": 1150})
print(f"Updated sales data: {sales_data}")

# Iterating
print("Monthly sales:")
for month, sales in sales_data.items():
    print(f"  {month}: ${sales}")
print()

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
print("Created sample_data.csv")

# Reading the file back
with open("sample_data.csv", "r") as file:
    content = file.read()
    print("File content:")
    print(content)

# Reading line by line
print("\nReading line by line:")
with open("sample_data.csv", "r") as file:
    lines = file.readlines()
    for i, line in enumerate(lines):
        print(f"Line {i+1}: {line.strip()}")
print()

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
print(f"10 / 2 = {safe_divide(10, 2)}")
print(f"10 / 0 = {safe_divide(10, 0)}")
print(f"'10' / 2 = {safe_divide('10', 2)}")
print()

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

print("Student Performance Analysis:")
print("-" * 50)
for student in analyzed_data:
    print(f"{student['name']}: Math={student['math']}, Science={student['science']}, "
          f"English={student['english']}, Average={student['average']:.1f}")

print(f"\nSubject Averages:")
for subject, avg in subject_avgs.items():
    print(f"  {subject.capitalize()}: {avg:.1f}")

# Find top performer
top_student = max(analyzed_data, key=lambda x: x["average"])
# lambda is a Python keyword that creates anonymous functions.
# lambda x: x["average"] creates an anonymous function that takes each student record (x) and returns their "average" field
# equivalent to...
# def get_average(x):
#     return x["average"]
# But lambda allows for this to be writing inline without defining a separate function
# So instead of comparing the entire dictionary objects, max() compares just the average values
print(f"\nTop performer: {top_student['name']} with average {top_student['average']:.1f}")

# =============================================================================
# 9. Managing Environment Variables
# =============================================================================

# shows all variable names in current scope
dir()

# returns dictionary of all local variables
locals()
