# 01 - Python Environment Setup for Data Analysis
# This guide will help you set up a proper Python environment for data analysis

"""
PYTHON ENVIRONMENT SETUP GUIDE
==============================

A proper environment setup is crucial for data analysis work. This prevents
package conflicts and keeps your projects organized.

STEP 1: Check Your Python Installation
-------------------------------------
"""

import sys
import subprocess

def check_python_version():
    """Check current Python version"""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if Python 3.8+ (recommended for data analysis)
    if sys.version_info >= (3, 8):
        print("✅ Python version is suitable for data analysis")
    else:
        print("⚠️  Consider upgrading to Python 3.8 or higher")

check_python_version()

"""
STEP 2: Package Managers
-----------------------

pip: Built-in Python package manager
conda: Popular for data science (comes with Anaconda/Miniconda)

Check what you have:
"""

def check_package_managers():
    """Check available package managers"""
    managers = []
    
    try:
        result = subprocess.run(['pip', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            managers.append(f"pip: {result.stdout.strip()}")
    except FileNotFoundError:
        pass
    
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            managers.append(f"conda: {result.stdout.strip()}")
    except FileNotFoundError:
        pass
    
    if managers:
        print("Available package managers:")
        for manager in managers:
            print(f"  ✅ {manager}")
    else:
        print("⚠️  No package managers found")

check_package_managers()

"""
STEP 3: Virtual Environments
----------------------------

Virtual environments isolate your project dependencies.
This prevents conflicts between different projects.

OPTION A: Using venv (built-in)
------------------------------
# Create virtual environment
python -m venv data_analysis_env

# Activate (macOS/Linux)
source data_analysis_env/bin/activate

# Activate (Windows)
data_analysis_env\\Scripts\\activate

# Deactivate
deactivate

OPTION B: Using conda
--------------------
# Create environment
conda create -n data_analysis python=3.11

# Activate
conda activate data_analysis

# Deactivate
conda deactivate

STEP 4: Essential Data Analysis Packages
---------------------------------------

Once your environment is activated, install these packages:

# Core data analysis stack
pip install numpy pandas matplotlib seaborn

# Jupyter for interactive analysis
pip install jupyter jupyterlab

# Additional useful packages
pip install scikit-learn plotly requests beautifulsoup4

# Or install everything at once
pip install numpy pandas matplotlib seaborn jupyter jupyterlab scikit-learn plotly requests beautifulsoup4

STEP 5: Verify Installation
--------------------------
"""

def verify_data_packages():
    """Check if essential data analysis packages are installed"""
    essential_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'jupyter', 'sklearn', 'requests'
    ]
    
    print("Checking essential packages:")
    for package in essential_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Not installed")

verify_data_packages()

"""
STEP 6: IDE/Editor Setup
-----------------------

Good options for data analysis:

1. Jupyter Notebook/Lab (Interactive, great for exploration)
   - Start with: jupyter notebook
   - Or: jupyter lab

2. VS Code (Great all-around editor)
   - Install Python extension
   - Install Jupyter extension

3. PyCharm (Full-featured IDE)
   - Professional edition has great data science tools

4. Spyder (Designed for scientific computing)
   - Similar to RStudio/MATLAB

STEP 7: Project Structure
------------------------

Organize your data analysis projects like this:

project_name/
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned, transformed data
│   └── external/     # External data sources
├── notebooks/        # Jupyter notebooks
├── src/             # Source code/modules
├── reports/         # Generated reports
├── requirements.txt # Package dependencies
└── README.md        # Project documentation

STEP 8: Best Practices
---------------------

1. Always use virtual environments
2. Keep a requirements.txt file:
   pip freeze > requirements.txt

3. Use version control (Git)
4. Document your analysis process
5. Keep raw data immutable
6. Use meaningful variable names
7. Comment your code

NEXT STEPS
----------
1. Set up your virtual environment
2. Install the essential packages
3. Test with a simple import
4. Move to file 02 - NumPy basics
"""

# Simple test to verify your setup works
if __name__ == "__main__":
    print("\n" + "="*50)
    print("ENVIRONMENT SETUP TEST")
    print("="*50)
    
    # Test basic Python functionality
    test_data = [1, 2, 3, 4, 5]
    print(f"Basic Python test: sum of {test_data} = {sum(test_data)}")
    
    # Try importing essential packages
    try:
        import numpy as np
        print("✅ NumPy import successful")
        print(f"   NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not available - install with: pip install numpy")
    
    try:
        import pandas as pd
        print("✅ Pandas import successful")
        print(f"   Pandas version: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not available - install with: pip install pandas")
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib import successful")
    except ImportError:
        print("❌ Matplotlib not available - install with: pip install matplotlib")
    
    print("\nIf you see ❌ symbols, install the missing packages before proceeding!")
    print("Once everything shows ✅, you're ready for the next file!")