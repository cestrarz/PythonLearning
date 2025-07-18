# 🐍 Python Learning Repository

A structured collection of Python learning materials, utilities, and projects focused on data analysis and software development best practices.

## 📁 Project Structure

```text
PythonLearning/
├── code/                     # All code examples and projects
│   ├── other imported/       # Third-party code examples
│   ├── self-made course/     # Custom learning materials
│   └── testing/              # Testing scripts and templates
│
├── data/                     # Data files for analysis
│
├── output/                   # Generated outputs and results
│   └── logs/                 # Log files from scripts
│
└── python/                   # Core Python utilities
    └── utils/                # Reusable utility modules
        ├── __init__.py
        └── logging_utils.py  # Logging configuration and utilities
```

## 🛠️ Key Components

### Core Utilities

- **logging_utils.py**: A robust logging setup that captures both console output and log files
- **Template Scripts**: Reusable starting points for new Python projects

### Learning Materials

- **Self-Made Course**: Custom learning modules and exercises
- **Imported Examples**: Curated third-party examples for reference

## 🚀 Getting Started

1. **Setup Python Environment**

   ```bash
   # Create and activate virtual environment
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   
   # Install dependencies
   pip install -r python/requirements.txt
   ```

2. **Using the Logging Utility**

   ```python
   from python.utils.logging_utils import setup_logging
   
   # Initialize logger
   logger = setup_logging(log_file="my_script.log")
   logger.info("Your message here")
   ```

## 📊 Data Analysis Workflow

1. Place input data in `data/`
2. Create analysis scripts in `code/`
3. Output results to `output/`
4. Review logs in `output/logs/`


---

### **02 - NumPy Fundamentals**
*Prerequisites: Files 00-01*
- NumPy arrays and vectorized operations
- Mathematical functions and broadcasting
- Statistical operations and aggregations
- Array manipulation and reshaping
- Performance optimization techniques

**Key Skills:** Numerical computing, efficient data processing

---

### **03 - Pandas Data Manipulation**
*Prerequisites: Files 00-02*
- DataFrames and Series operations
- Data cleaning and transformation
- Grouping, aggregation, and pivot tables
- Merging and joining datasets
- Time series analysis basics
- File I/O (CSV, Excel, JSON, SQL)

**Key Skills:** Data wrangling, exploratory data analysis

---

### **04 - Matplotlib Visualization**
*Prerequisites: Files 00-03*
- Basic plotting and customization
- Multiple plot types (line, bar, scatter, histogram)
- Subplots and complex layouts
- Statistical visualizations
- Publication-quality figures
- Business dashboard creation

**Key Skills:** Data visualization, presentation graphics

---

### **05 - Seaborn Statistical Visualization**
*Prerequisites: Files 00-04*
- Advanced statistical plots
- Distribution and relationship analysis
- Categorical data visualization
- Multi-plot grids and faceting
- Beautiful default themes
- Integration with statistical analysis

**Key Skills:** Statistical graphics, exploratory data analysis

---

## 🎓 **ECONOMETRICS SPECIALIZATION**

### **06 - Econometrics Fundamentals**
*Prerequisites: Files 00-05*
- **Ordinary Least Squares (OLS)** regression
- **Robust standard errors** (White, HC1, HC3)
- **Clustered standard errors** for grouped data
- **Fixed effects** and **random effects** models
- **Instrumental Variables (IV)** estimation
- Regression diagnostics and assumption testing
- Model comparison and statistical testing

**Key Skills:** Causal inference, regression analysis, econometric theory

**Methods Covered:**
- OLS with heteroscedasticity-robust inference
- Panel data methods (within/between estimators)
- Two-stage least squares (2SLS)
- Hausman tests and specification testing

---

### **07 - Advanced Econometric Methods**
*Prerequisites: Files 00-06*
- **Regression Discontinuity Design (RDD)**
  - Sharp and fuzzy RD
  - Bandwidth selection
  - Robustness checks
- **Difference-in-Differences (DiD)**
  - Parallel trends assumption
  - Two-way fixed effects
  - Event study analysis
- **Synthetic Control Method**
  - Donor pool construction
  - Weight optimization
  - Placebo tests
- **Propensity Score Matching**
  - Treatment effect estimation
  - Covariate balance
  - Sensitivity analysis

**Key Skills:** Program evaluation, policy analysis, causal identification

**Methods Covered:**
- RDD with local polynomial regression
- DiD with staggered treatment timing
- Synthetic control for comparative case studies
- Matching estimators and overlap diagnostics

---

### **08 - LaTeX Tables & Publication Output**
*Prerequisites: Files 00-07*
- **Professional regression tables** with Stargazer
- **Summary statistics tables** by groups
- **Correlation matrices** and descriptive statistics
- **Robustness and sensitivity tables**
- **Custom table formatting** functions
- **Export to LaTeX, HTML, and Word**
- Publication best practices and style guides

**Key Skills:** Academic writing, professional presentation, reproducible research

**Output Types:**
- Journal-ready regression tables
- Conference presentation tables
- Policy report formatting
- Thesis and dissertation tables

---

## 🚀 **ADVANCED METHODS**

### **09 - Machine Learning with Scikit-learn**
*Prerequisites: Files 00-08*
- Classification and regression algorithms
- Model selection and evaluation
- Cross-validation and hyperparameter tuning
- Clustering and dimensionality reduction
- Feature engineering and preprocessing
- Complete ML pipeline development

**Key Skills:** Predictive modeling, pattern recognition, automated analysis

---

### **10 - Multi-core Parallelization**
*Prerequisites: Files 00-09*
- Understanding the Global Interpreter Lock (GIL)
- Multiprocessing vs threading
- Joblib for scientific computing
- Pandas parallel processing
- Memory-efficient computation
- Asynchronous programming for I/O

**Key Skills:** High-performance computing, scalable analysis, optimization

---