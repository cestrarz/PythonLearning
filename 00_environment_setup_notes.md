# Python Environment Setup

This guide will help you set up a proper Python environment for data analysis.

## What is a virtual environment?

A virtual environment is an isolated environment that allows you to manage dependencies for your project. It creates an isolated space where you can install packages and dependencies without affecting the global Python environment.

## Why use a virtual environment?

- It allows you to manage dependencies for your project.
- It creates an isolated space where you can install packages and dependencies without affecting the global Python environment.
- It allows you to have different versions of packages for different projects.

## Initial Set up

1. **Open the folder/directory** where your project/code will be stored.
2. **Open the terminal** in that folder/directory.
3. **Initialize a virtual environment**:
   ```bash
   python3 -m venv env
   ```
4. **Activate the virtual environment**:
   ```bash
   source env/bin/activate
   ```

## Other useful commands

- **Deactivate the virtual environment**:
  ```bash
  deactivate
  ```
- **Check packages installed in the virtual environment**:
  ```bash
  pip list
  ```
- **Export installed packages to a requirements file**:
  ```bash
  pip freeze > requirements.txt
  ```
- **Install packages from a requirements file**:
  ```bash
  pip install -r requirements.txt
  ```
