# AWS Bedrock Model Performance Comparsion

## Setup

### Python Package Manager

uv is used for the package manager given its robustness and performance.

Make sure you have uv installed in your environment.

```bash
pip install uv
```

### Python Virtual Environement

It is always a best practice to create a virtual environment for running a Python project instead of the global python environment.

Make sure you create and activate the project virtual environment.

```bash
uv venv ./venv
source venv/Scripts/activate
```

### Project Dependency

Install required dependencies

```bash
uv pip install -r requirements.txt
```

### Linter and Formatter

Ruff is used in this project as linter and formatter, promote code quality and team collaboration.

Here are some commonly used commands:

Check Python error

```bash
ruff check
```

Format all Python files

```bash
ruff format
```
