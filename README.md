# Machine Learning Operations : MLOps

Project advance 1 for the subject Problems resolution with technology application.

# Module 1 - Key concepts of ML systems

## Base-line definition and project scope

**Base-line -**
This the baseline of the project.

* A Random Forest model is implemented, which loads a dataset and applies StandardScaler transformation to all numerical fields, also applies 70-30 partition for train and test sets respectively, achieving a final accuracy around 85%, and cross validation accuracy around 82%.

**Scope**

* The expected scope of this project is implementation of techniques and good practices to achieve deployment of the full functionality of this code through REST API.

**Situation**

* This is an excersice taken from kaggle to work with, in which the objective is to try to determine the median value of owner-occupied homes in 
] (MEDV, dependent variable), given a serie of independent variables like structural, neighborhood, accessibility and air pollution data in Boston around 70's.

* To know more about the dataset you can see directly kaggle [link](https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data).

## Notebook

* **Base-line** code may be found at [base_line.ipynb](https://github.com/JDEQ413/mlops_project/blob/main/docs/base_line.ipynb).
  * It was tacken from several notebooks developed by other users in kaggle platform.
    * [MAGANTI IT](https://www.kaggle.com/code/magantiit/linearregression)
    * [SADIK AL JARIF](https://www.kaggle.com/code/sadikaljarif/boston-housing-price-prediction)
    * [MARCIN RUTECKI](https://www.kaggle.com/code/marcinrutecki/regression-models-evaluation-metrics)
    * [UNMOVED](https://www.kaggle.com/code/unmoved/regress-boston-house-prices)
 
* **Sectioned notebook** contains the code that was identified in every recommended section.
  * It may be found at [house-price.ipynb](https://github.com/JDEQ413/mlops_project/blob/main/docs/house-price.ipynb)
 
* **.py file** is the result of first steps to acomplish refactorization, and is the base to separate sections in folders.
  * It may be found at [house-price.py](https://github.com/JDEQ413/mlops_project/blob/main/docs/house-price.py)
 

# Module 2 - Basic concepts and tools for software development

## Virtual environments

* Change the directory to the "mlops_project" folder.
* Create a virtual environment with Python 3+:
    ```bash
    python3 -m venv venv
    ```
    * Windows cmd:
      ```bash
      py -3.10 -m venv venv310
      ```
 
* Activate the virtual environment
    ```bash
    source venv/bin/activate
    ```
    * Windows cmd:
      ```bash
      venv310\scripts\activate.bat
      ```

* Install the other libraries. Run the following command to install the libraries/packages.
    ```bash
    pip install -r requirements-310.txt
    ```
    * Requirements file may be consulted [here](https://github.com/JDEQ413/mlops_project/blob/main/requirements-310.txt).

* Don´t forget to select the new environment at kernel when using VSCode.

## Continuous use of GitHub

* GitHub was used continuosly during the development of this project, increasing graduately the content of the repository.
  * [mlops_project](https://github.com/JDEQ413/mlops_project)

## Unit tests

* Unit tests developed to acomplish basic validations on the project.
  * Folder: [tests](https://github.com/JDEQ413/mlops_project/tree/main/tests)

## Pre-Commits

* Pre-commits were implemented on the project from [house-price.py](https://github.com/JDEQ413/mlops_project/blob/main/docs/house-price.py) file onward, since this was the first element pre-commits are able to evaluate.
* This are the repos implemented for linting and formatting:
  * isort
  * autoflake
  * autopep8
  * flake8
    * _Every one has its own hooks to represent specific checks on the code._
    * _The corresponding libraries are contained inside requirements-310.txt file. They may be installed but nothing will happen if .yaml file does not exist or is empty, or pre-commit has not been initialized on the project for the first time._
* Configuration file can be found at [.pre-commit-config.yaml](https://github.com/JDEQ413/mlops_project/blob/main/.pre-commit-config.yaml)

**Setup pre-commits**

* Open your terminal or command prompt, navigate to the root directory of your project
* Pre-commit needs to be installed, at this time it already was by que _requirements_ file.
  * If you are installing it for the firs time use:
    ```bash
    pip install pre-commit
    ```
* After creating the .pre-commit-config.yaml file, initialize pre-commit for the project:
  ```bash
  pre-commit install
  ```
* With the pre-commit hooks installed, you can now make changes to your Python code. When you're ready to commit your changes, run the following command to trigger the pre-commit checks:
  ```bash
  git commit -m "add pre-commit file"
  ```
* If every check "passed", then you are ready to upload your changes to the repository at GitHub.
  ```bash
  git push
  ```
