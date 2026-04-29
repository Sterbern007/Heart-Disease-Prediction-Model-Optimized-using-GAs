# CardioGA Optimizer

CardioGA Optimizer is a Streamlit web application for heart disease prediction using the UCI Heart Disease dataset. The project uses a Genetic Algorithm to optimize machine learning models for two goals:

- Improve prediction accuracy
- Reduce model training time

The app is designed as an academic machine learning project with model comparison, Pareto front analysis, feature importance, and patient-level prediction.

## Features

- Heart disease prediction using clinical patient data
- Genetic Algorithm based hyperparameter optimization
- Multi-objective optimization for accuracy and training time
- Supports Logistic Regression, Decision Tree, and Random Forest
- Pareto solutions such as:
  - Fastest Model
  - Balanced Model
  - Most Accurate Model
- Model comparison table and chart
- Fitness and accuracy convergence graphs
- Pareto front visualization
- Feature importance chart
- Patient prediction with probability and risk level
- Download Pareto results as CSV
- Download best solutions as JSON

## Dataset

The project uses the processed Cleveland subset of the UCI Heart Disease dataset.

Main features include:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Slope
- Number of major vessels
- Thalassemia

The target column is converted into a binary class:

```text
0 = No heart disease
1 = Heart disease present
```

## Genetic Algorithm

In this project, each chromosome represents one model configuration. Each gene represents a hyperparameter.

The Genetic Algorithm follows these steps:

```text
Initial Population
Fitness Evaluation
Selection
Crossover
Mutation
New Generation
Pareto Solution Selection
```

The fitness function balances accuracy and training time:

```text
Fitness = 0.82 * Accuracy + 0.18 * Time Score
```

Higher accuracy increases fitness, while lower training time improves the time score.

## Models Used

### Logistic Regression

Optimized parameters:

- C
- penalty
- max_iter
- class_weight
- tol

### Decision Tree

Optimized parameters:

- max_depth
- min_samples_split
- min_samples_leaf
- criterion
- class_weight

### Random Forest

Optimized parameters:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features
- bootstrap

## Project Structure

```text
New project/
│
├── app.py
├── ga.py
├── main.py
├── model.py
├── utils.py
├── requirements.txt
├── README.md
│
├── data/
│   └── processed.cleveland.data
│
└── outputs/
```

## Installation

Open the project folder in VS Code, then run:

```powershell
cd "D:\CSE275 Project\New project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd ".\New project"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Run the App

```powershell
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## How to Use

1. Select the models from the sidebar.
2. Choose population size, number of generations, and mutation rate.
3. Click **Run GA**.
4. View optimized results in the Results tab.
5. Check convergence and Pareto graphs in the Graphs tab.
6. Select a Pareto solution for prediction.
7. Enter patient details in the Prediction tab.
8. View probability, risk level, and top contributing features.

## Output

The app shows:

- Test accuracy
- Training time
- Fitness score
- Pareto-optimal solutions
- Model comparison
- Feature importance
- Heart disease probability
- Risk level: Low, Medium, or High

## Purpose

This project demonstrates how Genetic Algorithms can be used for multi-objective optimization in healthcare machine learning. Instead of focusing only on accuracy, it also considers training time, making the model more practical and efficient.

## Disclaimer

This project is for academic and educational purposes only. It is not a certified medical diagnosis system.
