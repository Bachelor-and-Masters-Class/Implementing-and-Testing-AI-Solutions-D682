**Overview**

This repository contains my completed submission for the WGU course **D682: Implementing and Testing AI Solutions**.

The project implements and validates a complete supervised machine learning system for predicting urban air quality and associated population health outcomes. Rather than a single notebook, the deliverable is a parameterized, command-line driven pipeline that can be pointed at a dataset, configured for one or more prediction targets, and run reproducibly to produce a full evidence trail of results.

The focus of this course is the disciplined implementation and testing of AI solutions: verifying that preprocessing behaves correctly on imperfect data, that validation strategy matches the structure of the data, and that results are captured in a form that can be independently reviewed.

**Scenario**

Environmental and health monitoring data is collected across an urban area over time. The organization needs predictive models for several related outcomes drawn from the same underlying dataset:

-PM2.5 particulate concentration

-Health risk score

-Severity score

Because observations are time-ordered, any evaluation that randomly shuffles the data risks training on future information and reporting inflated accuracy. Correct validation design is therefore part of the requirement, not an afterthought.

**Project Objectives**

-Implement a reusable machine learning pipeline capable of handling multiple targets

-Build preprocessing that is resilient to missing values, mixed data types, and unseen categories

-Select and justify a validation strategy appropriate to time-series structured data

-Perform systematic hyperparameter optimization

-Test and evaluate model performance using multiple complementary error metrics

-Produce reproducible artifacts that document every result

**Technical Implementation**

**Data Loading and Discovery**
Excel input is read through pandas and openpyxl. A dedicated column listing mode prints all available columns with their data types and heuristically identifies plausible numeric prediction targets, supporting exploratory work before committing to a model configuration.

**Preprocessing**
A ColumnTransformer separates numeric and categorical features. Numeric columns receive median imputation; categorical columns receive most-frequent imputation followed by one-hot encoding configured to ignore unknown categories at inference time.

**Feature Engineering**
When a timestamp column is provided, calendar features are derived automatically. Epoch values are detected as either seconds or milliseconds and converted to timezone-naive datetimes before extraction.

**Model Selection and Tuning**
A GradientBoostingRegressor is tuned with RandomizedSearchCV over a compact but effective search space covering number of estimators, learning rate, maximum depth, subsample ratio, minimum samples per leaf, and feature sampling. Scoring is based on negative root mean squared error.

**Testing and Validation**
Time-aware runs use a chronological holdout split with TimeSeriesSplit cross-validation to prevent leakage. Non-temporal runs use a randomized split with shuffled K-Fold. Final models are scored on a test set never seen during training or tuning.

**Evaluation Metrics**
RMSE, MAE, and MAPE are computed for each target. MAPE is calculated with a guard against division by near-zero true values.

**Artifact Output**
Each target receives its own artifacts directory containing metrics in JSON and CSV, a predictions file, a ranked feature importance table, a predicted-versus-actual plot, a top-25 feature importance chart, and a generated summary README.

**Skills Demonstrated**

-Machine learning pipeline implementation and modularization

-Test design for data preprocessing and model validation

-Prevention of data leakage in time-ordered datasets

-Automated hyperparameter search

-Multi-metric regression evaluation

-Model interpretability through feature importance analysis

-Reproducible experiment documentation

-CLI application design with configurable parameters

**How to Run**

Install dependencies:

```
pip install -r requirements.txt
```

Explore the dataset:

```
python dqn1_model.py --data "DQN1 Dataset.xlsx" --list-columns
```

Train and evaluate:

```
python dqn1_model.py --data "DQN1 Dataset.xlsx" --target "PM2.5,health_risk_score" --datecol sunriseEpoch
```

**Repository Contents**

-`dqn1_model.py`: Complete pipeline: ingestion, preprocessing, tuning, evaluation, and artifact generation

-`requirements.txt`: Python dependencies

-`artifacts/pm2.5/`: Evaluation results and visualizations for PM2.5 prediction

-`artifacts/healthRiskScore/`: Evaluation results and visualizations for health risk prediction

-`artifacts/severityScore/`: Evaluation results and visualizations for severity prediction
