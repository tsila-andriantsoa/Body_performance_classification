# Body_performance_classification

Welcome to the **Body_performance_classification** project! 

Physical fitness and body performance are key indicators of an individual's overall health and well-being. Many organizations, such as sports teams, healthcare institutions, and fitness centers, rely on physical performance assessments to classify individuals based on their strength, flexibility, endurance, and body composition. The Body Performance Data dataset provides various physiological attributes (age, gender, height, weight, body fat percentage, blood pressure, and physical performance metrics) to classify individuals into four categories: A (best), B, C, and D (worst).

Understanding how different physiological and performance-related factors influence body performance can help in designing targeted fitness programs, medical assessments, and personalized training plans. 

This repository provides tools and resources and aim to predict body performance levels based on available physiological features, offering valuable insights for fitness trainers, medical professionals, and sports analysts.

![Body performance classification](https://github.com/tsila-andriantsoa/health_insurance_lead_prediction/blob/main/img/body_performance_classification.jpeg)

## Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/kukuroo3/body-performance-data)

| Variable | Definition |
|------------------|-----------------|
| age | Age of the individual in years (20 ~64). |
| gender | Biological sex of the individual, denoted as 'M' for male and 'F' for female. |
| height_cm | Height of the individual measured in centimeters. (If you want to convert to feet, divide by 30.48) |
| weight_kg | Weight of the individual measured in kilograms. |
| body fat_% | Percentage of the individual's body mass that is fat tissue. |
| diastolic | Diastolic blood pressure of the individual (min) |
| systolic | Systolic blood pressure of the individual (min) |
| gripForce | Grip strength of the individual |
| sit and bend forward_cm | Distance reached in the sit-and-reach test, measured in centimeters. |
| sit-ups counts | Number of sit-ups completed by the individual. |
| broad jump_cm | Distance covered in a standing broad jump, measured in centimeters. |
| class | Categorical variable representing the individual's overall body performance, classified into four levels: 'A' (best), 'B', 'C', and 'D' (worst). |

## Multiclass classification task

You’re already familiar with binary classification, where a model predicts one of two possible outcomes (e.g., "Yes" or "No", "Spam" or "Not Spam"). But what if we need to classify data into more than two categories?

This is where multiclass classification comes in. Instead of choosing between just two labels, the model must decide which one of several categories an example belongs to.

Analogy: Sorting Fruits
Imagine you work at a grocery store and need to sort fruits.

A binary classifier would only distinguish apples vs. non-apples.
A multiclass classifier, however, would identify if a fruit is an apple, banana, orange, or pear.
How It Works in Our Case
In the Body Performance Classification project, instead of just predicting whether someone has "good" or "poor" fitness (binary classification), we classify them into four performance categories (A, B, C, or D) based on their physical attributes and test results.

Multiclass classification is just like binary classification but instead of two choices, we have multiple categories and the model learns to recognize patterns that distinguish between them.

## Evaluation metric

For evaluation metric, we use F1 score. Using the F1 score allows for a fairer evaluation of the model by balancing precision and recall, making it well-suited for multi-class problems with potential class imbalances like body performance classification.

## Model training

The first step in the project was data preprocessing. Missing values were handled appropriately, and column data types were transformed to ensure compatibility with machine learning models. Following this, an exploratory data analysis (EDA) was conducted to identify basic patterns and relationships in the dataset, such as trends and correlations among key features.

Next, feature importance analysis was performed to understand the contribution of each variable to the prediction task.

For the modeling phase, three algorithms were chosen based on their suitability for the project:

- Random Forest
- Decision Tree
- XGBoost

In the second step, baseline models were trained using all features and the best hyperparameters for each algorithm. The best baseline model was then selected based on its performance using the F1 score, achieving a baseline score of **75%**.

To further improve the model, we dived into feature engineering and create 4 new features based on crossing existant features. After analysis, we observed that introduction these features didn't inprove the baseline model since information has been already captured by the model.

## Setup Instructions

To set up this project locally with pipenv, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/tsila-andriantsoa/Body_performance_classification.git
   ```

2. Activate virtual environment (make sure pipenv is already installed):
   ```bash
   pipenv shell
   ```

3. Install Dependencies:
   ```bash
   pipenv install
   ```
   XGBoost dependencies could not be resolved using pipenv and python version used for the environment. Run the following command to bypass this mecanism
   ```bash
   pipenv run pip install xgboost streamlit
   ```
   
4. Running the project

- Run the project locally with **pipenv**

  A Trained model is already available within the folder **model**. However, if one wants to re-train the model, it can be done by running the following command.
   ```bash
   pipenv run python scr/train.py
   ```
   
  To serve the model, run the following command.
   ```bash
   pipenv run python scr/predict.py
   ```
   
  Once app deployed, requests can be made using the following command that provides an example of prediction using a sample json data.
   ```bash
   pipenv run python src/predict_test.py
   ```
   
   
- Set up the projet using **Docker Container**

  Build the docker image (make sure docker is already installed):
   ```bash
   docker build -t predict-app .
   ```

  Run the image as Docker container:
   ```bash
   docker run -d -p 5000:5000 predict-app
   ```   
