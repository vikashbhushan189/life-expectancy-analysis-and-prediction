# 📈 Life Expectancy Analysis and Prediction

## Project Overview

This project focuses on analyzing factors influencing life expectancy across various countries and years (2000-2015), as well as building a machine learning model to predict life expectancy based on these factors. The analysis covers demographic, economic, social, immunization, and mortality-related indicators.

The goal is to understand which variables significantly impact average lifespan and provide a tool to predict life expectancy, aiding in identifying key areas for health and development improvements.

## Key Features

*   **Comprehensive Data Collection & Cleaning:** Handles raw dataset loading, standardizes column names, imputes missing numerical values using column means, and removes duplicate entries.
*   **Outlier Handling:** Implements IQR-based outlier detection and replacement with column means to ensure robust statistical analysis and model training.
*   **Exploratory Data Analysis (EDA):**
    *   Visualizes the distribution of life expectancy and other key factors.
    *   Generates detailed correlation matrices.
    *   Explores relationships between life expectancy and economic indicators (e.g., GDP).
    *   Analyzes trends of average life expectancy and related health factors over time.
    *   Compares key metrics across "Developed" vs. "Developing" countries.
*   **Statistical Analysis:**
    *   Calculates Pearson correlation coefficients between life expectancy and influential factors.
    *   Performs hypothesis testing (e.g., T-test) to compare life expectancy between different country statuses.
    *   Constructs and evaluates a **Multivariate Linear Regression model (OLS)** to understand feature contributions.
*   **Machine Learning Regression Models:**
    *   Compares various advanced regression algorithms including **Random Forest Regressor**, **Extra Trees Regressor**, **Gradient Boosting Regressor**, and **XGB Regressor**.
    *   Applies **Standard Scaling** to numerical features for optimal model performance.
    *   Performs **K-Fold Cross-Validation** on the best-performing model to assess its generalization capability.
*   **Model Persistence:** The best-trained regression model and all necessary preprocessing tools (scaler, LabelEncoder for Status) are saved.
*   **Streamlit Web Application:** Provides an interactive interface for users to input a country's attributes and get a predicted life expectancy.

## Project Structure

life-expectancy-analysis-and-prediction/
├── customer_support_tickets.csv # NOTE: This file appears to be from a previous project.
# You will likely have 'Life Expectancy Data.csv' here instead.
├── life_expectancy_artifacts/ # Directory for saved model and preprocessing objects
│ ├── best_life_expectancy_model.joblib
│ ├── scaler_ml.joblib
│ ├── ml_feature_names.joblib
│ └── label_encoder_status.joblib # (Optional, only if Status column was processed as object)
├── train_and_save_model.py # Primary script for data processing, EDA, training, and artifact saving
├── life_expectancy_app.py # Python script for the Streamlit web application
├── README.md # This README file
└── requirements.txt # List of Python dependencies for easy setup
## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd your-repository-name
    ```
    (Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub details)

2.  **Dataset:** Place your `Life Expectancy Data.csv` dataset directly into the project's root directory. The `train_and_save_model.py` script expects it there.

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```

4.  **Activate the Virtual Environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

5.  **Install Dependencies:**
    Make sure you have `requirements.txt` generated from your final environment (run `pip freeze > requirements.txt` after successfully installing all packages locally).
    ```bash
    pip install -r requirements.txt
    ```
    **Note:** Ensure `scikit-learn`, `imbalanced-learn`, `xgboost`, `wordcloud`, `streamlit` are installed and up-to-date. You might need to update/install specific versions:
    ```bash
    pip install -U scikit-learn imbalanced-learn  # Important for compatibility
    pip install xgboost wordcloud streamlit        # Ensure all core libraries are installed
    ```

6.  **Download NLTK Data:**
    The scripts will attempt to download these automatically, but you can do it manually if preferred:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    ```
    (You can run these two lines in a Python interpreter.)

## How to Run

### Step 1: Analyze Data, Train Model, and Save Artifacts

This step processes the data, performs the analysis outlined above, trains the machine learning regression model, and saves the trained model along with all necessary preprocessing components (`scaler`, `LabelEncoder` for Status, and feature names) into the `life_expectancy_artifacts` directory.

1.  Ensure you are in the project's root directory and your virtual environment is active.
2.  **Create the `life_expectancy_artifacts` directory** (if it doesn't already exist from a previous run):
    ```bash
    mkdir life_expectancy_artifacts
    ```
3.  Run the main analysis and training script:
    ```bash
    python train_and_save_model.py
    ```
    This script will print detailed logs regarding data loading, cleaning, EDA insights, statistical analysis results, model training performance, and will confirm when all artifacts are successfully saved. Plots generated by EDA will be displayed (you may need to close them to proceed).

### Step 2: Run the Streamlit Web Application

Once `train_and_save_model.py` completes its execution successfully and the `life_expectancy_artifacts` folder is populated, you can launch the interactive Streamlit application.

1.  Ensure you are in the project's root directory and your virtual environment is active.
2.  Run the Streamlit application:
    ```bash
    streamlit run life_expectancy_app.py
    ```
    Your default web browser will open to the Streamlit app interface, allowing you to predict life expectancy based on user-defined inputs.

## Model Performance & Discussion

The machine learning model is a regression model aiming to predict a continuous value (Life Expectancy in years).

*   **Best Model Identified:** Refer to the output of `train_and_save_model.py` for the specific best model, typically an **XGB Regressor** or **Gradient Boosting Regressor** based on `R2 Score` and `RMSE`.
*   **Key Performance Metrics:**
    *   **Mean Cross-Validated R2 Score:** Approximately `0.96` (indicating that the model explains about 96% of the variance in life expectancy, based on typical results for this dataset).
    *   **Mean Cross-Validated RMSE:** Typically low, indicating good predictive accuracy in years.
    (Refer to the specific output of `train_and_save_model.py` for the exact figures for your run).

**Insights from Analysis:**

*   **(Example based on common findings for this dataset):** You would include key takeaways from your EDA and Statistical Analysis, such as:
    *   Strong positive correlation between "Schooling" and "Life expectancy".
    *   Strong negative correlation between "Adult Mortality", "infant deaths", "under-five deaths", and "HIV/AIDS" with "Life expectancy".
    *   GDP and percentage expenditure on health also tend to have a positive correlation.
    *   Developed countries generally exhibit higher life expectancy than developing ones.

## Future Enhancements

*   **Feature Engineering:** Explore more complex feature interactions, time-series aspects (e.g., trend of individual country's factors over years).
*   **Deep Learning Models:** Experiment with neural networks for regression tasks, especially for potential non-linear relationships.
*   **Uncertainty Quantification:** Provide confidence intervals or prediction intervals around the predicted life expectancy, to convey the model's uncertainty.
*   **Advanced Model Tuning:** Implement more extensive hyperparameter tuning strategies (e.g., Bayesian optimization) for optimal model performance.
*   **Containerization (Docker):** Package the application and its environment into a Docker container for easier deployment and portability.
*   **Cloud Deployment:** Deploy the Streamlit application on cloud platforms (e.g., Streamlit Community Cloud, AWS EC2/ECS, Google Cloud Run) for wider accessibility.

---

