import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Load Preprocessing Artifacts and Model ---
@st.cache_resource # Cache resource to load these heavy objects only once
def load_artifacts():
    try:
        model = joblib.load('life_expectancy_artifacts/best_life_expectancy_model.joblib')
        scaler = joblib.load('life_expectancy_artifacts/scaler_ml.joblib')
        ml_feature_names = joblib.load('life_expectancy_artifacts/ml_feature_names.joblib')
        
        # Try to load the status LabelEncoder only if it was saved (i.e., 'Status' was encoded)
        le_status = None
        try:
            le_status = joblib.load('life_expectancy_artifacts/label_encoder_status.joblib')
        except FileNotFoundError:
            st.info("LabelEncoder for 'Status' not found (maybe 'Status' was not an object or was not explicitly saved). Proceeding without it.")

        return model, scaler, ml_feature_names, le_status
    except FileNotFoundError as e:
        st.error(f"Error loading artifacts: {e}. Please ensure 'train_and_save_model.py' has been run successfully and the 'life_expectancy_artifacts' folder is correctly populated.")
        st.stop() # Stop the app if crucial artifacts are missing
    except Exception as e:
        st.error(f"An unexpected error occurred during artifact loading: {e}. Check the integrity of your .joblib files.")
        st.stop()

model, scaler_ml, ml_feature_names, le_status = load_artifacts()


# --- Streamlit UI ---
st.set_page_config(page_title="Life Expectancy Prediction", layout="wide", initial_sidebar_state="auto")

st.title("🌍 Life Expectancy Prediction Model")
st.markdown("""
    This application predicts the Life Expectancy (in years) based on various health, economic, and social factors for a country.
    Please input the details below.
""")

st.header("Country & Health Factors")

# Define input widgets for features expected by your model
# This list MUST correspond to the `ml_feature_names` saved by the training script.
# Categorize inputs into sections. Use default values relevant for typical inputs.

input_values = {}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Economics")
    input_values['Year'] = st.number_input("Year", min_value=2000, max_value=2015, value=2010, help="The year for which to predict life expectancy.")
    
    # Check if Status_encoded is in ml_feature_names
    status_feature_present = False
    if 'Status_encoded' in ml_feature_names:
        status_feature_present = True
        if le_status: # Check if label_encoder_status was successfully loaded
            status_options = list(le_status.classes_)
            selected_status_text = st.selectbox("Status (Developed/Developing)", status_options, index=status_options.index("Developing"), help="Developed or Developing status of the country.")
        else: # Fallback if le_status didn't load for some reason, directly ask for numeric input (less user-friendly)
            input_values['Status'] = st.number_input("Status (0 for Developing, 1 for Developed)", min_value=0, max_value=1, value=0)

    input_values['Population'] = st.number_input("Population", min_value=10000.0, max_value=1500000000.0, value=10000000.0, format="%.0f", help="Population of the country.")
    input_values['GDP'] = st.number_input("GDP per capita (USD)", min_value=10.0, max_value=150000.0, value=5000.0, format="%.1f", help="Gross Domestic Product per capita.")
    input_values['Income composition of resources'] = st.number_input("Income Composition of Resources", min_value=0.0, max_value=1.0, value=0.6, format="%.2f", help="Income composition of resources (e.g., Human Development Index component).")
    input_values['Schooling'] = st.number_input("Schooling (years)", min_value=0.0, max_value=25.0, value=12.0, format="%.1f", help="Number of years of Schooling.")


with col2:
    st.subheader("Health & Mortality Rates")
    input_values['Adult Mortality'] = st.number_input("Adult Mortality Rates (per 1000 pop)", min_value=0.0, max_value=700.0, value=150.0, format="%.1f", help="Probability of dying between 15-60 years per 1000 population.")
    input_values['infant deaths'] = st.number_input("Infant Deaths (per 1000 pop)", min_value=0.0, max_value=2000.0, value=30.0, format="%.1f", help="Number of infant deaths per 1000 population.")
    input_values['under-five deaths'] = st.number_input("Under-five Deaths (per 1000 pop)", min_value=0.0, max_value=2500.0, value=40.0, format="%.1f", help="Number of under-five deaths per 1000 population.")
    input_values['percentage expenditure'] = st.number_input("Percentage Expenditure on Health", min_value=0.0, max_value=20000.0, value=500.0, format="%.2f", help="Expenditure on health as a percentage of GDP per capita.")
    input_values['Total expenditure'] = st.number_input("Total Govt Expenditure on Health", min_value=0.0, max_value=20.0, value=6.0, format="%.2f", help="General government expenditure on health as percentage of total government expenditure.")
    input_values['Alcohol'] = st.number_input("Alcohol Consumption (liters per capita)", min_value=0.0, max_value=20.0, value=5.0, format="%.1f", help="Alcohol, recorded per capita (15+) consumption.")
    input_values['BMI'] = st.number_input("Average BMI", min_value=0.0, max_value=80.0, value=30.0, format="%.1f", help="Average Body Mass Index of entire population.")
    input_values['HIV/AIDS'] = st.number_input("HIV/AIDS Deaths (per 1000 births)", min_value=0.0, max_value=0.85, value=0.1, format="%.3f", help="Deaths per 1000 live births HIV/AIDS (0-4 years).")
    input_values['thinness 1-19 years'] = st.number_input("Thinness 10-19 years (%)", min_value=0.0, max_value=25.0, value=5.0, format="%.1f", help="Prevalence of thinness among children (10-19%).")
    input_values['thinness 5-9 years'] = st.number_input("Thinness 5-9 years (%)", min_value=0.0, max_value=25.0, value=4.0, format="%.1f", help="Prevalence of thinness among children (5-9%).")
    
    st.subheader("Immunization Coverage")
    input_values['Hepatitis B'] = st.number_input("Hepatitis B Coverage (%)", min_value=0.0, max_value=100.0, value=80.0, format="%.1f", help="HepB immunization coverage among 1-year-olds.")
    input_values['Measles'] = st.number_input("Measles Reported Cases (per 1000 pop)", min_value=0.0, max_value=2500.0, value=50.0, format="%.1f", help="Measles - number of reported cases per 1000 population.")
    input_values['Polio'] = st.number_input("Polio Coverage (%)", min_value=0.0, max_value=100.0, value=85.0, format="%.1f", help="Polio immunization coverage among 1-year-olds.")
    input_values['Diphtheria'] = st.number_input("Diphtheria Coverage (%)", min_value=0.0, max_value=100.0, value=85.0, format="%.1f", help="Diphtheria tetanus toxoid and pertussis immunization coverage among 1-year-olds.")


st.markdown("---")
if st.button("Predict Life Expectancy"):
    st.info("Predicting Life Expectancy... Please wait.")
    try:
        # Create a DataFrame from current inputs
        input_df = pd.DataFrame([input_values])

        # Preprocessing: Handle 'Status' if it was encoded in training
        if status_feature_present:
            if le_status and 'Status' in input_df.columns:
                # Use the fitted LabelEncoder to transform the selected text status
                input_df['Status_encoded'] = le_status.transform([selected_status_text])
                input_df = input_df.drop('Status', axis=1) # Remove the text column
            else: # Fallback if le_status is None, assume numerical Status was input
                # The input already assumed numerical for this scenario, so it would just be 'Status' in input_values
                pass
        
        # Ensure column order and presence for scaling is correct using ml_feature_names.
        # This creates a dummy DataFrame with all required columns initialized to 0.0,
        # then fills it with user input values. This handles missing (e.g., implicitly zero)
        # features correctly as expected by the scaler (if fitted on dense matrix).
        final_input_df_for_scaling = pd.DataFrame(0.0, index=[0], columns=ml_feature_names)

        for col in input_df.columns:
            if col in final_input_df_for_scaling.columns:
                final_input_df_for_scaling.loc[0, col] = input_df.loc[0, col]
        
        # The 'Status_encoded' needs special handling to ensure it replaces original 'Status' in `final_input_df_for_scaling`.
        # if `le_status` was None in load_artifacts and we input `Status` numerically, that's already taken care of by col-wise update above.
        # If text Status was converted to `Status_encoded` above.
        if status_feature_present and 'Status_encoded' in input_df.columns:
            final_input_df_for_scaling['Status_encoded'] = input_df['Status_encoded']


        # Apply scaling
        scaled_input_array = scaler_ml.transform(final_input_df_for_scaling)
        # Convert back to DataFrame, crucial for retaining column names/order for the model
        scaled_input_df = pd.DataFrame(scaled_input_array, columns=ml_feature_names, index=[0])


        # Make prediction
        predicted_life_expectancy = model.predict(scaled_input_df)[0]

        st.success(f"### Predicted Life Expectancy: **{predicted_life_expectancy:.2f} years**")
        st.markdown("---")
        st.markdown(f"**Note:** Life expectancy is an complex outcome influenced by many factors. This prediction is based on a statistical model and the data it was trained on.")

    except KeyError as e:
        st.error(f"Missing input column: {e}. A column required by the model was not provided. This could indicate an issue with how features are constructed or named.")
        st.markdown("Please verify all input fields are correctly matched to model's expected features and ensure no custom feature names were misspelled.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.markdown("Please check your input values. If the issue persists, ensure your Python environment is set up correctly according to `requirements.txt`.")

st.markdown("---")
st.markdown("## About the Model")
st.markdown("""
- **Model Type:** Regression Model (specifically Gradient Boosting Regressor or XGBoost Regressor from training).
- **Dataset Period:** 2000-2015 for 193 countries.
- **Key Factors Considered:** Immunization, mortality rates, economic indicators (GDP, expenditure), social factors (schooling, alcohol consumption, BMI, thinness), and prevalence of diseases (HIV/AIDS, Measles).
- **Outlier Handling:** Outliers were identified using the IQR method and replaced with the column mean.
- **Feature Scaling:** All numerical features are scaled using `StandardScaler`.
""")

st.markdown("For more details, refer to the project's `README.md` and `train_and_save_model.py` files.")