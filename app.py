import streamlit as st
import joblib
import pandas as pd
from streamlit_lottie import st_lottie

# Load the trained model
model_path = r"C:\Users\madha\Desktop\projects\HEART_ML\\"
model_filename = "model.pkl"
random_forest_model = joblib.load(model_path + model_filename)

animation_url_0 = "https://lottie.host/4652fffe-dd14-47d2-a020-a1aeca6cd2ad/G2C36DFnkY.json"
animation_url_1 = "https://lottie.host/319953c8-9024-456f-955a-73c003d29e0f/mYtPv3ODnR.json"

def load_lottieurl(url: str):
            import requests
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

def make_prediction(input_data):
    # Ensure the input data has the same format as used during training
    input_df = pd.DataFrame(input_data, index=[0])

    # Ensure the order of columns matches the one used during training
    selected_features_for_prediction = ['Age', 'Cholesterol', 'Heart Rate',
                                         'Diabetes', 'Family History', 'Smoking', 'Obesity',
                                         'Alcohol Consumption', 'Exercise Hours Per Week',
                                         'Diet',
                                         'Previous Heart Problems', 'Medication Use', 'Stress Level',
                                         'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
                                         'Physical Activity Days Per Week', 'Sleep Hours Per Day',
                                         'BP_Systolic', 'BP_Diastolic', 'Sex_Female', 'Sex_Male']

    # Select only the features used during training in the prediction data
    input_df_for_prediction = input_df[selected_features_for_prediction]

    # Make prediction for the single input
    prediction = random_forest_model.predict(input_df_for_prediction)
    
    return prediction[0]


# Streamlit app
def main():
    st.title("Heart Disease Risk Prediction")
    st.sidebar.header("User Input")

    diet_mapping = {'Average': 0, 'Healthy': 1, 'Unhealthy': 2}

    # Collect user input
    age = st.sidebar.slider("Age", 20, 100, 40)
    cholesterol = st.sidebar.slider("Cholesterol", 100, 300, 200)
    heart_rate = st.sidebar.slider("Heart Rate", 50, 150, 75)
    diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
    family_history = st.sidebar.selectbox("Family History", [0, 1])
    smoking = st.sidebar.selectbox("Smoking", [0, 1])
    obesity = st.sidebar.selectbox("Obesity", [0, 1])
    alcohol_consumption = st.sidebar.slider("Alcohol Consumption", 0.0, 10.0, 5.0)
    exercise_hours_per_week = st.sidebar.slider("Exercise Hours Per Week", 0.0, 20.0, 5.0)
    diet = st.sidebar.selectbox("Diet", ['Average', 'Healthy', 'Unhealthy'])
    previous_heart_problems = st.sidebar.selectbox("Previous Heart Problems", [0, 1])
    medication_use = st.sidebar.selectbox("Medication Use", [0, 1])
    stress_level = st.sidebar.slider("Stress Level", 0, 10, 5)
    sedentary_hours_per_day = st.sidebar.slider("Sedentary Hours Per Day", 0.0, 24.0, 8.0)
    income = st.sidebar.slider("Income", 0, 500000, 50000)
    bmi = st.sidebar.slider("BMI", 10.0, 40.0, 25.0)
    triglycerides = st.sidebar.slider("Triglycerides", 50, 500, 150)
    physical_activity_days_per_week = st.sidebar.slider("Physical Activity Days Per Week", 0, 7, 3)
    sleep_hours_per_day = st.sidebar.slider("Sleep Hours Per Day", 0, 12, 7)
    bp_systolic = st.sidebar.slider("Blood Pressure - Systolic", 80, 200, 120)
    bp_diastolic = st.sidebar.slider("Blood Pressure - Diastolic", 50, 150, 80)
    sex_female = st.sidebar.selectbox("Sex - Female", [0, 1])
    sex_male = st.sidebar.selectbox("Sex - Male", [0, 1])

    
    
    if st.button("Predict"):
        # Package user inputs into a dictionary
        user_input = {
        'Age': age,
        'Cholesterol': cholesterol,
        'Heart Rate': heart_rate,
        'Diabetes': diabetes,
        'Family History': family_history,
        'Smoking': smoking,
        'Obesity': obesity,
        'Alcohol Consumption': alcohol_consumption,
        'Exercise Hours Per Week': exercise_hours_per_week,
        'Diet':diet_mapping[diet],
        'Previous Heart Problems': previous_heart_problems,
        'Medication Use': medication_use,
        'Stress Level': stress_level,
        'Sedentary Hours Per Day': sedentary_hours_per_day,
        'Income': income,
        'BMI': bmi,
        'Triglycerides': triglycerides,
        'Physical Activity Days Per Week': physical_activity_days_per_week,
        'Sleep Hours Per Day': sleep_hours_per_day,
        'BP_Systolic': bp_systolic,
        'BP_Diastolic': bp_diastolic,
        'Sex_Female': sex_female,
        'Sex_Male': sex_male,
    }


        # Make prediction and display the result
        prediction = make_prediction(user_input)
        if prediction == 0:
            lottie_visual = load_lottieurl(animation_url_0)
            st_lottie(lottie_visual)
        else:
            lottie_visual = load_lottieurl(animation_url_1)
            st_lottie(lottie_visual)


if __name__ == '__main__':
    main()
