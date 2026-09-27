import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load(open('linear_regression_model.sav','rb'))

st.title('Insurance Charges Prediction App')
st.write('Enter the details below to predict the insurance charges.')

# Input fields for numerical features
age = st.slider('Age', 18, 100, 30)
bmi = st.slider('BMI', 10.0, 60.0, 25.0)
children = st.slider('Number of Children', 0, 5, 0)

# Input fields for categorical features
sex = st.selectbox('Sex', ['female', 'male'])
smoker = st.selectbox('Smoker', ['no', 'yes'])
region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

# Prepare input data for the model
def preprocess_input(age, bmi, children, sex, smoker, region):
    # Create a DataFrame with the same columns as X_train used during training
    # Initialize all one-hot encoded columns to False
    input_data = pd.DataFrame(np.zeros((1, 8)), columns=[
        'age', 'bmi', 'children', 'sex_male', 'smoker_yes',
        'region_northwest', 'region_southeast', 'region_southwest'
    ])

    input_data['age'] = age
    input_data['bmi'] = bmi
    input_data['children'] = children

    # One-hot encode sex
    if sex == 'male':
        input_data['sex_male'] = True

    # One-hot encode smoker
    if smoker == 'yes':
        input_data['smoker_yes'] = True

    # One-hot encode region
    if region == 'northwest':
        input_data['region_northwest'] = True
    elif region == 'southeast':
        input_data['region_southeast'] = True
    elif region == 'southwest':
        input_data['region_southwest'] = True
    # 'northeast' will have all region_ columns as False (reference category)

    return input_data

# When the user clicks the Predict button
if st.button('Predict Charges'):
    # Preprocess the input data
    processed_input = preprocess_input(age, bmi, children, sex, smoker, region)

    # Make prediction
    prediction = model.predict(processed_input)

    st.success(f'Predicted Insurance Charges: ${prediction[0]:,.2f}')
