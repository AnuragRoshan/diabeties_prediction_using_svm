import streamlit as st
import numpy as np
import pickle

# Load the classifier and scaler
classifier = pickle.load(open('classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

def values(a, b, c, d, e, f, g, h):
    input_data = (a, b, c, d, e, f, g, h)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    print("Input Data (reshaped):", input_data_reshaped)  # Debug statement

    std_data = scaler.transform(input_data_reshaped)
    print("Standardized Data:", std_data)  # Debug statement

    prediction = classifier.predict(std_data)
    print("Prediction:", prediction)  # Debug statement

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.shutterstock.com/image-vector/abstract-medical-banner-background-template-260nw-2075486077.jpg");
        background-size: cover;
    }
    .input-label {
        font-size: 18px;
        color: black;
    }
    .stNumberInput > div > label {
        font-size: 56px;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title('Diabetes Prediction Form')

with st.form('diabetes_form'):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        a = st.number_input('Pregnancies')
    with col2:
        b = st.number_input('Glucose')
    with col3:
        c = st.number_input('BloodPressure')
    with col4:
        d = st.number_input('SkinThickness')

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        e = st.number_input('Insulin')
    with col6:
        f = st.number_input('BMI')
    with col7:
        g = st.number_input('DiabetesPedigreeFunction')
    with col8:
        h = st.number_input('Age')

    submit = st.form_submit_button('Submit')

if submit:
    result = values(a, b, c, d, e, f, g, h)
    st.success(result)
