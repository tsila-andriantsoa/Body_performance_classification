import streamlit as st
import requests
import base64
import joblib
import pandas as pd

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
    
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_image}");
        background-size: cover;
    }}
    
    h1, p {{
        color: #FFFFFF;  /* Set font color for inputs */
    }}
    .stTextInput, .stNumberInput, .stSelectbox, .stButton {{
        color: #FFFFFF;  /* Set font color for inputs */
    }}
    
    .stButton button {{
        background-color: #4CAF50;  /* Button background color */
        color: white;  /* Button text color */
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }}
    
    .stButton button:hover {{
        background-color: #45a049;  /* Button hover color */
    }}
    </style>
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the model and utilities 
loaded_pipeline = joblib.load('model/pipeline_baseline.pkl')    
    
# Set background image
set_background("img/background.jpg")

st.title("Body Performance Classification")

# Define layout with two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    body_fat = st.number_input("Body Fat (%)", min_value=1.0, max_value=50.0, value=20.0)
    diastolic = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=130, value=80)

with col2:
    systolic = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
    grip_force = st.number_input("Grip Force", min_value=10.0, max_value=100.0, value=50.0)
    sit_and_bend_forward = st.number_input("Sit and Bend Forward (cm)", min_value=-20.0, max_value=50.0, value=10.0)
    sit_ups_counts = st.number_input("Sit-ups Count", min_value=0, max_value=100, value=30)
    broad_jump = st.number_input("Broad Jump (cm)", min_value=50, max_value=300, value=150)

# Button to make prediction
if st.button("Predict Performance"):
    input_data = {
        "age": age,
        "gender": gender,
        "height_cm": height,
        "weight_kg": weight,
        "body_fat_%": body_fat,
        "diastolic": diastolic,
        "systolic": systolic,
        "gripForce": grip_force,
        "sit_and_bend_forward_cm": sit_and_bend_forward,
        "sit_ups_counts": sit_ups_counts,
        "broad_jump_cm": broad_jump
    }
    try:
        # Create dataframe based on JSON object
        df = pd.DataFrame(input_data, index = [0])       

        # rename data columns
        new_col_names = ['age', 'gender', 'height_cm', 'weight_kg', 'body_fat_%', 'diastolic',
           'systolic', 'gripForce', 'sit_and_bend_forward_cm', 'sit_ups_counts', 'broad_jump_cm', ]

        df.columns = new_col_names

        # Make a prediction
        prediction = loaded_pipeline.predict(df)[0]
        
        st.success(f"Predicted Performance: {prediction}")
        
    except Exception as e:
        
        st.error("Error in prediction.")
