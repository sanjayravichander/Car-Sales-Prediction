# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:46:13 2024

@author: DELL
"""

import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
with open('C:\\Users\\DELL\\Downloads\\Car_Price_Prediction\\GB_Model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset
data = pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDheko_df.xlsx")

# Preprocess the data
# Assuming you have already performed preprocessing steps such as handling missing values and encoding categorical variables

# Define a function to predict car price
def predict_price(car_data):
    X = pd.DataFrame(car_data, index=[0])
    prediction = model.predict(X)
    return prediction[0]

# Streamlit app
import base64
def main():
    st.title('Car Price Prediction')

    # Adding Background Image
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    # Streamlit App
    def run():
        # Set background image
        add_bg_from_local('C:/Users/DELL/Downloads/Car_Rent.jpeg')
        
    # Show dataset
    if st.checkbox('Show Dataset'):
        st.write(data)

# Sidebar inputs
st.sidebar.header('Enter Car Details')

# Collect user inputs
car_model_options = data['Car_model'].unique()
car_model_mapping = {model: idx for idx, model in enumerate(car_model_options)}
car_model = st.sidebar.selectbox('Car Model', car_model_options)

year_of_manufacture = st.sidebar.number_input('Year of Manufacture', min_value=1900, max_value=2024, value=2022)
kilometers_driven = st.sidebar.number_input('Kilometers Driven', min_value=0, value=0)
num_previous_owners = st.sidebar.number_input('Number of Previous Owners', min_value=0, value=0)

transmission_type_mapping = {'Manual': 0, 'Automatic': 1}
transmission_type_options = list(transmission_type_mapping.keys())
transmission_type = st.sidebar.selectbox('Transmission Type', transmission_type_options)

fuel_type_options = data['Fuel_type'].unique()
fuel_type_mapping = {fuel: idx for idx, fuel in enumerate(fuel_type_options)}
fuel_type = st.sidebar.selectbox('Fuel Type', fuel_type_options)

body_type_options = data['Body_type'].unique()
body_type_mapping = {body: idx for idx, body in enumerate(body_type_options)}
body_type = st.sidebar.selectbox('Body Type', body_type_options)

city_options = data['city'].unique()
city_mapping = {city: idx for idx, city in enumerate(city_options)}
city = st.sidebar.selectbox('City', city_options)

def format_price(price):
    if price >= 1e8:  # Crore
        return f'{price / 1e7:.2f} Crore'
    elif price >= 1e5:  # Lakh
        return f'{price / 1e5:.2f} Lakhs'
    elif price >= 1e3:  # Thousand
        return f'{price / 1e3:.2f} K'
    else:
        return str(price)

# Predicting price
if st.sidebar.button('Predict'):
    user_inputs = {
        'Car_model': car_model_mapping[car_model],
        'Year_of_car_manufacture': year_of_manufacture,
        'Kilometers_driven': kilometers_driven,
        'Number_of_previous_owners': num_previous_owners,
        'Transmission_type': transmission_type_mapping[transmission_type],
        'Fuel_type': fuel_type_mapping[fuel_type],
        'Body_type': body_type_mapping[body_type],
        'city': city_mapping[city]
    }
    price_prediction = predict_price(user_inputs)
    formatted_price = format_price(price_prediction)
    st.sidebar.subheader(f'Predicted Price: {formatted_price}')

if __name__ == '__main__':
    main()
