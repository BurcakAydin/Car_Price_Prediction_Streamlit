import streamlit as st
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv('car_price_prediction_edit.csv')

# Load the trained machine learning model
predicted_model = joblib.load('lasso_model.pkl')

# Create dictionaries to map Manufacturer, Model, and Category to Fuel Type and Gear Type
unique_fuel_and_gear = df.groupby(['Manufacturer', 'Model', 'Category'])[['Fuel_type', 'Gear_type']].first().reset_index()
fuel_type_dict = unique_fuel_and_gear.groupby(['Manufacturer', 'Model', 'Category'])['Fuel_type'].unique().to_dict()
gear_type_dict = unique_fuel_and_gear.groupby(['Manufacturer', 'Model', 'Category'])['Gear_type'].unique().to_dict()

# Alphabetical order for Manufacturer and Category
manufacturers = sorted(df['Manufacturer'].unique())
categories = sorted(df['Category'].unique())

def main():
    st.title("Car Details Input")
    
    st.sidebar.header("Input Features")
    
    manufacturer = st.sidebar.selectbox("Manufacturer", manufacturers)
    
    models = sorted(df[df['Manufacturer'] == manufacturer]['Model'].unique())
    model = st.sidebar.selectbox("Model", models)
    
    category = st.sidebar.selectbox("Category", categories)
    
    fuel_type = st.sidebar.selectbox(
        "Fuel Type",
        fuel_type_dict.get((manufacturer, model, category), ["Unknown"])
    )
    
    gear_type = st.sidebar.selectbox(
        "Gear Type",
        gear_type_dict.get((manufacturer, model, category), ["Unknown"])
    )
    
    produced_year = st.sidebar.slider("Produced Year", min_value=2000, max_value=2023, value=2010, step=1)
    
    display_data = {
        'Manufacturer': manufacturer,
        'Model': model,
        'Produced Year': str(produced_year),
        'Category': category,
        'Fuel Type': fuel_type,
        'Gear Type': gear_type
    }
    
    st.subheader("User Input Features")
    st.table(pd.DataFrame([display_data]))  # Hide index by using st.table
    
    data_for_prediction = {
        'Manufacturer': [manufacturer],
        'Model': [model],
        'Produced_year': [produced_year],
        'Category': [category],
        'Fuel_type': [fuel_type],
        'Gear_type': [gear_type]
    }
    
    predicted_price = predicted_model.predict(pd.DataFrame(data_for_prediction))
    
    st.subheader('Predicted Price')
    st.success(f"The estimated price of your car is ${int(predicted_price[0])}.")

if __name__ == '__main__':
    main()
