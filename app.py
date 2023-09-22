import streamlit as st
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv('car_price_prediction_edit.csv')

# Load the trained machine learning model
predicted_model = joblib.load('lasso_model.pkl')

# Create nested dictionary for fuel type by Manufacturer -> Model -> Fuel Type
fuel_type_dict = {}
for _, row in df.iterrows():
    manufacturer = row['Manufacturer']
    model = row['Model']
    fuel_type = row['Fuel_type']
    if manufacturer not in fuel_type_dict:
        fuel_type_dict[manufacturer] = {}
    if model not in fuel_type_dict[manufacturer]:
        fuel_type_dict[manufacturer][model] = []
    fuel_type_dict[manufacturer][model].append(fuel_type)
    fuel_type_dict[manufacturer][model] = list(set(fuel_type_dict[manufacturer][model]))

def main():
    st.title("Car Details Input")
    st.sidebar.header("Input Features")

    # Sort and display manufacturers
    sorted_manufacturers = sorted(df['Manufacturer'].unique())
    manufacturer = st.sidebar.selectbox("Manufacturer", sorted_manufacturers)

    # Filter and sort models based on manufacturer
    models_for_manufacturer = sorted(df[df['Manufacturer'] == manufacturer]['Model'].unique())
    model = st.sidebar.selectbox("Model", models_for_manufacturer)
    
    # Category and Gear Type
    category = st.sidebar.selectbox("Category", df['Category'].unique())
    gear_type = st.sidebar.selectbox("Gear Type", df['Gear_type'].unique())

    # Filter and sort fuel types based on manufacturer and model
    fuel_types_for_model = sorted(fuel_type_dict.get(manufacturer, {}).get(model, []))
    fuel_type = st.sidebar.selectbox("Fuel Type", fuel_types_for_model)

    # Year selection
    produced_year = st.sidebar.slider("Produced Year", min_value=2000, max_value=2023, value=2010, step=1)

    # Create a dataframe with user input and make prediction
    data_for_prediction = {
        'Manufacturer': [manufacturer],
        'Model': [model],
        'Produced_year': [produced_year],
        'Category': [category],
        'Fuel_type': [fuel_type],
        'Gear_type': [gear_type]
    }

    # Display the user input
    st.subheader("User Input Features")
    display_data = {
        'Manufacturer': manufacturer,
        'Model': model,
        'Produced Year': str(produced_year),  # Remove any commas
        'Category': category,
        'Fuel Type': fuel_type,
        'Gear Type': gear_type
    }
    st.write(pd.DataFrame([display_data]))

    try:
        predicted_price = predicted_model.predict(pd.DataFrame(data_for_prediction))
        st.subheader('Predicted Price')
        st.success(f"The estimated price of your car is ${int(predicted_price[0])}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
