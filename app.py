import streamlit as st
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv('car_price_prediction_edit.csv')

# Load the trained machine learning model
predicted_model = joblib.load('lasso_model.pkl')

# Streamlit UI
def main():
    st.title("Car Details Input")
    st.sidebar.header("Input Features")

    # Sort manufacturers and display them
    sorted_manufacturers = sorted(df['Manufacturer'].unique())
    manufacturer = st.sidebar.selectbox("Manufacturer", sorted_manufacturers)
    
    # Filter models based on manufacturer and sort them
    models_for_manufacturer = sorted(df[df['Manufacturer'] == manufacturer]['Model'].unique())
    model = st.sidebar.selectbox("Model", models_for_manufacturer)
    
    # Filter and sort categories based on model
    categories_for_model = sorted(df[df['Model'] == model]['Category'].unique())
    category = st.sidebar.selectbox("Select Category", categories_for_model)
    
    # Filter and sort fuel types and gear types based on category
    fuel_types_for_category = sorted(df[df['Category'] == category]['Fuel_type'].unique())
    fuel_type = st.sidebar.selectbox("Fuel Type", fuel_types_for_category)
    
    gear_types_for_category = sorted(df[df['Category'] == category]['Gear_type'].unique())
    gear_type = st.sidebar.selectbox("Gear Type", gear_types_for_category)
    
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
    st.write(pd.DataFrame(data_for_prediction))

    try:
        predicted_price = predicted_model.predict(pd.DataFrame(data_for_prediction))
        st.subheader('Predicted Price')
        st.success(f"The estimated price of your car is ${int(predicted_price[0])}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
