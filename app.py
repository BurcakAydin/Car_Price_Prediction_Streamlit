import streamlit as st
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv('car_price_prediction_edit.csv')

# Load the trained machine learning model
predicted_model = joblib.load('lasso_model.pkl')

# Group by 'Manufacturer' to get unique 'Model', 'Category', 'Fuel_type', 'Gear_type'
model_dict = df.groupby('Manufacturer')['Model'].unique().to_dict()
category_dict = df.groupby(['Manufacturer', 'Model'])['Category'].unique().to_dict()
fuel_dict = df.groupby(['Manufacturer', 'Model', 'Category'])['Fuel_type'].unique().to_dict()
gear_dict = df.groupby(['Manufacturer', 'Model', 'Category'])['Gear_type'].unique().to_dict()

# Streamlit UI
def main():
    st.title("Car Details Input")
    st.sidebar.header("Input Features")

    # Sort and display manufacturers
    sorted_manufacturers = sorted(df['Manufacturer'].unique())
    manufacturer = st.sidebar.selectbox("Manufacturer", sorted_manufacturers)

    # Filter and sort models based on manufacturer
    models_for_manufacturer = sorted(model_dict.get(manufacturer, []))
    model = st.sidebar.selectbox("Model", models_for_manufacturer)

    # Filter and sort categories based on manufacturer and model
    categories_for_model = sorted(category_dict.get((manufacturer, model), []))
    category = st.sidebar.selectbox("Category", categories_for_model)

    # Filter and sort Fuel Type and Gear Type based on Manufacturer, Model, and Category
    fuel_types_for_category = ['Electric'] if manufacturer == "Tesla" else sorted(fuel_dict.get((manufacturer, model, category), []))
    gear_types_for_category = sorted(gear_dict.get((manufacturer, model, category), []))

    fuel_type = st.sidebar.selectbox("Fuel Type", fuel_types_for_category)
    gear_type = st.sidebar.selectbox("Gear Type", gear_types_for_category)

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
        'Produced Year': str(produced_year),
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
