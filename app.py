import streamlit as st
import pandas as pd
import joblib


# Load the dataset
df = pd.read_csv('car_price_prediction_edit.csv')

# Load the trained machine learning model
predicted_model = joblib.load('lasso_model.pkl')

# Group by 'Manufacturer' and then get unique values for each group
manufacturer_to_model = df.groupby('Manufacturer')['Model'].unique().to_dict()
manufacturer_to_fuel = df.groupby('Manufacturer')['Fuel_type'].unique().to_dict()
manufacturer_to_gear = df.groupby('Manufacturer')['Gear_type'].unique().to_dict()

# Convert numpy arrays to lists for better compatibility
for dictionary in [manufacturer_to_model, manufacturer_to_fuel, manufacturer_to_gear]:
    for key, value in dictionary.items():
        dictionary[key] = list(value)

# Streamlit UI
def main():
    st.title("Car Price Prediction")

    # Sidebar with feature input
    st.sidebar.header("Input Features")

    # Manufacturer Selection
    manufacturer = st.sidebar.selectbox("Manufacturer", df['Manufacturer'].unique())

    # Based on Manufacturer, display the Models
    model = st.sidebar.selectbox("Model", manufacturer_to_model.get(manufacturer, []))

    # Based on Manufacturer, display the Fuel Types and Gear Types
    fuel_type = st.sidebar.selectbox("Fuel Type", manufacturer_to_fuel.get(manufacturer, []))
    gear_type = st.sidebar.selectbox("Gear Type", manufacturer_to_gear.get(manufacturer, []))

    # Produced Year
    produced_year = st.sidebar.slider("Produced Year", min_value=2000, max_value=2023, value=2010, step=1)

    # Displaying the user input for Streamlit view
    display_data = {
        'Manufacturer': manufacturer,
        'Model': model,
        'Produced Year': produced_year,
        'Fuel Type': fuel_type,
        'Gear Type': gear_type
    }
    st.subheader("User Input Features")
    st.write(pd.DataFrame([display_data]))

    # If any of the features are empty, show a message and do not proceed with prediction
    if "" in display_data.values():
        st.warning("Please complete all the fields for prediction.")
    else:
        data_for_prediction = {
            'Manufacturer': [manufacturer],
            'Model': [model],
            'Produced_year': [produced_year],
            'Fuel_type': [fuel_type],
            'Gear_type': [gear_type]
        }
        predicted_price = predicted_model.predict(pd.DataFrame(data_for_prediction))

        # Display the prediction in the Streamlit app
        st.subheader('Predicted Price')
        st.success("The estimated price of your car is ${}".format(int(predicted_price[0])))


if __name__ == '__main__':
    main()
