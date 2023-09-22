import streamlit as st
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv('car_price_prediction_edit.csv')

# Load the trained machine learning model
try:
    predicted_model = joblib.load('lasso_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    # Create a nested dictionary for fuel type by Manufacturer -> Model -> Fuel Type
    fuel_type_dict = df.groupby('Manufacturer').apply(lambda x: x.groupby('Model')['Fuel_type'].unique()).to_dict()

    # Convert numpy arrays to lists for better compatibility
    for manufacturer, model_fuel_dict in fuel_type_dict.items():
        for model, fuel_types in model_fuel_dict.items():
            model_fuel_dict[model] = list(fuel_types)

    def main():
        st.title("Car Details Input")
        st.sidebar.header("Input Features")

        # Sort and display manufacturers
        sorted_manufacturers = sorted(df['Manufacturer'].unique())
        manufacturer = st.sidebar.selectbox("Manufacturer", sorted_manufacturers)

        # Filter and sort models based on manufacturer
        models_for_manufacturer = sorted(df[df['Manufacturer'] == manufacturer]['Model'].unique())
        model = st.sidebar.selectbox("Model", models_for_manufacturer)

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
            'Fuel_type': [fuel_type]
        }

        # Display the user input
        st.subheader("User Input Features")
        display_data = {
            'Manufacturer': manufacturer,
            'Model': model,
            'Produced Year': str(produced_year),  # Display without formatting
            'Fuel Type': fuel_type
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
except AttributeError as ae:
    st.error(f"AttributeError encountered: {ae}")
except Exception as e:
    st.error(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
