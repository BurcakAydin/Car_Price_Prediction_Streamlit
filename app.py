import streamlit as st
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv('car_price_prediction_edit.csv')

# Load the trained machine learning model
predicted_model = joblib.load('lasso_model.pkl')

# Create dynamic choices
model_dict = df.groupby('Manufacturer')['Model'].unique().apply(list).to_dict()
category_dict = df.groupby('Model')['Category'].unique().apply(list).to_dict()
fuel_type_dict = df.groupby(['Manufacturer', 'Model'])['Fuel_type'].unique().apply(list).to_dict()
gear_type_dict = df.groupby(['Manufacturer', 'Model'])['Gear_type'].unique().apply(list).to_dict()

# Streamlit UI
def main():
    st.title("Car Details Input")
    st.sidebar.header("Input Features")

    # Manufacturer Selection
    manufacturer = st.sidebar.selectbox("Manufacturer", sorted(df['Manufacturer'].unique()))

    # Model Selection
    model = st.sidebar.selectbox("Model", sorted(model_dict.get(manufacturer, [])))

    # Category Selection
    category = st.sidebar.selectbox("Category", sorted(category_dict.get(model, [])))

    # Fuel Type and Gear Type Selection based on Manufacturer and Model
    fuel_type = st.sidebar.selectbox("Fuel Type", sorted(fuel_type_dict.get((manufacturer, model), [])))
    gear_type = st.sidebar.selectbox("Gear Type", sorted(gear_type_dict.get((manufacturer, model), [])))

    # Produced Year Slider
    produced_year = st.sidebar.slider("Produced Year", min_value=2000, max_value=2023, value=2010, step=1)

    # Displaying user input
    st.subheader("User Input Features")
    display_data = pd.DataFrame({
        'Manufacturer': [manufacturer],
        'Model': [model],
        'Produced Year': [produced_year],
        'Category': [category],
        'Fuel Type': [fuel_type],
        'Gear Type': [gear_type]
    })

    st.write(display_data.assign(hack='').set_index('hack'))

    # Prediction
    data_for_prediction = {
        'Manufacturer': [manufacturer],
        'Model': [model],
        'Produced_year': [produced_year],
        'Category': [category],
        'Fuel_type': [fuel_type],
        'Gear_type': [gear_type]
    }
    df_for_prediction = pd.DataFrame.from_dict(data_for_prediction)
    predicted_price = predicted_model.predict(df_for_prediction)

    st.subheader('Predicted Price')
    st.success(f"The estimated price of your car is ${int(predicted_price[0])}.")

if __name__ == '__main__':
    main()
