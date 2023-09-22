import streamlit as st
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv('car_price_prediction_edit.csv')

# Load the trained machine learning model
predicted_model = joblib.load('lasso_model.pkl')

# Build a dynamic dictionary for drop-downs
dynamic_dict = df.groupby(['Manufacturer', 'Model', 'Category']).agg({
    'Fuel_type': lambda x: list(x.unique()),
    'Gear_type': lambda x: list(x.unique())
}).reset_index()

# Streamlit UI
def main():
    st.title("Car Price Prediction")
    st.sidebar.header("Input Features")

    manufacturer = st.sidebar.selectbox("Manufacturer", df['Manufacturer'].unique())
    filtered_df = dynamic_dict[dynamic_dict['Manufacturer'] == manufacturer]
    
    model = st.sidebar.selectbox("Model", filtered_df['Model'].unique())
    filtered_df = filtered_df[filtered_df['Model'] == model]

    category = st.sidebar.selectbox("Category", filtered_df['Category'].unique())
    filtered_df = filtered_df[filtered_df['Category'] == category]

    fuel_type = st.sidebar.selectbox("Fuel Type", filtered_df['Fuel_type'].explode().unique())
    gear_type = st.sidebar.selectbox("Gear Type", filtered_df['Gear_type'].explode().unique())

    produced_year = st.sidebar.slider("Produced Year", min_value=2000, max_value=2023, value=2010, step=1)

    user_input = {
        'Manufacturer': manufacturer,
        'Model': model,
        'Category': category,
        'Fuel_type': fuel_type,
        'Gear_type': gear_type,
        'Produced_year': produced_year
    }

    st.subheader("User Input:")
    st.write(pd.DataFrame([user_input]))

    # Make prediction
    try:
        prediction = predicted_model.predict(pd.DataFrame([user_input]))
        st.subheader("Predicted Price:")
        st.write(f"The predicted price is ${int(prediction[0])}")
    except Exception as e:
        st.write("Error in prediction: ", e)

if __name__ == '__main__':
    main()
