import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Function to filter categorical features based on their higher-level categories
def filter_categories(row, mappings):
    new_row = row.copy()
    for higher_cat, lower_cats in mappings.items():
        if row[higher_cat] in lower_cats:
            if row[lower_cats[higher_cat]] not in lower_cats[row[higher_cat]]:
                new_row[lower_cats[higher_cat]] = 'Other'
    return new_row

# 1. Load the dataset
df = pd.read_csv("car_price_prediction.csv")

# 2. Data Preprocessing
# Rename columns
new_columns = {
    'Prod. year': 'Produced_year',
    'Gear box type': 'Gear_type',
    'Fuel type': 'Fuel_type'
}
df = df.rename(columns=new_columns)

# Map higher-level categories to lower-level ones dynamically based on data
manufacturer_to_model = df.groupby('Manufacturer')['Model'].apply(set).apply(list).to_dict()
manufacturer_to_fuel = df.groupby('Manufacturer')['Fuel_type'].apply(set).apply(list).to_dict()
manufacturer_to_gear = df.groupby('Manufacturer')['Gear_type'].apply(set).apply(list).to_dict()

# Create a mapping dictionary
mappings = {
    'Manufacturer': {
        'Model': manufacturer_to_model,
        'Fuel_type': manufacturer_to_fuel,
        'Gear_type': manufacturer_to_gear
    },
    # Add other higher-level categories here
}

# Handle missing values (you can change this strategy)
df.dropna(inplace=True)

# Update categorical features dynamically based on higher-level categories
df = df.apply(lambda row: filter_categories(row, mappings), axis=1)

# Define features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Define numeric and categorical columns
numeric_features = ['Produced_year']
categorical_features = ['Manufacturer', 'Model', 'Category', 'Fuel_type', 'Gear_type']

# Create transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define and train the model
# Use ColumnTransformer to apply the transformations to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that first applies the column transformer and then fits the model
lasso_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=1.0))
])

# Train the model
lasso_model.fit(X_train, y_train)

# 5. Save the model
joblib.dump(lasso_model, 'lasso_model.pkl')

# Save the processed dataframe
df.to_csv("car_price_prediction_edit.csv", index=False)
