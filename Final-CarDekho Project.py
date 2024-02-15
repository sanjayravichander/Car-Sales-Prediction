#!/usr/bin/env python
# coding: utf-8

# In[96]:


## CarDekho Project - Final Project
import pandas as pd
cars_list=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_data.xlsx")


# In[2]:


cars_list.head(2)


# In[22]:


cars_list.columns


# In[97]:


cars_list=cars_list.drop(['Unnamed: 0','Color'],axis=1)


# In[24]:


cars_list.describe()


# In[25]:


cars_list.info()


# In[26]:


cars_list['Price_of_the_used_car'].value_counts()


# In[27]:


# Filter values with occurrence equal to 1
values_with_occurrence_one = cars_list['Price_of_the_used_car'].value_counts()
values_with_occurrence_one = values_with_occurrence_one[values_with_occurrence_one == 1]

# Display the values
print(values_with_occurrence_one)


# In[28]:


# Filter values with 'Crore' in the 'Price_of_the_used_car' column
crore_values = cars_list[cars_list['Price_of_the_used_car'].str.contains('Crore')]

# Display the filtered values
print(crore_values['Price_of_the_used_car'])


# In[98]:


import re

def convert_price(price_str):
    # Define a dictionary to map units to their numerical equivalents
    units = {'Lakh': 100000, 'Crore': 10000000}
    
    # Use regular expression to extract the numeric part and the unit
    match = re.match(r'₹ ([\d.]+) ([a-zA-Z]+)', price_str)
    
    if match:
        # Extract the numeric part and the unit
        numeric_part = float(match.group(1))
        unit = match.group(2)
        
        # Convert the value to its numerical equivalent
        return numeric_part * units.get(unit, 1)
    else:
        return None

# Apply the conversion function to the 'Price_of_the_used_car' column
cars_list['Price'] = cars_list['Price_of_the_used_car'].apply(convert_price)


# In[99]:


cars_list['Kilometers_driven'] = cars_list['Kilometers_driven'].str.replace(',', '')


# In[100]:


df_1=cars_list.drop(['Price_of_the_used_car'],axis=1).copy()


# In[101]:


df_1


# In[46]:


df_1.info()


# In[47]:


df_1.head(3)


# In[59]:


import plotly.express as px

# Plot the distribution of prices interactively
plt.figure(figsize=(6, 6))
fig = px.histogram(df_1, x='Price', title='Distribution of Prices')
fig.update_layout(xaxis_title='Price', yaxis_title='Frequency')
fig.show()


# In[112]:


import numpy as np

df_1['Body_type'] = df_1['Body_type'].fillna(df_1['Body_type'].mode()[0])

# Median imputation
df_1['Price'].fillna(df_1['Price'].median(), inplace=True)


# In[113]:


df_1.isnull().sum()/len(df_1)*100


# In[82]:


df_1.head(2)


# In[102]:


import plotly.express as px

# Create an interactive boxplot
fig = px.box(df_1, x='Kilometers_driven', y='Price', title='Boxplot of Kilometers Driven vs. Price')
fig.update_layout(xaxis_title='Kilometers Driven', yaxis_title='Price')
fig.show()


# In[93]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure and axes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Plot Price_of_the_used_car
sns.histplot(data=df_1, x='Price', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Price of the Used Car Distribution')

# Plot Kilometers_driven
sns.histplot(data=df_1, x='Kilometers_driven', kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Kilometers Driven Distribution')

# Plot Number_of_previous_owners
sns.countplot(data=df_1, x='Number_of_previous_owners', ax=axes[1, 0])
axes[1, 0].set_title('Number of Previous Owners Distribution')

# Plot Year_of_car_manufacture
sns.histplot(data=df_1, x='Year_of_car_manufacture', kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Year of Car Manufacture Distribution')

# Adjust layout
plt.tight_layout()
plt.show()


# In[91]:


import plotly.graph_objects as go

# Categorical features: Assuming you have categorical features in your DataFrame
categorical_features = ['city', 'Fuel_type', 'Body_type', 'Transmission_type']

# Set up the figure for categorical features
fig = go.Figure()

# Plot count plots for categorical features
for feature in categorical_features:
    if feature == 'Body_type':
        fig.add_trace(go.Bar(x=df_1[feature].value_counts().index, 
                             y=df_1[feature].value_counts().values,
                             name=f'Distribution of {feature}', marker_color='lightskyblue'))
    else:
        fig.add_trace(go.Bar(x=df_1[feature].value_counts().index, 
                             y=df_1[feature].value_counts().values,
                             name=f'Distribution of {feature}'))

# Update layout for categorical features
fig.update_layout(title='Distribution of Categorical Features',
                  xaxis_title='Category',
                  yaxis_title='Count')

# Show the interactive plot
fig.show()


# In[104]:


numerical_features = ['Price', 'Kilometers_driven', 'Number_of_previous_owners', 'Year_of_car_manufacture']

import plotly.graph_objects as go

# Calculate correlation matrix
correlation_matrix = df_1[numerical_features].corr()

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    colorscale='Viridis',
    colorbar=dict(title='Correlation'),
))


# In[ ]:





# In[109]:


sns.boxplot(data=df_1, x='Fuel_type', y='Price')
plt.xlabel('Fuel Type')
plt.ylabel('Price of Used Car')
plt.title('Fuel type by Number of Previous Owners')

plt.show()


# In[108]:


sns.boxplot(data=df_1, x='Transmission_type', y='Price')
plt.xlabel('Fuel Type')
plt.ylabel('Price of Used Car')
plt.title('Transmission type by Number of Previous Owners')

plt.show()


# In[119]:


df_1['Kilometers_driven'] = pd.to_numeric(df_1['Kilometers_driven'])


# In[120]:


# Data Quality Checking
def perform_data_quality_checks(data):
    # Check for duplicates
    duplicate_rows = data[data.duplicated()]
    if not duplicate_rows.empty:
        print("Duplicate Rows:")
        print(duplicate_rows)
    else:
        print("No duplicate rows found.")

    # Check for missing values
    missing_values = data.isnull().sum()
    if not missing_values.empty:
        print("\nMissing Values:")
        print(missing_values)
    else:
        print("\nNo missing values found.")

    # Check data types
    data_types = data.dtypes
    print("\nData Types:")
    print(data_types)

    # Additional checks can be added as needed

# Call the function with your DataFrame
perform_data_quality_checks(df_1)


# In[121]:


import plotly.graph_objects as go

# Select numerical columns
numeric_columns = df_1.select_dtypes(include=['number'])

# Calculate correlation matrix
correlation_with_price = numeric_columns.corr()['Price'].drop('Price')

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=correlation_with_price.values.reshape(1, -1),
    x=correlation_with_price.index,
    y=['Price'],
    colorscale='RdBu',
    colorbar=dict(title='Correlation'),
))

# Update layout
fig.update_layout(
    title='Correlation with Price',
    xaxis_title='Features',
    yaxis_title='Price',
)

# Show the interactive plot
fig.show()


# In[124]:


## ANOVA test for checking how categorical columns are relating to output
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Select categorical columns
categorical_columns = df_1.select_dtypes(include=['object'])

# Create a list to store results
anova_results = []

# Perform ANOVA for each categorical column
for column in categorical_columns:
    model = ols('Price ~ ' + column, data=df_1).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_value = anova_table.loc[column, 'PR(>F)']
    anova_results.append((column, p_value))

# Print ANOVA results
for column, p_value in anova_results:
    if p_value < 0.05:  # Significance level of 0.05
        print(f'{column}: p-value = {p_value} (Statistically significant)')
    else:
        print(f'{column}: p-value = {p_value} (Not statistically significant)')


# In[127]:


# Remove duplicate rows from the dataset
df_1 = df_1.drop_duplicates()

# Verify if duplicates are removed
print("Original dataset length:", len(cars_list))
print("After removing Duplicates:", len(cars_list))


# In[128]:


df_1.columns


# In[332]:


Inputs=df_1[['Car_model', 'Year_of_car_manufacture', 'Kilometers_driven',
       'Number_of_previous_owners', 'Transmission_type', 'Fuel_type',
       'Body_type','city']]
Output=df_1['Price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(Inputs,Output,test_size=0.3,random_state=32)


# In[333]:


# Check for empty rows in x_train
empty_rows_train = x_train.isnull().any(axis=1)

# Check for empty rows in x_test
empty_rows_test = x_test.isnull().any(axis=1)

# Display the count of empty rows in x_train and x_test
print("Number of empty rows in x_train:", empty_rows_train.sum())
print("Number of empty rows in x_test:", empty_rows_test.sum())


# In[334]:


import category_encoders as ce

# Create a target encoder
encoder = ce.TargetEncoder(cols=['Car_model'])

# Fit the encoder on the training data
encoder.fit(x_train, y_train)

# Transform the 'Car_model' column in both the training and testing sets
x_train['Car_model'] = encoder.transform(x_train)['Car_model']
x_test['Car_model'] = encoder.transform(x_test)['Car_model']


# In[335]:


# Check for empty rows in x_train
empty_rows_train = x_train.isnull().any(axis=1)

# Check for empty rows in x_test
empty_rows_test = x_test.isnull().any(axis=1)

# Display the count of empty rows in x_train and x_test
print("Number of empty rows in x_train:", empty_rows_train.sum())
print("Number of empty rows in x_test:", empty_rows_test.sum())


# In[336]:


# Replace 'Manual' with 0 and 'Automatic' with 1 in the 'Transmission_type' column in both training and testing sets
x_train['Transmission_type'].replace({'Manual': 0, 'Automatic': 1}, inplace=True)
x_test['Transmission_type'].replace({'Manual': 0, 'Automatic': 1}, inplace=True)


# In[337]:


# Check for empty rows in x_train
empty_rows_train = x_train.isnull().any(axis=1)

# Check for empty rows in x_test
empty_rows_test = x_test.isnull().any(axis=1)

# Display the count of empty rows in x_train and x_test
print("Number of empty rows in x_train:", empty_rows_train.sum())
print("Number of empty rows in x_test:", empty_rows_test.sum())


# In[338]:


# Define the mapping of fuel types to numerical values
fuel_type_mapping = {
    'Petrol': 1,
    'Diesel': 2,
    'Cng': 3,
    'Electric': 4,
    'Lpg': 5
}

# Apply mapping to the 'Fuel_type' column in the training set
x_train['Fuel_type'] = x_train['Fuel_type'].map(fuel_type_mapping)

# Apply mapping to the 'Fuel_type' column in the testing set
x_test['Fuel_type'] = x_test['Fuel_type'].map(fuel_type_mapping)


# In[339]:


x_train.head(3)


# In[340]:


# Check for empty rows in x_train
empty_rows_train = x_train.isnull().any(axis=1)

# Check for empty rows in x_test
empty_rows_test = x_test.isnull().any(axis=1)

# Display the count of empty rows in x_train and x_test
print("Number of empty rows in x_train:", empty_rows_train.sum())
print("Number of empty rows in x_test:", empty_rows_test.sum())


# In[341]:


import category_encoders as ce

# Calculate the mean target value for each category in the 'Body_type' feature using the training set
body_type_target_mean_train = y_train.groupby(x_train['Body_type']).mean()

# Create a target encoder
encoder = ce.TargetEncoder(cols=['Body_type'])

# Fit the encoder on the training data
encoder.fit(x_train, y_train)

# Transform the 'Body_type' column in both the training and testing sets
x_train['Body_type'] = encoder.transform(x_train)['Body_type']
x_test['Body_type'] = encoder.transform(x_test)['Body_type']


# In[342]:


# Check for empty rows in x_train
empty_rows_train = x_train.isnull().any(axis=1)

# Check for empty rows in x_test
empty_rows_test = x_test.isnull().any(axis=1)

# Display the count of empty rows in x_train and x_test
print("Number of empty rows in x_train:", empty_rows_train.sum())
print("Number of empty rows in x_test:", empty_rows_test.sum())


# In[343]:


city_frequency_train = x_train['city'].value_counts(normalize=True)

# Replace each category with its frequency in the training set
x_train['city'] = x_train['city'].map(city_frequency_train)

# Calculate the frequency of each category in the 'city' feature in the testing set
city_frequency_test = x_test['city'].value_counts(normalize=True)

# Replace each category with its frequency in the testing set
x_test['city'] = x_test['city'].map(city_frequency_train)


# In[345]:


# Check for empty rows in x_train
empty_rows_train = x_train.isnull().any(axis=1)

# Check for empty rows in x_test
empty_rows_test = x_test.isnull().any(axis=1)

# Display the count of empty rows in x_train and x_test
print("Number of empty rows in x_train:", empty_rows_train.sum())
print("Number of empty rows in x_test:", empty_rows_test.sum())


# In[346]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
col=['Kilometers_driven','Year_of_car_manufacture','Number_of_previous_owners']

# Combine x_training and x_testing data for Scaling
combined_scale_data = pd.concat([x_train[col], x_test[col]], axis=0)

# Fit the encoder on the combined data
scaler.fit(combined_scale_data)

# Encode both training and testing data
x_train_scaled = scaler.transform(x_train[col])
x_test_scaled = scaler.transform(x_test[col])

# Replace the encoded values in the DataFrames
x_train[col] = x_train_scaled
x_test[col] = x_test_scaled


# In[347]:


# Check for empty rows in x_train
empty_rows_train = x_train.isnull().any(axis=1)

# Check for empty rows in x_test
empty_rows_test = x_test.isnull().any(axis=1)

# Display the count of empty rows in x_train and x_test
print("Number of empty rows in x_train:", empty_rows_train.sum())
print("Number of empty rows in x_test:", empty_rows_test.sum())


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define regression models and their respective hyperparameter grids
Regression_Models = {
    'Linear Regression': (LinearRegression(), {}),
    'Ridge Regression': (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
    'Lasso Regression': (Lasso(), {'alpha': [0.1, 1.0, 10.0]}),
    'ElasticNet Regression': (ElasticNet(), {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}),
    'Huber Regressor': (HuberRegressor(), {'epsilon': [1.0, 1.5, 2.0],'max_iter':[1000]}),
    'Bayesian Ridge Regression': (BayesianRidge(), {}),
    'Decision Tree': (DecisionTreeRegressor(), {'criterion': ['friedman_mse'], 'max_depth': [None, 5, 10]}),
    'Random Forest': (RandomForestRegressor(), {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}),
    'Gradient Boosting': (GradientBoostingRegressor(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01], 'max_depth': [3, 5, 7]}),
    'XG Boost': (XGBRegressor(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01], 'max_depth': [3, 5, 7]}),
    #Support Vector Machine': (SVR(), {'kernel': ['linear', 'rbf'], 'C': [0.1, 1.0, 10.0]}),
    'K-Nearest Neighbors': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance']}),
}

for model_name, (model, param_grid) in Regression_Models.items():
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train.values.ravel())  # Convert y_train to a 1D array using ravel()
    
    # Predict on test set
    y_pred = grid_search.predict(x_test)
    
    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print(f"Model: {model_name}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Mean Squared Error: {grid_search.best_score_}")
    print(f"Test Mean Squared Error: {mse}")
    print(f"Test Mean Absolute Error: {mae}")
    print(f"Test R-squared: {r2}")
    print()


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures

# Initialize PolynomialFeatures to create polynomial features
poly_features = PolynomialFeatures(degree=3)

# Transform the features to polynomial features
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.transform(x_test)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(x_train_poly, y_train)

# Make predictions on the testing data
y_pred = model.predict(x_test_poly)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
print()


# In[261]:


import numpy as np
from sklearn.linear_model import LinearRegression

# Assuming x_train and y_train are your training data
# Assuming x_test and y_test are your testing data

# Generate quadratic features
x_train_quad = np.column_stack((x_train, x_train**2))
x_test_quad = np.column_stack((x_test, x_test**2))

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(x_train_quad, y_train)

# Make predictions on the testing data
y_pred = model.predict(x_test_quad)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
print()


# In[263]:


import numpy as np

# Assuming 'data' is your DataFrame containing the features with outliers
# Replace 'Kilometers_driven' and 'Price_of_the_used_car' with the actual column names

# Calculate the interquartile range (IQR) for each feature
Q1 = df_1['Kilometers_driven'].quantile(0.25)
Q3 = df_1['Kilometers_driven'].quantile(0.75)
IQR = Q3 - Q1

# Calculate the overall range of values for each feature
min_value = df_1['Kilometers_driven'].min()
max_value = df_1['Kilometers_driven'].max()
overall_range = max_value - min_value

# Calculate the ratio of IQR to overall range
outlier_magnitude_km = IQR / overall_range

print("Magnitude of outliers for Kilometers_driven:", outlier_magnitude_km)

# Repeat the same process for 'Price_of_the_used_car'
Q1 = df_1['Price'].quantile(0.25)
Q3 = df_1['Price'].quantile(0.75)
IQR = Q3 - Q1

min_value = df_1['Price'].min()
max_value = df_1['Price'].max()
overall_range = max_value - min_value

outlier_magnitude_price = IQR / overall_range

print("Magnitude of outliers for Price:", outlier_magnitude_price)


# In[293]:


xg=XGBRegressor(learning_rate= 0.01, max_depth= 7, n_estimators= 200)
xg.fit(x_train,y_train)
y_pred=xg.predict(x_test)

# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)

print(mae)
print(r2)


# In[295]:


import pickle
file="C:\\Users\\DELL\\Downloads\\Car_Price_Prediction\\XGB_Model.pkl"
with open(file,'wb') as f:
    pickle.dump(xg,f)
with open(file,'rb') as f:
    pickle.load(f)


# In[300]:


Inputs.to_csv('CarDheko.csv')


# In[298]:


Inputs.columns


# In[301]:


Inputs['Car_model'].value_counts()


# In[ ]:


cars_list['tran']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




