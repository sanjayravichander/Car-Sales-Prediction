#!/usr/bin/env python
# coding: utf-8

# In[1]:


# CarDekho Price Prediction

# Importing Chennai Cars
import pandas as pd
chennai_cars=pd.read_excel("C:\\Users\\DELL\\Downloads\\chennai_cars.xlsx")


# In[2]:


chennai_cars


# In[3]:


chennai_cars['new_car_detail'][0]#['it']


# In[4]:


import ast  # For evaluating the string representation of dictionaries

# Convert string representation of dictionaries to actual dictionaries
chennai_cars['new_car_detail'] = chennai_cars['new_car_detail'].apply(ast.literal_eval)

# Normalize the 'new_car_detail' column into separate columns
chennai_cars_normalized = pd.json_normalize(chennai_cars['new_car_detail'])

# Concatenate the normalized columns with the original DataFrame
chennai_cars_df = pd.concat([chennai_cars, chennai_cars_normalized], axis=1)

# Drop the original 'new_car_detail' column
chennai_cars_df = chennai_cars_df.drop('new_car_detail', axis=1)


# In[5]:


chennai_cars['new_car_detail'][0]#['it']#['trendingText']['imgUrl']


# In[6]:


chennai_cars_df.head(2)


# In[7]:


# Convert string representation of dictionaries to actual dictionaries
chennai_cars_df['new_car_overview']=chennai_cars_df['new_car_overview'].apply(ast.literal_eval)

# Normalize the 'new_car_detail' column into separate columns
chennai_cars_normalized = pd.json_normalize(chennai_cars_df['new_car_overview'])

# Concatenate the normalized columns with the original DataFrame
chennai_cars_df = pd.concat([chennai_cars_df, chennai_cars_normalized], axis=1)

# Drop the original 'new_car_detail' column
chennai_cars_df = chennai_cars_df.drop('new_car_overview', axis=1)


# In[8]:


chennai_cars_df.head(2)


# In[9]:


# For checking purpose
chennai_cars_df['top'][0][0]['icon']#['key']#['value']['icon']


# In[10]:


# Creating a new Dataframe to store required values
chennai_df_1 = {"Car_Info": [], "Values_for_Car_Info": [], "Image_link_of_car": []}

for index, row in chennai_cars_df.iterrows():
    for item in row['top']:
        key = item['key']
        values = item['value']
        icon = item['icon']
        chennai_df_1["Car_Info"].append(key)
        chennai_df_1["Values_for_Car_Info"].append(values)
        chennai_df_1["Image_link_of_car"].append(icon)

# Convert the dictionary to a DataFrame
chennai_df_1 = pd.DataFrame(chennai_df_1)


# In[11]:


chennai_df_1.head(5)


# In[12]:


chennai_cars_df = chennai_cars_df.drop(['top'], axis=1)
chennai_cars_df = chennai_cars_df.drop(['bottomData','heading'], axis=1)


# In[13]:


# Convert string representation of dictionaries to actual dictionaries
chennai_cars_df['new_car_feature']=chennai_cars_df['new_car_feature'].apply(ast.literal_eval)

# Normalize the 'new_car_detail' column into separate columns
chennai_cars_normalized = pd.json_normalize(chennai_cars_df['new_car_feature'])

# Concatenate the normalized columns with the original DataFrame
chennai_cars_df = pd.concat([chennai_cars_df, chennai_cars_normalized], axis=1)

# Drop the original 'new_car_detail' column
chennai_cars_df = chennai_cars_df.drop('new_car_feature', axis=1)


# In[14]:


chennai_cars_df.head(2)


# In[15]:


# For checking purpose
chennai_cars_df['top']#[0]['value']


# In[16]:


# Creating a new Dataframe to store required values
chennai_df_2 = { "Features": []}

for index, row in chennai_cars_df.iterrows():
    for item in row['top']:
        values = item['value']
        chennai_df_2["Features"].append(values)

# Convert the dictionary to a DataFrame
chennai_df_2 = pd.DataFrame(chennai_df_2)


# In[17]:


chennai_df_2.head(2)


# In[18]:


# For checking purpose
chennai_cars_df['data'][0]#['list'][0]


# In[19]:


# Creating a new Dataframe to store required values
chennai_df_3 = {"Comfort_and_Convenience": [], "Comfort_and_Convenience_type": []}

for index, row in chennai_cars_df.iterrows():
    for item in row['data']:
        heading = item['heading']
        values = item['list'][0]['value']
        chennai_df_3["Comfort_and_Convenience"].append(heading)
        chennai_df_3["Comfort_and_Convenience_type"].append(values)

# Convert the dictionary to a DataFrame
chennai_df_3 = pd.DataFrame(chennai_df_3)


# In[20]:


chennai_df_3.head(2)


# In[21]:


# Drop the original 'new_car_detail' column
chennai_cars_df = chennai_cars_df.drop(['top','data'], axis=1)


# In[22]:


# Convert string representation of dictionaries to actual dictionaries
chennai_cars_df['new_car_specs']=chennai_cars_df['new_car_specs'].apply(ast.literal_eval)

# Normalize the 'new_car_detail' column into separate columns
chennai_cars_normalized = pd.json_normalize(chennai_cars_df['new_car_specs'])

# Concatenate the normalized columns with the original DataFrame
chennai_cars_df = pd.concat([chennai_cars_df, chennai_cars_normalized], axis=1)

# Drop the original 'new_car_detail' column
chennai_cars_df = chennai_cars_df.drop('new_car_specs', axis=1)


# In[23]:


chennai_cars_df.head(2)


# In[24]:


# For checking purpose
chennai_cars_df['top'][0]#[0]['value']#[0]['list'][0]['value']


# In[25]:


# Creating a new Dataframe to store required values
chennai_df_4 = {"Car_Specifications": [], "Values_for_Car_Specifications": []}

for index, row in chennai_cars_df.iterrows():
    for item in row['top']:
        key=item['key']
        values = item['value']
        chennai_df_4["Car_Specifications"].append(key)
        chennai_df_4["Values_for_Car_Specifications"].append(values)

# Convert the dictionary to a DataFrame
chennai_df_4 = pd.DataFrame(chennai_df_4)


# In[26]:


chennai_df_4.head(3)


# In[27]:


# For checking purpose
chennai_cars_df['data'][0]#[0]['list'][0]['value']


# In[28]:


# Creating a new Dataframe to store required values
chennai_df_5 = {"Feature_heading": [], "Feature_key_heading": [],"Feature_value":[]}

for index, row in chennai_cars_df.iterrows():
    for item in row['data']:
        heading = item['subHeading']
        key=item['list'][0]['key']
        values = item['list'][0]['value']
        chennai_df_5["Feature_heading"].append(heading)
        chennai_df_5["Feature_key_heading"].append(key)
        chennai_df_5["Feature_value"].append(values)

# Convert the dictionary to a DataFrame
chennai_df_5 = pd.DataFrame(chennai_df_5)


# In[29]:


chennai_df_5.head(2)


# In[30]:


# Drop the original 'new_car_detail' column
chennai_cars_df = chennai_cars_df.drop(['top','data'], axis=1)


# In[31]:


chennai_cars_df.head(2)


# In[32]:


print("chennai_cars_df columns:", chennai_cars_df.columns)
print("chennai_df_1 columns:", chennai_df_1.columns)
print("chennai_df_2 columns:", chennai_df_2.columns)
print("chennai_df_3 columns:", chennai_df_3.columns)
print("chennai_df_4 columns:", chennai_df_4.columns)
print("chennai_df_5 columns:", chennai_df_5.columns)


# In[33]:


print("chennai_cars_df shape:", chennai_cars_df.shape)
print("chennai_df_1 shape:", chennai_df_1.shape)
print("chennai_df_2 shape:", chennai_df_2.shape)
print("chennai_df_3 shape:", chennai_df_3.shape)
print("chennai_df_4 shape:", chennai_df_4.shape)
print("chennai_df_5 shape:", chennai_df_5.shape)


# In[34]:


# Merging all DataFrames

# Merge chennai_df_1 with chennai_cars_df
chennai_cars_df = pd.merge(chennai_cars_df, chennai_df_1, how='left', left_index=True, right_index=True)

# Merge chennai_df_2 with chennai_cars_df
chennai_cars_df = pd.merge(chennai_cars_df, chennai_df_2, how='left', left_index=True, right_index=True)

# Merge chennai_df_3 with chennai_cars_df
chennai_cars_df = pd.merge(chennai_cars_df, chennai_df_3, how='left', left_index=True, right_index=True)

# Merge chennai_df_4 with chennai_cars_df
chennai_cars_df = pd.merge(chennai_cars_df, chennai_df_4, how='left', left_index=True, right_index=True)

# Merge chennai_df_5 with chennai_cars_df
chennai_cars_df = pd.merge(chennai_cars_df, chennai_df_5, how='left', left_index=True, right_index=True)


# In[35]:


chennai_cars_df.shape


# In[36]:


chennai_cars_df.columns


# In[37]:


import pandas as pd

# Assuming you have a DataFrame named chennai_cars_df

# Define a dictionary mapping current column names to new names
column_rename_mapping = {
    'it': 'Ignition_type',
    'ft': 'Fuel_type',
    'bt': 'Body_type',
    'km': 'Kilometers_driven',
    'transmission': 'Transmission_type',
    'ownerNo': 'Number_of_previous_owners',
    'owner': 'Ownership_details',
    'oem': 'Original_Equipment_Manufacturer',
    'model': 'Car_model',
    'modelYear': 'Year_of_car_manufacture',
    'centralVariantId': 'Central_variant_ID',
    'variantName': 'Variant_name',
    'price': 'Price_of_the_used_car',
    'priceActual': 'Actual_price',
    'priceSaving': 'Price_saving_information',
    'priceFixedText': 'Fixed price details',
    'trendingText.imgUrl': 'Image_URL',
    'trendingText.heading': 'Trending_heading',
    'trendingText.desc': 'Trending_description'
}

# Rename columns in chennai_cars_df
chennai_cars_df.rename(columns=column_rename_mapping, inplace=True)


# In[38]:


chennai_cars_df.isnull().sum()/len(chennai_cars_df)*100


# In[39]:


chennai_cars_df.info()


# In[40]:


chennai_cars_df.head(10)


# In[41]:


chennai_cars_df.columns


# In[42]:


url_links_chennai=chennai_cars_df[['car_links','Image_link_of_car']]


# In[43]:


chennai_cars_df.drop(['car_links','Image_link_of_car'],axis=1,inplace=True)


# In[44]:


chennai_cars_df.head(2)


# In[45]:


chennai_cars_df['city'] = 'Chennai'


# In[93]:


chennai_cars_df.head(2)


# In[94]:


chennai_cars_df.to_excel('Final_chennai_cars_list.xlsx')


# In[95]:


chennai_cars_df.shape


# In[48]:


# Importing Banglore Cars
blr_cars=pd.read_excel("C:\\Users\\DELL\\Downloads\\bangalore_cars.xlsx")


# In[49]:


blr_cars.head(2)


# In[50]:


def json_normalize(df, column_name):
    import ast
    df[column_name] = df[column_name].apply(ast.literal_eval)
    df_normalized = pd.json_normalize(df[column_name])
    df_1 = pd.concat([df, df_normalized], axis=1)
    df_1 = df_1.drop(column_name, axis=1)
    return df_1

blr_cars_df = json_normalize(blr_cars, 'new_car_detail')


# In[51]:


blr_cars_df.head(2)


# In[52]:


def rename_columns(df, column_rename_mapping):
    column_rename_mapping = {
    'it': 'Ignition_type',
    'ft': 'Fuel_type',
    'bt': 'Body_type',
    'km': 'Kilometers_driven',
    'transmission': 'Transmission_type',
    'ownerNo': 'Number_of_previous_owners',
    'owner': 'Ownership_details',
    'oem': 'Original_Equipment_Manufacturer',
    'model': 'Car_model',
    'modelYear': 'Year_of_car_manufacture',
    'centralVariantId': 'Central_variant_ID',
    'variantName': 'Variant_name',
    'price': 'Price_of_the_used_car',
    'priceActual': 'Actual_price',
    'priceSaving': 'Price_saving_information',
    'priceFixedText': 'Fixed price details',
    'trendingText.imgUrl': 'Image_URL',
    'trendingText.heading': 'Trending_heading',
    'trendingText.desc': 'Trending_description'}
    df.rename(columns=column_rename_mapping, inplace=True)
    return df

# Rename columns in banglore_cars_df
blr_cars_df.rename(columns=column_rename_mapping, inplace=True)
    


# In[53]:


blr_cars_df.head(2)


# In[54]:


blr_cars_df = json_normalize(blr_cars_df, 'new_car_overview')


# In[55]:


blr_cars_df.head(2)


# In[56]:


# For checking purpose
blr_cars_df['top'][0][0]#['icon']#['key']#['value']['icon']


# In[57]:


# Creating a new Dataframe to store required values
blr_df_1 = {"Car_Info": [], "Values_for_Car_Info": [], "Image_link_of_car": []}

for index, row in blr_cars_df.iterrows():
    for item in row['top']:
        key = item['key']
        values = item['value']
        icon = item['icon']
        blr_df_1["Car_Info"].append(key)
        blr_df_1["Values_for_Car_Info"].append(values)
        blr_df_1["Image_link_of_car"].append(icon)

# Convert the dictionary to a DataFrame
blr_df_1 = pd.DataFrame(blr_df_1)


# In[58]:


blr_df_1.head(2)


# In[59]:


blr_cars_df.head(2)


# In[60]:


blr_cars_df = blr_cars_df.drop(['top'], axis=1)


# In[61]:


blr_cars_df = blr_cars_df.drop(['bottomData','heading'], axis=1)


# In[62]:


blr_cars_df.head(2)


# In[63]:


blr_cars_df = json_normalize(blr_cars_df, 'new_car_feature')


# In[64]:


blr_cars_df.head(2)


# In[65]:


# Creating a new Dataframe to store required values
blr_df_2 = { "Features": []}

for index, row in blr_cars_df.iterrows():
    for item in row['top']:
        values = item['value']
        blr_df_2["Features"].append(values)

# Convert the dictionary to a DataFrame
blr_df_2 = pd.DataFrame(blr_df_2)


# In[66]:


blr_df_2.head(2)


# In[67]:


# Creating a new Dataframe to store required values
blr_df_3 = {"Comfort_and_Convenience": [], "Comfort_and_Convenience_type": []}

for index, row in blr_cars_df.iterrows():
    for item in row['data']:
        heading = item['heading']
        values = item['list'][0]['value']
        blr_df_3["Comfort_and_Convenience"].append(heading)
        blr_df_3["Comfort_and_Convenience_type"].append(values)

# Convert the dictionary to a DataFrame
blr_df_3 = pd.DataFrame(blr_df_3)


# In[68]:


blr_df_3.head(2)


# In[69]:


blr_cars_df.head(2)


# In[70]:


blr_cars_df=blr_cars_df.drop(['top','data'],axis=1)


# In[71]:


blr_cars_df.head(2)


# In[72]:


blr_cars_df = json_normalize(blr_cars_df, 'new_car_specs')


# In[73]:


blr_cars_df.head(2)


# In[74]:


# For checking purpose
blr_cars_df['top'][0]#[0]['list'][0]['value']


# In[75]:


# Creating a new Dataframe to store required values
blr_df_4 = {"Car_Specifications": [], "Values_for_Car_Specifications": []}

for index, row in blr_cars_df.iterrows():
    for item in row['top']:
        key=item['key']
        values = item['value']
        blr_df_4["Car_Specifications"].append(key)
        blr_df_4["Values_for_Car_Specifications"].append(values)

# Convert the dictionary to a DataFrame
blr_df_4 = pd.DataFrame(blr_df_4)


# In[76]:


blr_df_4.head(2)


# In[77]:


# For checking purpose
blr_cars_df['data'][0]#[0]['list'][0]['value']


# In[78]:


# Creating a new Dataframe to store required values
blr_df_5 = {"Feature_heading": [], "Feature_key_heading": [],"Feature_value":[]}

for index, row in blr_cars_df.iterrows():
    for item in row['data']:
        heading = item['subHeading']
        key=item['list'][0]['key']
        values = item['list'][0]['value']
        blr_df_5["Feature_heading"].append(heading)
        blr_df_5["Feature_key_heading"].append(key)
        blr_df_5["Feature_value"].append(values)

# Convert the dictionary to a DataFrame
blr_df_5 = pd.DataFrame(blr_df_5)


# In[79]:


blr_df_5.head(2)


# In[80]:


blr_cars_df.head(2)


# In[81]:


# Drop the top and data column
blr_cars_df = blr_cars_df.drop(['top','data'], axis=1)


# In[82]:


blr_cars_df.head(2)


# In[83]:


# Merging all DataFrames

# Merge blr_df_1 with blr_cars_df
blr_cars_df = pd.merge(blr_cars_df, blr_df_1, how='left', left_index=True, right_index=True)

# Merge blr_df_2 with blr_cars_df
blr_cars_df = pd.merge(blr_cars_df, blr_df_2, how='left', left_index=True, right_index=True)

# Merge blr_df_3 with blr_cars_df
blr_cars_df = pd.merge(blr_cars_df, blr_df_3, how='left', left_index=True, right_index=True)

# Merge blr_df_4 with blr_cars_df
blr_cars_df = pd.merge(blr_cars_df, blr_df_4, how='left', left_index=True, right_index=True)

# Merge blr_df_5 with blr_cars_df
blr_cars_df = pd.merge(blr_cars_df, blr_df_5, how='left', left_index=True, right_index=True)


# In[84]:


blr_cars_df.head(2)


# In[85]:


blr_cars_df.shape


# In[86]:


blr_cars_df.columns


# In[87]:


url_links_blr=blr_cars_df[['car_links','Image_link_of_car']]


# In[88]:


blr_cars_df=blr_cars_df.drop(['car_links','Image_link_of_car'],axis=1)


# In[89]:


blr_cars_df['city']='Banglore'


# In[90]:


blr_cars_df.head(2)


# In[91]:


blr_cars_df.to_excel('Final_banglore_cars_list.xlsx')


# In[92]:


blr_cars_df.shape


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





# In[ ]:





# In[ ]:





# In[ ]:




