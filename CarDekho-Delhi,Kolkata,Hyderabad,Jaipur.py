#!/usr/bin/env python
# coding: utf-8

# In[95]:


# Importing delhi Cars
import pandas as pd
delhi_cars=pd.read_excel("C:\\Users\\DELL\\Downloads\\delhi_cars.xlsx")


# In[71]:


delhi_cars.head(2)


# In[96]:


def json_normalize(df, column_name):
    import ast
    df[column_name] = df[column_name].apply(ast.literal_eval)
    df_normalized = pd.json_normalize(df[column_name])
    df_1 = pd.concat([df, df_normalized], axis=1)
    df_1 = df_1.drop(column_name, axis=1)
    return df_1

delhi_cars_df = json_normalize(delhi_cars, 'new_car_detail')


# In[73]:


delhi_cars_df.head(2)


# In[97]:


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

# Rename columns in delhi_cars_df
delhi_cars_df.rename(columns=column_rename_mapping, inplace=True)


# In[75]:


delhi_cars_df.head(2)


# In[98]:


delhi_cars_df = json_normalize(delhi_cars_df, 'new_car_overview')


# In[99]:


# Creating a new Dataframe to store required values
delhi_df_1 = {"Car_Info": [], "Values_for_Car_Info": [], "Image_link_of_car": []}

for index, row in delhi_cars_df.iterrows():
    for item in row['top']:
        key = item['key']
        values = item['value']
        icon = item['icon']
        delhi_df_1["Car_Info"].append(key)
        delhi_df_1["Values_for_Car_Info"].append(values)
        delhi_df_1["Image_link_of_car"].append(icon)

# Convert the dictionary to a DataFrame
delhi_df_1 = pd.DataFrame(delhi_df_1)


# In[79]:


delhi_df_1.head(2)


# In[100]:


delhi_cars_df = delhi_cars_df.drop(['top'], axis=1)


# In[101]:


delhi_cars_df = delhi_cars_df.drop(['bottomData','heading'], axis=1)


# In[82]:


delhi_cars_df.head(2)


# In[102]:


delhi_cars_df = json_normalize(delhi_cars_df, 'new_car_feature')


# In[103]:


# Creating a new Dataframe to store required values
delhi_df_2 = { "Features": []}

for index, row in delhi_cars_df.iterrows():
    for item in row['top']:
        values = item['value']
        delhi_df_2["Features"].append(values)

# Convert the dictionary to a DataFrame
delhi_df_2 = pd.DataFrame(delhi_df_2)


# In[85]:


delhi_df_2.head(2)


# In[104]:


# Creating a new Dataframe to store required values
delhi_df_3 = {"Comfort_and_Convenience": [], "Comfort_and_Convenience_type": []}

for index, row in delhi_cars_df.iterrows():
    for item in row['data']:
        heading = item['heading']
        values = item['list'][0]['value']
        delhi_df_3["Comfort_and_Convenience"].append(heading)
        delhi_df_3["Comfort_and_Convenience_type"].append(values)

# Convert the dictionary to a DataFrame
delhi_df_3 = pd.DataFrame(delhi_df_3)


# In[87]:


delhi_df_3.head(2)


# In[105]:


delhi_cars_df=delhi_cars_df.drop(['top','data'],axis=1)


# In[89]:


delhi_cars_df.head(2)


# In[106]:


delhi_cars_df = json_normalize(delhi_cars_df, 'new_car_specs')


# In[107]:


# Creating a new Dataframe to store required values
delhi_df_4 = {"Car_Specifications": [], "Values_for_Car_Specifications": []}

for index, row in delhi_cars_df.iterrows():
    for item in row['top']:
        key=item['key']
        values = item['value']
        delhi_df_4["Car_Specifications"].append(key)
        delhi_df_4["Values_for_Car_Specifications"].append(values)

# Convert the dictionary to a DataFrame
delhi_df_4 = pd.DataFrame(delhi_df_4)


# In[92]:


delhi_df_4.head(2)


# In[93]:


delhi_cars_df.head(2)


# In[108]:


# Creating a new Dataframe to store required values
delhi_df_5 = {"Feature_heading": [], "Feature_key_heading": [],"Feature_value":[]}

for index, row in delhi_cars_df.iterrows():
    for item in row['data']:
        heading = item['subHeading']
        key=item['list'][0]['key']
        values = item['list'][0]['value']
        delhi_df_5["Feature_heading"].append(heading)
        delhi_df_5["Feature_key_heading"].append(key)
        delhi_df_5["Feature_value"].append(values)

# Convert the dictionary to a DataFrame
delhi_df_5 = pd.DataFrame(delhi_df_5)


# In[109]:


delhi_df_5.head(2)


# In[110]:


delhi_cars_df.head(2)


# In[111]:


delhi_cars_df=delhi_cars_df.drop(['top','data'],axis=1)


# In[112]:


# Merging all DataFrames

# Merge delhi_df_1 with delhi_cars_df
delhi_cars_df = pd.merge(delhi_cars_df, delhi_df_1, how='left', left_index=True, right_index=True)

# Merge delhi_df_2 with delhi_cars_df
delhi_cars_df = pd.merge(delhi_cars_df, delhi_df_2, how='left', left_index=True, right_index=True)

# Merge delhi_df_3 with delhi_cars_df
delhi_cars_df = pd.merge(delhi_cars_df, delhi_df_3, how='left', left_index=True, right_index=True)

# Merge delhi_df_4 with delhi_cars_df
delhi_cars_df = pd.merge(delhi_cars_df, delhi_df_4, how='left', left_index=True, right_index=True)

# Merge delhi_df_5 with delhi_cars_df
delhi_cars_df = pd.merge(delhi_cars_df, delhi_df_5, how='left', left_index=True, right_index=True)


# In[113]:


delhi_cars_df.columns


# In[114]:


url_links_delhi=delhi_cars_df[['car_links','Image_link_of_car']]


# In[115]:


delhi_cars_df=delhi_cars_df.drop(['car_links','Image_link_of_car'],axis=1)


# In[116]:


delhi_cars_df['city']='Delhi'


# In[117]:


delhi_cars_df.shape


# In[118]:


delhi_cars_df.to_excel('Final_Delhi_cars_list.xlsx')


# In[119]:


# Importing delhi Cars
import pandas as pd
hyd_cars=pd.read_excel("C:\\Users\\DELL\\Downloads\\hyderabad_cars.xlsx")


# In[121]:


def json_normalize(df, column_name):
    import ast
    df[column_name] = df[column_name].apply(ast.literal_eval)
    df_normalized = pd.json_normalize(df[column_name])
    df_1 = pd.concat([df, df_normalized], axis=1)
    df_1 = df_1.drop(column_name, axis=1)
    return df_1

hyd_cars_df = json_normalize(hyd_cars, 'new_car_detail')


# In[123]:


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
hyd_cars_df.rename(columns=column_rename_mapping, inplace=True)


# In[124]:


hyd_cars_df.head(2)


# In[126]:


hyd_cars_df = json_normalize(hyd_cars_df, 'new_car_overview')


# In[128]:


# Creating a new Dataframe to store required values
hyd_df_1 = {"Car_Info": [], "Values_for_Car_Info": [], "Image_link_of_car": []}

for index, row in hyd_cars_df.iterrows():
    for item in row['top']:
        key = item['key']
        values = item['value']
        icon = item['icon']
        hyd_df_1["Car_Info"].append(key)
        hyd_df_1["Values_for_Car_Info"].append(values)
        hyd_df_1["Image_link_of_car"].append(icon)

# Convert the dictionary to a DataFrame
hyd_df_1 = pd.DataFrame(hyd_df_1)


# In[130]:


hyd_cars_df = hyd_cars_df.drop(['top'], axis=1)


# In[131]:


hyd_cars_df = hyd_cars_df.drop(['bottomData','heading'], axis=1)


# In[132]:


hyd_cars_df = json_normalize(hyd_cars_df, 'new_car_feature')


# In[134]:


# Creating a new Dataframe to store required values
hyd_df_2 = { "Features": []}

for index, row in hyd_cars_df.iterrows():
    for item in row['top']:
        values = item['value']
        hyd_df_2["Features"].append(values)

# Convert the dictionary to a DataFrame
hyd_df_2 = pd.DataFrame(hyd_df_2)


# In[135]:


# Creating a new Dataframe to store required values
hyd_df_3 = {"Comfort_and_Convenience": [], "Comfort_and_Convenience_type": []}

for index, row in hyd_cars_df.iterrows():
    for item in row['data']:
        heading = item['heading']
        values = item['list'][0]['value']
        hyd_df_3["Comfort_and_Convenience"].append(heading)
        hyd_df_3["Comfort_and_Convenience_type"].append(values)

# Convert the dictionary to a DataFrame
hyd_df_3 = pd.DataFrame(hyd_df_3)


# In[136]:


hyd_cars_df=hyd_cars_df.drop(['top','data'],axis=1)


# In[137]:


hyd_cars_df = json_normalize(hyd_cars_df, 'new_car_specs')


# In[138]:


# Creating a new Dataframe to store required values
hyd_df_4 = {"Car_Specifications": [], "Values_for_Car_Specifications": []}

for index, row in hyd_cars_df.iterrows():
    for item in row['top']:
        key=item['key']
        values = item['value']
        hyd_df_4["Car_Specifications"].append(key)
        hyd_df_4["Values_for_Car_Specifications"].append(values)

# Convert the dictionary to a DataFrame
hyd_df_4 = pd.DataFrame(hyd_df_4)


# In[139]:


# Creating a new Dataframe to store required values
hyd_df_5 = {"Feature_heading": [], "Feature_key_heading": [],"Feature_value":[]}

for index, row in hyd_cars_df.iterrows():
    for item in row['data']:
        heading = item['subHeading']
        key=item['list'][0]['key']
        values = item['list'][0]['value']
        hyd_df_5["Feature_heading"].append(heading)
        hyd_df_5["Feature_key_heading"].append(key)
        hyd_df_5["Feature_value"].append(values)

# Convert the dictionary to a DataFrame
hyd_df_5 = pd.DataFrame(hyd_df_5)


# In[140]:


# Drop the top and data column
hyd_cars_df = hyd_cars_df.drop(['top','data'], axis=1)


# In[141]:


# Merging all DataFrames

# Merge hyd_df_1 with hyd_cars_df
hyd_cars_df = pd.merge(hyd_cars_df, hyd_df_1, how='left', left_index=True, right_index=True)

# Merge hyd_df_2 with hyd_cars_df
hyd_cars_df = pd.merge(hyd_cars_df, hyd_df_2, how='left', left_index=True, right_index=True)

# Merge hyd_df_3 with hyd_cars_df
hyd_cars_df = pd.merge(hyd_cars_df, hyd_df_3, how='left', left_index=True, right_index=True)

# Merge hyd_df_4 with hyd_cars_df
hyd_cars_df = pd.merge(hyd_cars_df, hyd_df_4, how='left', left_index=True, right_index=True)

# Merge hyd_df_5 with hyd_cars_df
hyd_cars_df = pd.merge(hyd_cars_df, hyd_df_5, how='left', left_index=True, right_index=True)


# In[142]:


hyd_cars_df.head(2)


# In[143]:


url_links_hyd=hyd_cars_df[['car_links','Image_link_of_car']]


# In[144]:


hyd_cars_df=hyd_cars_df.drop(['car_links','Image_link_of_car'],axis=1)


# In[145]:


hyd_cars_df['city']='Hyderabad'


# In[146]:


hyd_cars_df.shape


# In[147]:


hyd_cars_df.to_excel('Final_Hyderabad_cars_list.xlsx')


# In[149]:


# Importing Jaipur Cars
jaipur_cars=pd.read_excel("C:\\Users\\DELL\\Downloads\\jaipur_cars.xlsx")


# In[150]:


def json_normalize(df, column_name):
    import ast
    df[column_name] = df[column_name].apply(ast.literal_eval)
    df_normalized = pd.json_normalize(df[column_name])
    df_1 = pd.concat([df, df_normalized], axis=1)
    df_1 = df_1.drop(column_name, axis=1)
    return df_1

jaipur_cars_df = json_normalize(jaipur_cars, 'new_car_detail')


# In[151]:


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
jaipur_cars_df.rename(columns=column_rename_mapping, inplace=True)


# In[152]:


jaipur_cars_df = json_normalize(jaipur_cars_df, 'new_car_overview')


# In[153]:


# Creating a new Dataframe to store required values
jaipur_df_1 = {"Car_Info": [], "Values_for_Car_Info": [], "Image_link_of_car": []}

for index, row in jaipur_cars_df.iterrows():
    for item in row['top']:
        key = item['key']
        values = item['value']
        icon = item['icon']
        jaipur_df_1["Car_Info"].append(key)
        jaipur_df_1["Values_for_Car_Info"].append(values)
        jaipur_df_1["Image_link_of_car"].append(icon)

# Convert the dictionary to a DataFrame
jaipur_df_1 = pd.DataFrame(jaipur_df_1)


# In[154]:


jaipur_cars_df = jaipur_cars_df.drop(['top'], axis=1)


# In[155]:


jaipur_cars_df = jaipur_cars_df.drop(['bottomData','heading'], axis=1)


# In[156]:


jaipur_cars_df = json_normalize(jaipur_cars_df, 'new_car_feature')


# In[157]:


# Creating a new Dataframe to store required values
jaipur_df_2 = { "Features": []}

for index, row in jaipur_cars_df.iterrows():
    for item in row['top']:
        values = item['value']
        jaipur_df_2["Features"].append(values)

# Convert the dictionary to a DataFrame
jaipur_df_2 = pd.DataFrame(jaipur_df_2)


# In[158]:


# Creating a new Dataframe to store required values
jaipur_df_3 = {"Comfort_and_Convenience": [], "Comfort_and_Convenience_type": []}

for index, row in jaipur_cars_df.iterrows():
    for item in row['data']:
        heading = item['heading']
        values = item['list'][0]['value']
        jaipur_df_3["Comfort_and_Convenience"].append(heading)
        jaipur_df_3["Comfort_and_Convenience_type"].append(values)

# Convert the dictionary to a DataFrame
jaipur_df_3 = pd.DataFrame(jaipur_df_3)


# In[159]:


jaipur_cars_df=jaipur_cars_df.drop(['top','data'],axis=1)


# In[160]:


jaipur_cars_df = json_normalize(jaipur_cars_df, 'new_car_specs')


# In[161]:


# Creating a new Dataframe to store required values
jaipur_df_4 = {"Car_Specifications": [], "Values_for_Car_Specifications": []}

for index, row in jaipur_cars_df.iterrows():
    for item in row['top']:
        key=item['key']
        values = item['value']
        jaipur_df_4["Car_Specifications"].append(key)
        jaipur_df_4["Values_for_Car_Specifications"].append(values)

# Convert the dictionary to a DataFrame
jaipur_df_4 = pd.DataFrame(jaipur_df_4)


# In[162]:


# Creating a new Dataframe to store required values
jaipur_df_5 = {"Feature_heading": [], "Feature_key_heading": [],"Feature_value":[]}

for index, row in jaipur_cars_df.iterrows():
    for item in row['data']:
        heading = item['subHeading']
        key=item['list'][0]['key']
        values = item['list'][0]['value']
        jaipur_df_5["Feature_heading"].append(heading)
        jaipur_df_5["Feature_key_heading"].append(key)
        jaipur_df_5["Feature_value"].append(values)

# Convert the dictionary to a DataFrame
jaipur_df_5 = pd.DataFrame(jaipur_df_5)


# In[163]:


# Drop the top and data column
jaipur_cars_df = jaipur_cars_df.drop(['top','data'], axis=1)


# In[164]:


# Merging all DataFrames

# Merge jaipur_df_1 with jaipur_cars_df
jaipur_cars_df = pd.merge(jaipur_cars_df, jaipur_df_1, how='left', left_index=True, right_index=True)

# Merge jaipur_df_2 with jaipur_cars_df
jaipur_cars_df = pd.merge(jaipur_cars_df, jaipur_df_2, how='left', left_index=True, right_index=True)

# Merge jaipur_df_3 with jaipur_cars_df
jaipur_cars_df = pd.merge(jaipur_cars_df, jaipur_df_3, how='left', left_index=True, right_index=True)

# Merge jaipur_df_4 with jaipur_cars_df
jaipur_cars_df = pd.merge(jaipur_cars_df, jaipur_df_4, how='left', left_index=True, right_index=True)

# Merge jaipur_df_5 with jaipur_cars_df
jaipur_cars_df = pd.merge(jaipur_cars_df, jaipur_df_5, how='left', left_index=True, right_index=True)


# In[165]:


url_links_jaipur=jaipur_cars_df[['car_links','Image_link_of_car']]


# In[166]:


jaipur_cars_df=jaipur_cars_df.drop(['car_links','Image_link_of_car'],axis=1)


# In[167]:


jaipur_cars_df['city']='Jaipur'


# In[168]:


jaipur_cars_df.shape


# In[169]:


jaipur_cars_df.to_excel('Final_Jaipur_cars_list.xlsx')


# In[186]:


# Importing Kolkata Cars
kol_cars=pd.read_excel("C:\\Users\\DELL\\Downloads\\kolkata_cars.xlsx")


# In[187]:


def json_normalize(df, column_name):
    import ast
    df[column_name] = df[column_name].apply(ast.literal_eval)
    df_normalized = pd.json_normalize(df[column_name])
    df_1 = pd.concat([df, df_normalized], axis=1)
    df_1 = df_1.drop(column_name, axis=1)
    return df_1

kol_cars_df = json_normalize(kol_cars, 'new_car_detail')


# In[188]:


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
kol_cars_df.rename(columns=column_rename_mapping, inplace=True)


# In[189]:


kol_cars_df = json_normalize(kol_cars_df, 'new_car_overview')


# In[190]:


# Creating a new Dataframe to store required values
kol_df_1 = {"Car_Info": [], "Values_for_Car_Info": [], "Image_link_of_car": []}

for index, row in kol_cars_df.iterrows():
    for item in row['top']:
        key = item['key']
        values = item['value']
        icon = item['icon']
        kol_df_1["Car_Info"].append(key)
        kol_df_1["Values_for_Car_Info"].append(values)
        kol_df_1["Image_link_of_car"].append(icon)

# Convert the dictionary to a DataFrame
kol_df_1 = pd.DataFrame(kol_df_1)


# In[191]:


kol_cars_df = kol_cars_df.drop(['top'], axis=1)


# In[192]:


kol_cars_df = kol_cars_df.drop(['bottomData','heading'], axis=1)


# In[193]:


kol_cars_df = json_normalize(kol_cars_df, 'new_car_feature')


# In[194]:


# Creating a new Dataframe to store required values
kol_df_2 = { "Features": []}

for index, row in kol_cars_df.iterrows():
    for item in row['top']:
        values = item['value']
        kol_df_2["Features"].append(values)

# Convert the dictionary to a DataFrame
kol_df_2 = pd.DataFrame(kol_df_2)


# In[195]:


# Creating a new Dataframe to store required values
kol_df_3 = {"Comfort_and_Convenience": [], "Comfort_and_Convenience_type": []}

for index, row in kol_cars_df.iterrows():
    for item in row['data']:
        heading = item['heading']
        values = item['list'][0]['value']
        kol_df_3["Comfort_and_Convenience"].append(heading)
        kol_df_3["Comfort_and_Convenience_type"].append(values)

# Convert the dictionary to a DataFrame
kol_df_3 = pd.DataFrame(kol_df_3)


# In[196]:


kol_cars_df=kol_cars_df.drop(['top','data'],axis=1)


# In[197]:


kol_cars_df = json_normalize(kol_cars_df, 'new_car_specs')


# In[198]:


# Creating a new Dataframe to store required values
kol_df_4 = {"Car_Specifications": [], "Values_for_Car_Specifications": []}

for index, row in kol_cars_df.iterrows():
    for item in row['top']:
        key=item['key']
        values = item['value']
        kol_df_4["Car_Specifications"].append(key)
        kol_df_4["Values_for_Car_Specifications"].append(values)

# Convert the dictionary to a DataFrame
kol_df_4 = pd.DataFrame(kol_df_4)


# In[199]:


# Creating a new Dataframe to store required values
kol_df_5 = {"Feature_heading": [], "Feature_key_heading": [],"Feature_value":[]}

for index, row in kol_cars_df.iterrows():
    for item in row['data']:
        heading = item['subHeading']
        key=item['list'][0]['key']
        values = item['list'][0]['value']
        kol_df_5["Feature_heading"].append(heading)
        kol_df_5["Feature_key_heading"].append(key)
        kol_df_5["Feature_value"].append(values)

# Convert the dictionary to a DataFrame
kol_df_5 = pd.DataFrame(kol_df_5)


# In[200]:


# Drop the top and data column
kol_cars_df = kol_cars_df.drop(['top','data'], axis=1)


# In[201]:


# Merging all DataFrames

# Merge kol_df_1 with kol_cars_df
kol_cars_df = pd.merge(kol_cars_df, kol_df_1, how='left', left_index=True, right_index=True)

# Merge kol_df_2 with kol_cars_df
kol_cars_df = pd.merge(kol_cars_df, kol_df_2, how='left', left_index=True, right_index=True)

# Merge kol_df_3 with kol_cars_df
kol_cars_df = pd.merge(kol_cars_df, kol_df_3, how='left', left_index=True, right_index=True)

# Merge kol_df_4 with kol_cars_df
kol_cars_df = pd.merge(kol_cars_df, kol_df_4, how='left', left_index=True, right_index=True)

# Merge kol_df_5 with kol_cars_df
kol_cars_df = pd.merge(kol_cars_df, kol_df_5, how='left', left_index=True, right_index=True)


# In[202]:


url_links_kol=kol_cars_df[['car_links','Image_link_of_car']]


# In[203]:


kol_cars_df=kol_cars_df.drop(['car_links','Image_link_of_car'],axis=1)


# In[205]:


kol_cars_df['city']='Kolkata'


# In[206]:


kol_cars_df.shape


# In[207]:


kol_cars_df.to_excel('Final_Kolkata_cars_list.xlsx')


# In[208]:


import pandas as pd
chennai=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_chennai_cars_list.xlsx")
banglore=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_banglore_cars_list.xlsx")
hyderabad=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_Hyderabad_cars_list.xlsx")
Jaipur=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_Jaipur_cars_list.xlsx")
Kolkata=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_Kolkata_cars_list.xlsx")
delhi=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_Delhi_cars_list.xlsx")

Final_cars_list=pd.concat([chennai,banglore,hyderabad,Jaipur,Kolkata,delhi],axis=0)
Final_cars_list.isnull().sum()/len(Final_cars_list)*100
Final_cars_list=Final_cars_list.drop(['Price_saving_information','Fixed price details','Actual_price'],axis=1)
Final_cars_list=Final_cars_list.drop(['commonIcon.1'],axis=1)

Final_cars_list.isnull().sum()/len(Final_cars_list)*100

Final_cars_list.to_excel('Final_CarDekho_list.xlsx')


# In[209]:


Final_cars_list.head(4)


# In[212]:


Final_cars_list.columns


# In[213]:


Final_cars_list=Final_cars_list.drop(['Unnamed: 0'],axis=1)


# In[214]:


Final_cars_list.head(4)


# In[ ]:


df=[['Ignition_type', 'Fuel_type', 'Body_type',
       'Kilometers_driven', 'Transmission_type', 'Number_of_previous_owners',
       'Ownership_details', 'Original_Equipment_Manufacturer', 'Car_model',
       'Year_of_car_manufacture', 'Central_variant_ID', 'Variant_name',
       'Price_of_the_used_car', 'Image_URL', 'Trending_heading',
       'Trending_description']]


# In[215]:


## CarDekho Project
## Importing all cities car list ## 
import pandas as pd
chennai=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_chennai_cars_list.xlsx")
banglore=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_banglore_cars_list.xlsx")
hyderabad=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_Hyderabad_cars_list.xlsx")
Jaipur=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_Jaipur_cars_list.xlsx")
Kolkata=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_Kolkata_cars_list.xlsx")
delhi=pd.read_excel("C:\\Users\\DELL\\Downloads\\CarDekho_project_excel_ouputs\\Final_Delhi_cars_list.xlsx")

Final_cars_list=pd.concat([chennai,banglore,hyderabad,Jaipur,Kolkata,delhi],axis=0)
Final_cars_list.isnull().sum()/len(Final_cars_list)*100
Final_cars_list=Final_cars_list.drop(['Price_saving_information','Fixed price details','Actual_price'],axis=1)
Final_cars_list=Final_cars_list.drop(['commonIcon.1'],axis=1)

Final_cars_list.isnull().sum()/len(Final_cars_list)*100

Final_cars_list.to_excel('Final_CarDekho_list.xlsx')

Final_cars_list['Trending_heading'].value_counts()
Final_cars_list['heading'].value_counts()
Final_cars_list['heading.1'].value_counts()
Final_cars_list['Trending_description'].value_counts()

Final_cars_list=Final_cars_list.drop(['Trending_heading','heading','heading.1'],axis=1)
Final_cars_list=Final_cars_list.drop(['Trending_description'],axis=1)

car_info=Final_cars_list[['Car_Info','Values_for_Car_Info']]
car_info_df=car_info.T

Final_cars_list['Car_Info'].value_counts()
Final_cars_list['Values_for_Car_Info'].value_counts()

# Create a pivot table
Car_Info = Final_cars_list.pivot_table(index=Final_cars_list.index, columns='Car_Info', values='Values_for_Car_Info', aggfunc='first')
Car_Info.isnull().sum()/len(Car_Info)*100

Final_cars_list=Final_cars_list.drop(['Car_Info','Values_for_Car_Info'],axis=1)

Final_cars_list = Final_cars_list.reset_index(drop=True)
Car_Info = Car_Info.reset_index(drop=True)

# Concatenate the DataFrames along axis=1
Final_cars_list = pd.concat([Final_cars_list, Car_Info], axis=1)

Final_cars_list=Final_cars_list.drop(['Features'],axis=1)

Final_cars_list.columns

Comfort_Convenience = Final_cars_list.pivot_table(index=Final_cars_list.index, columns='Comfort_and_Convenience', values='Comfort_and_Convenience_type', aggfunc='first')
Comfort_Convenience.isnull().sum()/len(Comfort_Convenience)*100

Final_cars_list=Final_cars_list.drop(['Comfort_and_Convenience','Comfort_and_Convenience_type'],axis=1)
Final_cars_list = pd.concat([Final_cars_list, Comfort_Convenience], axis=1)

Car_Specifications = Final_cars_list.pivot_table(index=Final_cars_list.index, columns='Car_Specifications', values='Values_for_Car_Specifications', aggfunc='first')
Car_Specifications.isnull().sum()/len(Car_Specifications)*100

Final_cars_list=Final_cars_list.drop(['Car_Specifications','Values_for_Car_Specifications'],axis=1)
Final_cars_list = pd.concat([Final_cars_list, Car_Specifications], axis=1)

Final_cars_list=Final_cars_list.drop(['Feature_heading'],axis=1)

Car_features = Final_cars_list.pivot_table(index=Final_cars_list.index, columns='Feature_key_heading', values='Feature_value', aggfunc='first')
Car_features.isnull().sum()/len(Car_features)*100

Final_cars_list=Final_cars_list.drop(['Feature_key_heading','Feature_value'],axis=1)
Final_cars_list = pd.concat([Final_cars_list, Car_features], axis=1)

img_url=Final_cars_list[['commonIcon','Image_URL']]
Final_cars_list=Final_cars_list.drop(['commonIcon','Image_URL'],axis=1)

Final_cars_list=Final_cars_list.drop(['Unnamed: 0'],axis=1)
Final_cars_list.columns

# getting null percentage for all the columns
null_percentage = (Final_cars_list.isnull().sum() / len(Final_cars_list)) * 100
print("Percentage of null values for each column:")
print(null_percentage)

## Setting an percentage so that to remove those columns
columns_to_remove = null_percentage[null_percentage >= 75].index

# Drop columns with null percentage >= 75%
Final_cars_list= Final_cars_list.drop(columns=columns_to_remove)
Final_cars_list.isnull().sum() / len(Final_cars_list) * 100

Final_df=Final_cars_list[['Car_model','Year_of_car_manufacture','Kilometers_driven','Number_of_previous_owners',
                          'Transmission_type','Fuel_type','Body_type','Color','Price_of_the_used_car','city']]
Final_df.to_excel('Final_data.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




