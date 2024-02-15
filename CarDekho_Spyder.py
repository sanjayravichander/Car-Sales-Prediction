# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:46:34 2024

@author: DELL
"""
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
