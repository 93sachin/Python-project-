import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#loading dataset
data=pd.read_csv("C:\\Users\\dell\\Downloads\\Electric_Vehicle_Population_Data_30000.csv")
print(data)
#Exploring dataset
print("Information: \n",data.info())
print("Description: \n",data.describe())
#Handling Missing Values
print("missing values ",data.isnull().sum())
data = data.dropna(subset=['Model Year', 'Make', 'Model', 'County', 'City', 'State', 'Electric Vehicle Type', 'Base MSRP', 'Legislative District', 'DOL Vehicle ID', 'Vehicle Location', 'Electric Utility', '2020 Census Tract'])
data['Electric Range'] = data['Electric Range'].fillna(0)
data['Postal Code'] = data['Postal Code'].fillna(0)
data['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] = data['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].fillna("Unknown")
print("missing values ",data.isnull().sum())
#remove duplicate rows
data=data.drop_duplicates()
print(data)
#Basic operation Performed
print("1st 10 rows of Dataset: \n",data.head(10))
print("1st 10 rows of Dataset: \n",data.tail(10))
print("Shape of Dataset: \n",data.shape)
print("Column of Dataset: \n",data.columns)
print("Datatype of Dataset: \n",data.dtypes)
data.to_csv("cleaned_dataset.csv", index=False)
print("New Dataset Succesfully")
#find top EV locations
data = data.dropna(subset=['City', 'County', 'Postal Code']) #any missing values
#Top 10 cities with most evs
top_cities = data['City'].value_counts().head(10)
print("Top 10 Cities:\n", top_cities)
#Top 10 countries with most evs
top_counties = data['County'].value_counts().head(10)
print("\nTop 10 Counties:\n", top_counties)
#Top 10 ZIP Codes with the Most EVs
top_zipcodes = data['Postal Code'].value_counts().head(10)
#plot barplot
top_cities_data = top_cities.reset_index()
top_cities_data.columns = ['City', 'EV_Count']
plt.figure(figsize=(8,5))
sns.barplot(data=top_cities_data, x='EV_Count', y='City', hue='City', palette='viridis', legend=False)
plt.title('Top 10 Cities by EV Registrations')
plt.xlabel('Number of EVs')
plt.ylabel('City')
plt.show()
print("\nTop 10 ZIP Codes:\n", top_zipcodes)
#proportions of different EV types
data=data.dropna(subset=['Electric Vehicle Type'])
ev_type_counts =data ['Electric Vehicle Type'].value_counts()
print(ev_type_counts)
#plot countplot
plt.figure(figsize=(8,5))
sns.countplot(data=data, x='Electric Vehicle Type')
plt.title('Distribution of Electric Vehicle Types')
plt.xlabel('EV Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
#Top EV brands
top_brands = data['Make'].value_counts().head(10)
print("Top 10 EV Brands:\n", top_brands)
#plot barplot
brand_data = top_brands.reset_index()
brand_data.columns = ['Make', 'Count']
plt.figure(figsize=(8,5))
sns.barplot(data=brand_data, x='Count', y='Make', hue='Make', palette='magma', legend=False)
plt.title('Top 10 EV Brands by Registration')
plt.xlabel('Number of Vehicles')
plt.ylabel('Brand')
plt.show()
#Top EV Models
top_models = data['Model'].value_counts().head(10)
print("Top 10 EV Models:\n", top_models)
#plot barplot
model_data = top_models.reset_index()
model_data.columns = ['Model', 'Count']
plt.figure(figsize=(8,5))
sns.barplot(data=model_data, x='Count', y='Model', hue='Model', palette='bright', legend=False)
plt.title('Top 10 EV Models by Registration')
plt.xlabel('Number of Vehicles')
plt.ylabel('Model')
plt.show()
#EV Adoption Trends by Model Year
data = data.dropna(subset=['Model Year'])
ev_trends_by_year = data['Model Year'].value_counts().sort_index()
print("EV Registrations by Model Year:\n", ev_trends_by_year)
#plot lineplot
plt.figure(figsize=(10,6))
sns.lineplot(x=ev_trends_by_year.index, y=ev_trends_by_year.values, marker='o', color='green')
plt.title('EV Registrations by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Number of Registrations')
plt.grid(True)
plt.show()
#Kde plot 
plt.figure(figsize=(10, 6))
sns.kdeplot(data=data, x='Model Year', fill=True, color='green', linewidth=2)
plt.title("Distribution of EV Registrations Across Model Years", fontsize=14)
plt.xlabel("Model Year")
plt.ylabel("Density")
plt.grid(True)
plt.show()
# correlation matrix heatmap
numeric_data = data[['Model Year', 'Electric Range', 'Legislative District', 'Postal Code']]
# Compute correlation matrix
corr = numeric_data.corr()
# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()
# combinations of numeric features scatterplot & pair plot
sample_data = numeric_data.sample(n=500, random_state=42)  # sample 500 rows
# pair plot
sns.pairplot(sample_data, diag_kind='kde', corner=True)
plt.suptitle("Pair Plot of Numeric Features", y=1.02)
plt.show()
#boxplot by electric range distribution
plt.figure(figsize=(12,6))
sns.boxplot(data=data[data['Electric Range'] > 0], x='Make', y='Electric Range')
plt.xticks(rotation=45)
plt.title('Electric Range Distribution by make')
plt.show()
#histogram by electric range distribution
plt.figure(figsize=(8,5))
sns.histplot(data['Electric Range'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Electric Range ')
plt.xlabel('Electric Range (miles)')
plt.ylabel('Frequency')
plt.show()






                      
                 





      
