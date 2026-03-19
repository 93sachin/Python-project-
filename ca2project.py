import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset
dt=pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\toolbox\\Electric_Vehicle_Population_Data1.csv",encoding="unicode_escape")
print(dt)

#Exploring dataset
print("Information: \n",dt.info())
print("Description: \n",dt.describe())

#Handling Missing Values
print("Missing values before handling:\n", dt.isnull().sum())
dt = dt.dropna(subset=['Model', 'County', 'City', 'State', 'Postal Code', 'Electric Vehicle Type', 'Base MSRP', 'Legislative District', 'DOL Vehicle ID', 'Vehicle Location', 'Electric Utility', '2020 Census Tract'])
dt['Electric Range'] = dt['Electric Range'].fillna(0)
dt['Base MSRP'] = dt['Base MSRP'].fillna(0)
dt['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] = dt['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].fillna("Unknown")
print("missing values ",dt.isnull().sum())

#remove duplicate rows
dt=dt.drop_duplicates()
print(dt)

#Basic operation Performed
print("1st 12 rows of Dataset: \n",dt.head(12))
print("1st 12 rows of Dataset: \n",dt.tail(12))
print("Shape of Dataset: \n",dt.shape)
print("Column of Dataset: \n",dt.columns)
print("Datatype of Dataset: \n",dt.dtypes)
dt.to_csv("cleaned_dataset.csv", index=False)
print("New Dataset Succesfully")

# Remove the column from given dataset 
print(dt.drop(['Electric Utility','2020 Census Tract'],axis=1,inplace=True))
print("Information: \n",dt.info())

# change the column
print(dt.columns)
dt.columns = dt.columns.str.strip()
dt.rename(columns={'Electric Vehicle Type': 'EVT'},inplace=True)
print(dt.columns)
print(dt.columns)
dt.columns = dt.columns.str.strip()
dt.rename(columns={'Legislative District': 'LD'},inplace=True)
print(dt.columns)

# Clean column names
dt.columns = dt.columns.str.strip()

#find top EV locations
dt = dt.dropna(subset=['City', 'County', 'Postal Code']) #any missing values

#Top 10 cities with most evs
top_cities = dt['City'].value_counts().head(10)
print("Top 10 Cities:\n", top_cities)

# Countplot for LD
plt.figure(figsize=(10, 6))
sns.countplot(x='Model Year', hue='Model Year', data=dt, palette='coolwarm', legend=False)
plt.title("Count Plot")
plt.show()

# Scatterplot for Age vs Salary 
sns.scatterplot(x='Electric Range', y='LD', data=dt, hue='LD', palette='coolwarm')
plt.title("Scatter Plot")
plt.show()

# Select a subset of data with fewer unique cities and postal codes
top_cities = dt['City'].value_counts().nlargest(10).index  # Top 10 cities
filtered_data = dt[dt['City'].isin(top_cities)]

#select a numerical column
column='LD'
plt.figure(figsize=(8,5))
sns.boxplot(x=dt[column])
plt.title("Boxplot for outliers Detection")

#The distribution is slightly skewed left
Q1=dt[column].quantile(0.25)
print('Q1:',Q1)
Q3=dt[column].quantile(0.75)
print('Q3:',Q3)
IQR=Q3-Q1
print('IQR:',IQR)

lower_bound=Q1-1.5*IQR
print('lower bound:',lower_bound)
upper_bound=Q3+1.5*IQR
print('upper bound:',upper_bound)

# Identifying outliers
outliers = dt[(dt[column] < lower_bound) | (dt[column] > upper_bound)]
print("Outliers detected:\n",outliers)

# Boxplot of DOL Vehicle ID
plt.figure(figsize=(8, 5))
sns.boxplot(x=dt['DOL Vehicle ID'], color='teal')
plt.title("Boxplot of DOL Vehicle ID")
plt.xlabel("DOL Vehicle ID")
plt.grid(True)
plt.show()

# Distribution of Electric Range
plt.figure(figsize=(8, 5))
sns.histplot(dt['Electric Range'], bins=30, kde=True, color='purple')
plt.title("Distribution of Electric Range")
plt.xlabel("Electric Range")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# City-wise EV Count (Top 10)
top_cities = dt['City'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_cities.values, y=top_cities.index, palette='magma')
plt.title("Top 10 Cities with Most EVs")
plt.xlabel("Number of EVs")
plt.ylabel("City")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Fill missing numerical values with 0 and categorical with 'Unknown'
dt['Electric Range'] = dt['Electric Range'].fillna(0)
dt['Base MSRP'] = dt['Base MSRP'].fillna(0)
dt['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] = dt['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].fillna("Unknown")

# Drop rows with missing critical values
dt.dropna(subset=['Model', 'County', 'City', 'Postal Code', 'EVT', 'Base MSRP'], inplace=True)

# Drop duplicates
dt.drop_duplicates(inplace=True)

# Create correlation heatmap for numeric columns
plt.figure(figsize=(10, 6))
sns.heatmap(dt.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for numerical features
num_cols = dt.select_dtypes(include=['int64', 'float64']).columns
sns.pairplot(dt[num_cols])
plt.suptitle("Pairplot of Numerical Features", y=1.02)
plt.show()

# Barplot for top 10 Postal Codes
top_postal = dt['Postal Code'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_postal.index, y=top_postal.values, palette='magma')
plt.title("Top 10 Postal Codes")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Boxplot of Electric Range by City (top 10 cities)
top_cities = dt['City'].value_counts().head(10).index
filtered_dt = dt[dt['City'].isin(top_cities)]
plt.figure(figsize=(12, 6))
sns.boxplot(x='City', y='Electric Range', data=filtered_dt)
plt.title("Electric Range by City")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# EV Type Distribution in Top 5 Counties
top5_counties = dt['County'].value_counts().head(5).index
filtered = dt[dt['County'].isin(top5_counties)]
plt.figure(figsize=(13, 9))
sns.countplot(data=filtered, x='County', hue='EVT', palette='Set1')
plt.title("EV Type Distribution in Top 5 Counties")
plt.xlabel("County")
plt.ylabel("Count")
plt.legend(title='EV Type')
plt.show()

# Most Popular EV Models
top_models = dt['Model'].value_counts().head(10)
plt.figure(figsize=(15, 10))
sns.barplot(x=top_models.values, y=top_models.index, palette='Accent')
plt.title("Top 10 Most Popular EV Models")
plt.xlabel("Count")
plt.ylabel("Model")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Average Electric Range by Model (Top 10)
avg_range_by_model = dt.groupby('Model')['Electric Range'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_range_by_model.values, y=avg_range_by_model.index, palette='cool')
plt.title("Average Electric Range by Top EV Models")
plt.xlabel("Average Electric Range")
plt.ylabel("Model")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

