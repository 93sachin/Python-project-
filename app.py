import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🚗 Electric Vehicle Data Analysis Dashboard")

# Load default dataset
data = pd.read_csv("Electric_Vehicle_Population_Data.csv")

st.subheader("📊 Raw Data")
st.write(data.head())

st.subheader("📌 Dataset Info")
st.write(data.describe())

st.subheader("❌ Missing Values")
st.write(data.isnull().sum())

# Cleaning
data = data.dropna(subset=['Model Year','Make','Model','County','City','State'])
data['Electric Range'] = data['Electric Range'].fillna(0)

# 🔥 Top Cities
st.subheader("🏙 Top 10 Cities")
top_cities = data['City'].value_counts().head(10)
st.bar_chart(top_cities)

# 🔥 EV Type
st.subheader("🚘 EV Type Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=data, x='Electric Vehicle Type', ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# 🔥 Top Brands
st.subheader("🏆 Top Brands")
top_brands = data['Make'].value_counts().head(10)
st.bar_chart(top_brands)

# 🔥 Top Models
st.subheader("🚗 Top Models")
top_models = data['Model'].value_counts().head(10)
st.bar_chart(top_models)

# 🔥 Trend
st.subheader("📈 EV Trend by Year")
trend = data['Model Year'].value_counts().sort_index()
st.line_chart(trend)

# 🔥 Histogram
st.subheader("📊 Electric Range Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(data['Electric Range'], bins=30, kde=True, ax=ax2)
st.pyplot(fig2)

# 🔥 KDE
st.subheader("📉 Density Plot")
fig3, ax3 = plt.subplots()
sns.kdeplot(data=data, x='Model Year', fill=True, ax=ax3)
st.pyplot(fig3)

# 🔥 Heatmap
st.subheader("🔥 Correlation Heatmap")
numeric_data = data[['Model Year','Electric Range']]
corr = numeric_data.corr()

fig4, ax4 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)

# 🔥 Boxplot
st.subheader("📦 Electric Range by Brand")
fig5, ax5 = plt.subplots()
sns.boxplot(data=data[data['Electric Range'] > 0], x='Make', y='Electric Range', ax=ax5)
plt.xticks(rotation=45)
st.pyplot(fig5)