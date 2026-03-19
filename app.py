import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🚗 Electric Vehicle Data Analysis Dashboard")

# Upload CSV file
file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if file is not None:
    data = pd.read_csv(file)

    st.subheader("📊 Raw Data")
    st.write(data.head())

    st.subheader("📌 Dataset Info")
    st.write(data.describe())

    st.subheader("❌ Missing Values")
    st.write(data.isnull().sum())

    # Data cleaning
    data = data.dropna(subset=['Model Year','Make','Model','County','City','State'])
    data['Electric Range'] = data['Electric Range'].fillna(0)

    # Top Cities
    st.subheader("🏙 Top 10 Cities with EVs")
    top_cities = data['City'].value_counts().head(10)
    st.bar_chart(top_cities)

    # EV Types
    st.subheader("🚘 EV Type Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=data, x='Electric Vehicle Type', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Top Brands
    st.subheader("🏆 Top EV Brands")
    top_brands = data['Make'].value_counts().head(10)
    st.bar_chart(top_brands)

    # Model Year Trend
    st.subheader("📈 EV Trend by Year")
    trend = data['Model Year'].value_counts().sort_index()
    st.line_chart(trend)

