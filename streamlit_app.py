# streamlit_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="🌍 Global Food Wastage Dashboard", layout="wide")

st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .sidebar .sidebar-content {
        background: #2c5364;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 Global Food Wastage Analysis Dashboard")

# File Upload
uploaded_file = st.file_uploader("Upload the Dataset CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    # Show basic data
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Label Encoding
    df = df.dropna()
    le_country = LabelEncoder()
    le_category = LabelEncoder()
    df['Country'] = le_country.fit_transform(df['Country'])
    df['Food Category'] = le_category.fit_transform(df['Food Category'])

    # Correlation Heatmap
    st.subheader("🔍 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Economic Loss over Years (Top 5 Countries)
    st.subheader("📈 Economic Loss Over Time (Top 5 Countries)")
    df['Country Name'] = le_country.inverse_transform(df['Country'])
    top_countries = df['Country Name'].value_counts().head(5).index.tolist()
    filtered_df = df[df['Country Name'].isin(top_countries)]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=filtered_df, x='Year', y='Economic Loss (Million $)', hue='Country Name', marker='o', ax=ax2)
    st.pyplot(fig2)

    # Feature Input & Prediction
    st.subheader("📊 Predict Total Food Waste (Tons)")

    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("Country", le_country.classes_)
        food_cat = st.selectbox("Food Category", le_category.classes_)
    with col2:
        year = st.slider("Year", int(df['Year'].min()), int(df['Year'].max()))
        population = st.number_input("Population (Million)", min_value=0.0)
    with col3:
        economic_loss = st.number_input("Economic Loss (Million $)", min_value=0.0)
        per_capita = st.number_input("Avg Waste per Capita (Kg)", min_value=0.0)
        hh_waste = st.slider("Household Waste (%)", 0.0, 100.0)

    if st.button("Predict Waste"):
        # Prepare model input
        model_df = df.copy()
        X = model_df.drop(columns=['Total Waste (Tons)', 'Country Name'])
        y = model_df['Total Waste (Tons)']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_scaled, y)

        input_data = pd.DataFrame([[
            le_country.transform([country])[0],
            year,
            le_category.transform([food_cat])[0],
            economic_loss,
            per_capita,
            population,
            hh_waste
        ]], columns=X.columns)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"🌾 Predicted Total Waste (Tons): {prediction:.2f}")

else:
    st.warning("📁 Please upload a CSV file to begin.")
