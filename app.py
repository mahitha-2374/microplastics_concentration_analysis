import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Microplastic Pollution Dashboard")

st.title("🌊 Microplastic Concentration Analysis & Prediction")

# Upload section
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Microplastics CSV file", type=["csv"])


# Function to extract concentration from text
def extract_concentration(val):
    try:
        val = str(val).replace(',', '')
        numbers = list(map(float, re.findall(r'\d+\.?\d*', val)))
        return np.mean(numbers) if numbers else np.nan
    except:
        return np.nan

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df.columns = ['Country', 'Location', 'Microplastic_Concentration', 'Reference']

    # Cleaning
    df['Country'] = df['Country'].astype(str).str.strip()
    df['Location'] = df['Location'].astype(str).str.strip()
    df['Concentration_Num'] = df['Microplastic_Concentration'].apply(extract_concentration)
    df = df.dropna(subset=['Concentration_Num'])

    st.success("✅ File uploaded and cleaned successfully!")

    # Sidebar filters
    country_list = sorted(df['Country'].unique())
    selected_country = st.sidebar.selectbox("🌍 Select Country", ['All'] + country_list)

    if selected_country != 'All':
        df = df[df['Country'] == selected_country]

    location_list = sorted(df['Location'].unique())
    selected_location = st.sidebar.selectbox("🏞️ Select Waterbody", ['All'] + location_list)

    if selected_location != 'All':
        df = df[df['Location'] == selected_location]

    st.subheader("📊 Data Preview")
    st.dataframe(df[['Country', 'Location', 'Microplastic_Concentration', 'Concentration_Num']])

    # Visualization
    st.subheader("📌 Distribution of Microplastic Concentration")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Concentration_Num'], bins=30, kde=True, ax=ax1, color="skyblue")
    ax1.set_xlabel("Concentration")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    st.subheader("📦 Top Waterbodies by Concentration")
    top_locs = df.groupby("Location")["Concentration_Num"].mean().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(x=top_locs.values, y=top_locs.index, palette="viridis", ax=ax2)
    ax2.set_xlabel("Avg Microplastic Concentration")
    st.pyplot(fig2)

    # Machine Learning
    st.subheader("🧠 Machine Learning Prediction")

    # Encode and scale
    le_country = LabelEncoder()
    le_location = LabelEncoder()
    df['Country_Code'] = le_country.fit_transform(df['Country'].astype(str))
    df['Location_Code'] = le_location.fit_transform(df['Location'].astype(str))

    X = df[['Country_Code', 'Location_Code']]
    y = df['Concentration_Num']
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    r2 = round(np.corrcoef(y_test_real, y_pred_real)[0, 1]**2, 4)
    st.write(f"**R² Score:** {r2}")

    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=y_test_real, y=y_pred_real, ax=ax3)
    ax3.set_xlabel("Actual Concentration")
    ax3.set_ylabel("Predicted Concentration")
    ax3.plot([min(y_test_real), max(y_test_real)],
             [min(y_test_real), max(y_test_real)], 'r--')
    st.pyplot(fig3)

    # Downloadable prediction file
    if st.button("📥 Download Predictions"):
        pred_df = pd.DataFrame({
            "Actual_Concentration": y_test_real,
            "Predicted_Concentration": y_pred_real
        })
        pred_df.to_csv("Predictions.csv", index=False)
        st.success("Predictions saved as Predictions.csv!")

else:
    st.info("Please upload your cleaned CSV file to begin.")
