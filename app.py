import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --- Load model ---
@st.cache_resource
def load_model():
    return joblib.load("car_price_predictor_model.pkl")

# --- Load and clean data ---
@st.cache_data
def load_data():
    df = pd.read_csv("cleanedcar.csv")

    # Normalize relevant columns (strip and title case)
    for col in ["Brand", "model", "Transmission", "Owner", "FuelType"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Drop unnecessary columns if they exist
    for col in ["Unnamed: 0", "AdditionInfo", "PostedDate"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

df = load_data()
model = load_model()

# Prepare dropdown options
brand_options = sorted(df["Brand"].dropna().unique())
transmission_options = sorted(df["Transmission"].dropna().unique())
owner_options = sorted(df["Owner"].dropna().unique())
fuel_options = sorted(df["FuelType"].dropna().unique())

# Map brand to sorted model list
brand_model_dict = df.groupby("Brand")["model"].unique().apply(lambda x: sorted(x)).to_dict()

# App title with emoji and style
st.markdown(
    """
    <h1 style='text-align: center; color: #0B486B; font-weight: bold;'>
        🚗 Used Car Price Predictor
    </h1>
    <p style='text-align: center; color: #345678; font-size: 18px;'>
        Fill in the details below to estimate your car's market price.
    </p>
    <hr style='border: 1px solid #ddd'>
    """,
    unsafe_allow_html=True,
)

current_year = datetime.now().year

# Brand selector outside form with wider layout
brand = st.selectbox(
    "Select Brand 🚙",
    brand_options,
    index=0,
    help="Choose the brand of your car"
)

with st.form("car_form"):
    st.markdown("### Car Details")
    # Columns for a neat layout
    col1, col2 = st.columns(2)

    # Model selector based on brand
    model_options = brand_model_dict.get(brand, [])
    if not model_options:
        st.warning(f"No models found for brand '{brand}'. Please select another brand.")
        st.stop()

    with col1:
        model_name = st.selectbox("Model 🏷️", model_options)
        year = st.number_input(
            "Manufacturing Year 📅",
            min_value=1990,
            max_value=current_year,
            value=2018,
            help="Year the car was manufactured"
        )
        transmission = st.selectbox(
            "Transmission ⚙️",
            transmission_options,
            help="Select the transmission type"
        )

    with col2:
        km_driven = st.number_input(
            "Kilometers Driven 🚦",
            min_value=0,
            step=1000,
            value=50000,
            help="Total kilometers the car has driven"
        )
        owner = st.selectbox(
            "Owner Type 👤",
            owner_options,
            help="Type of owner"
        )
        fuel_type = st.selectbox(
            "Fuel Type ⛽",
            fuel_options,
            help="Type of fuel the car uses"
        )

    st.markdown("---")

    submitted = st.form_submit_button("Predict Price 💰")

    if submitted:
        age = current_year - year
        input_data = pd.DataFrame([{
            "Brand": brand,
            "model": model_name,
            "Year": year,
            "Age": age,
            "kmDriven": km_driven,
            "Transmission": transmission,
            "Owner": owner,
            "FuelType": fuel_type
        }])

        try:
            predicted_price = model.predict(input_data)[0]
            price_int = int(predicted_price)

            st.markdown(
                f"""
                <div style='
                    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    color: #7a2a0e;
                    font-weight: 700;
                    font-size: 24px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                '>
                    💰 Estimated Car Price: ₹{price_int:,}
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
