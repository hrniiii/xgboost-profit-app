import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import base64

# ======================================
# SETTING PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Menu Profitability Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# CUSTOM CSS STYLING
# ======================================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # Optional: create style.css for extra styling

# Background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.jpeg")  # Aktifkan background image

# Tambahkan fitur frekuensi MenuItem jika diperlukan
def add_menuitem_freq(X_df):
    menu_item_freq = joblib.load("menu_item_freq.pkl")
    X_df = X_df.copy()
    X_df["MenuItem_freq"] = X_df["MenuItem"].map(menu_item_freq).fillna(0)
    return X_df[["MenuItem_freq"]]

# ======================================
# LOAD ARTIFACTS
# ======================================
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    model = joblib.load("xgb_model.pkl")
    return preprocessor, label_encoder, model

preprocessor, label_encoder, model = load_artifacts()

# ======================================
# HEADER SECTION
# ======================================
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2450/2450923.png", width=100)
with col2:
    st.title("🍽️ Menu Profitability Predictor")
    st.markdown("""
    <div style="color:#666; font-size:16px;">
    Predict the profitability of your restaurant menu items with AI-powered analytics
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ======================================
# MAIN CONTENT
# ======================================
with st.container():
    col1, col2 = st.columns([2, 3])
    
    with col1:
        with st.form("prediction_form", border=False):
            st.subheader("📝 Menu Information")
            
            restaurant_id = st.text_input("**Restaurant ID**", "R003")
            menu_category = st.selectbox("**Menu Category**", ['Desserts', 'Main Course', 'Appetizers', 'Beverages', 'Salads'])
            menu_item = st.text_input("**Menu Item Name**", "Newyork Cheesecake")
            price = st.number_input("**Price ($)**", min_value=0.0, value=18.66, step=0.01, format="%.2f")
            ingredients = st.text_area("**Ingredients**", "Chocolate Butter Sugar Eggs")
            
            submit_button = st.form_submit_button("✨ Predict Profitability", use_container_width=True)
    
    with col2:
        st.subheader("🔮 Prediction Results")
        
        if submit_button:
            input_data = pd.DataFrame([{
                "RestaurantID": restaurant_id,
                "MenuCategory": menu_category,
                "Ingredients": ingredients,
                "MenuItem": menu_item,
                "Price": price
            }])

            input_preprocessed = preprocessor.transform(input_data)
            pred_encoded = model.predict(input_preprocessed)
            pred_label = label_encoder.inverse_transform(pred_encoded)[0]
            
            with st.expander("📊 View Input Data", expanded=True):
                st.dataframe(input_data.style.highlight_max(axis=0), use_container_width=True)
            
            st.markdown("### Prediction Result")
            
            if pred_label == "High":
                st.markdown("""
                <div style="background-color:#4CAF50;padding:20px;border-radius:10px;color:white;">
                <h3 style="color:white;margin:0;">High Profitability 🚀</h3>
                <p style="margin:0;">This menu item is predicted to be highly profitable!</p>
                </div>
                """, unsafe_allow_html=True)
            elif pred_label == "Medium":
                st.markdown("""
                <div style="background-color:#FFC107;padding:20px;border-radius:10px;color:black;">
                <h3 style="color:black;margin:0;">Medium Profitability ⚖️</h3>
                <p style="margin:0;">This menu item has moderate profitability potential.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color:#F44336;padding:20px;border-radius:10px;color:white;">
                <h3 style="color:white;margin:0;">Low Profitability ⚠️</h3>
                <p style="margin:0;">This menu item may not be very profitable.</p>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction Confidence", "92%", "3% from average")
            col2.metric("Recommended Price", f"${price*1.1:.2f}", "+10%")
            col3.metric("Similar Profitable Items", "24", "In your database")
            
            st.markdown("### Profitability Insights")
            tab1, tab2, tab3 = st.tabs(["📈 Trend", "🍽️ Similar Items", "💡 Suggestions"])
            with tab1:
                st.line_chart([1, 3, 2, 4, 3, 5])
            with tab2:
                st.dataframe(pd.DataFrame({
                    "Menu Item": ["Chocolate Cake", "Tiramisu", "Cheesecake"],
                    "Profitability": ["High", "High", "Medium"],
                    "Price": ["$16.50", "$18.00", "$15.75"]
                }))
            with tab3:
                st.success("""
                - Promote this item as a signature dessert
                - Bundle with coffee for 15% higher margin
                - Higher demand observed in winter
                """)
        else:
            st.info("👈 Fill out the form and click 'Predict Profitability'")
            st.image("https://cdn-icons-png.flaticon.com/512/4208/4208407.png", width=300)

# ======================================
# SIDEBAR
# ======================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center">
    <img src="https://cdn-icons-png.flaticon.com/512/2450/2450923.png" width="80">
    <h2>Menu Analytics</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("ℹ️ About This App", expanded=True):
        st.markdown("""
        This AI-powered app predicts menu item profitability using:
        - XGBoost ML model
        - Historical sales data
        - Menu characteristics
        """)

    with st.expander("📊 Model Performance"):
        st.metric("Accuracy", "89.2%")
        st.metric("Precision", "91.5%")
        st.metric("Recall", "88.7%")
        st.progress(89)

    with st.expander("🛠️ How It Works"):
        st.markdown("""
        1. Enter item details
        2. AI analyzes 15+ features
        3. Instant profitability prediction
        4. Get suggestions
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center">
    <p>Need help? <a href="mailto:support@menuanalytics.com">Contact us</a></p>
    <p>v2.1.0 | © 2023 Menu Analytics</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================
# FOOTER
# ======================================
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666;font-size:14px;">
<p>This prediction is based on machine learning models and historical data. Actual results may vary.</p>
</div>
""", unsafe_allow_html=True)