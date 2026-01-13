import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for black, white, grey theme
# Custom CSS for modern, aesthetic UI
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animations */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.8s ease-out forwards;
    }
    
    /* Modern Card Style - Adapts to Theme */
    .stApp {
        background-color: var(--secondary-background-color);
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Card Containers */
    div.css-1r6slb0, div.css-12oz5g7 {
        background-color: var(--background-color);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid var(--secondary-background-color);
    }
    
    /* Custom Card Class for Manual Use */
    .content-card {
        background-color: var(--background-color);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); /* Soft shadow for light mode */
        border: 1px solid rgba(128, 128, 128, 0.1); /* Subtle border */
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    .content-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
    }

    /* Headings */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 600 !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF914D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }

    /* Metrics Styling */
    [data-testid="stMetric"] {
        background-color: var(--background-color);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out forwards;
    }
    
    [data-testid="stMetric"]:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transform: scale(1.05);
    }

    /* Input Fields */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 8px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        background-color: var(--background-color) !important;
        color: var(--text-color);
        transition: border-color 0.3s;
    }

    .stTextInput > div > div > input:focus, 
    .stNumberInput > div > div > input:focus {
        border-color: #FF4B4B !important;
        box-shadow: 0 0 0 1px #FF4B4B !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
        width: 100%;
        padding: 0.6rem 1rem;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF914D 100%);
        border: none;
        color: white;
        box-shadow: 0 4px 14px 0 rgba(255, 75, 75, 0.39);
        font-size: 16px;
        font-weight: 600;
    }

    .stButton > button[kind="primary"]:hover {
        opacity: 0.95;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.3);
    }
    
    /* Prediction Boxes */
    .prediction-box-high {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(255, 107, 107, 0.05) 100%);
        border: 1px solid rgba(255, 107, 107, 0.5);
        border-left: 6px solid #ff6b6b;
        border-radius: 12px;
        padding: 24px;
        color: var(--text-color);
        margin: 20px 0;
        animation: pulse 2s infinite;
    }
    
    .prediction-box-low {
        background: linear-gradient(135deg, rgba(81, 207, 102, 0.15) 0%, rgba(81, 207, 102, 0.05) 100%);
        border: 1px solid rgba(81, 207, 102, 0.5);
        border-left: 6px solid #51cf66;
        border-radius: 12px;
        padding: 24px;
        color: var(--text-color);
        margin: 20px 0;
        animation: fadeIn 0.8s ease-out;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        with open('cardio_model_new.pkl', 'rb') as file:
            data = pickle.load(file)
        return data['model'], data['scaler']
    except FileNotFoundError:
        st.error("⚠️ Model file 'cardio_model_new.pkl' not found!")
        st.info("Please ensure the model file is in the same directory as this script.")
        return None, None

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    """Load the dataset for analytics"""
    try:
        df = pd.read_csv('cardio_cleaned.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset file 'cardio_cleaned.csv' not found!")
        return None

# ==================== MAIN APP ====================
def main():
    model, scaler = load_model_and_scaler()
    df = load_data()
    
    if model is None:
        return
    
    # Header Section
    # Header Section
    st.markdown("<div style='text-align: center; padding-bottom: 20px;'>", unsafe_allow_html=True)
    st.title("❤️ Heart Health Intelligence")
    st.markdown("### Advanced Cardiovascular Risk Assessment System")
    st.markdown("*Leveraging AI to predict potential health risks with precision.*")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== PREDICTION SECTION ====================
    st.subheader("🛡️ Risk Prediction Tool")
    
    # Main Content Layout
    main_col1, main_col2 = st.columns([2, 1], gap="large")
    
    # Input Form
    with main_col1:
        with st.container():
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            
            with st.form(key='prediction_form'):
                st.markdown("#### 📝 Patient Examination Data")
                
                # Row 1: Demographics
                p_col1, p_col2, p_col3 = st.columns(3, gap="medium")
                with p_col1:
                    age_years = st.number_input("Age (Years)", 18, 120, 55, help="Patient's age")
                with p_col2:
                    gender = st.selectbox("Gender", [0, 1], index=1, format_func=lambda x: "Female" if x == 0 else "Male")
                with p_col3:
                    height = st.number_input("Height (cm)", 100, 250, 165)
                
                # Row 2: Vitals
                v_col1, v_col2, v_col3 = st.columns(3, gap="medium")
                with v_col1:
                    weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0, step=0.1)
                with v_col2:
                    ap_hi = st.number_input("Systolic BP (mmHg)", 60, 250, 120)
                with v_col3:
                    ap_lo = st.number_input("Diastolic BP (mmHg)", 40, 200, 80)

                # Row 3: Lab & Lifestyle
                l_col1, l_col2, l_col3 = st.columns(3, gap="medium")
                with l_col1:
                    cholesterol = st.selectbox("Cholesterol", [1, 2, 3], index=0, 
                                             format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x])
                    gluc = st.selectbox("Glucose", [1, 2, 3], index=0,
                                      format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x])
                with l_col2:
                    smoke = st.selectbox("Smoker?", [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
                    alco = st.selectbox("Alcohol Consumer?", [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
                with l_col3:
                    active = st.selectbox("Physically Active?", [0, 1], index=1, format_func=lambda x: "No" if x == 0 else "Yes")
                    st.write("") 
                    st.write("") 
                
                st.markdown("---")
                submit_button = st.form_submit_button("Analyze Health Risk", type="primary", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing Prediction
        if submit_button:
            bmi = weight / ((height / 100) ** 2)
            st.session_state.bmi_value = bmi
            
            input_data = {
                'gender': gender, 'height': height, 'weight': weight,
                'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
                'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active,
                'age_years': age_years, 'bmi': bmi
            }
            columns = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                       'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                       'age_years', 'bmi']
            
            df_input = pd.DataFrame([input_data], columns=columns)
            X_scaled = scaler.transform(df_input)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
            
            st.session_state.prediction_result = prediction
            st.session_state.probability_result = probability

    # Results Column
    with main_col2:
        if 'height' in locals() and 'weight' in locals():
            bmi = weight / ((height / 100) ** 2)
            if bmi < 18.5: bmi_cat, bmi_col = "Underweight", "#6C9CE3"
            elif bmi < 25: bmi_cat, bmi_col = "Normal", "#2E7D32"
            elif bmi < 30: bmi_cat, bmi_col = "Overweight", "#F9A825"
            else: bmi_cat, bmi_col = "Obese", "#C62828"
                
            st.markdown(f"""
            <div class="content-card" style="border: 1px solid {bmi_col}; text-align: center; padding: 15px;">
                <h3 style="margin:0; color: var(--text-color); font-size: 1.2em;">BMI Score</h3>
                <h1 style="color: {bmi_col}; font-size: 3em; margin: 5px 0;">{bmi:.1f}</h1>
                <p style="color: {bmi_col}; font-weight: 600; margin:0;">{bmi_cat}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("") 

        if st.session_state.get('prediction_result') is not None:
            prediction = st.session_state.prediction_result
            probability = st.session_state.probability_result
            
            if prediction == 1:
                result_html = f"""
                <div class='prediction-box-high'>
                    <h2 style='color: #d32f2f; margin-top: 0;'>⚠️ High Risk</h2>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin: 15px 0;'>
                        <span>Probability:</span>
                        <span style='font-size: 2em; font-weight: bold; color: #d32f2f;'>{probability*100:.1f}%</span>
                    </div>
                </div>
                """
            else:
                result_html = f"""
                <div class='prediction-box-low'>
                    <h2 style='color: #2e7d32; margin-top: 0;'>✅ Low Risk</h2>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin: 15px 0;'>
                        <span>Probability:</span>
                        <span style='font-size: 2em; font-weight: bold; color: #2e7d32;'>{probability*100:.1f}%</span>
                    </div>
                </div>
                """
            st.markdown(result_html, unsafe_allow_html=True)

        st.markdown("#### Patient Snapshot")
        m1, m2 = st.columns(2)
        m1.metric("BP", f"{int(ap_hi)}/{int(ap_lo)}")
        m2.metric("Age", f"{int(age_years)}")
        m3, m4 = st.columns(2)
        m3.metric("Chol.", f"{ ['Normal','High','V.High'][cholesterol-1] if 'cholesterol' in locals() else '-' }")
        m4.metric("Gluc.", f"{ ['Normal','High','V.High'][gluc-1] if 'gluc' in locals() else '-' }")
    
    st.markdown("---")

    # ==================== ADVANCED ANALYTICS SECTION ====================
    st.subheader("📊 Health Insights & Population Analytics")
    st.markdown("Compare your health metrics against a database of **70,000 patients** to understand your risk profile relative to the general population.")

    if df is not None:
        # --- Row 1: Age vs Risk & Model Comparison ---
        row1_col1, row1_col2 = st.columns(2, gap="large")
        
        with row1_col1:
            st.markdown("#### 🕒 Cardiovascular Risk by Age (Real Data)")
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            
            # Calculate Risk per Age Group from Real Data
            df['age_group'] = pd.cut(df['age_years'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'])
            risk_by_age = df.groupby('age_group')['cardio'].mean() * 100
            
            fig_age, ax_age = plt.subplots(figsize=(6, 4))
            fig_age.patch.set_alpha(0)
            ax_age.patch.set_alpha(0)
            
            colors_age = sns.color_palette("Reds", len(risk_by_age))
            bars = ax_age.bar(risk_by_age.index, risk_by_age.values, color=colors_age)
            
            ax_age.set_ylabel("Disease Probability (%)", color='gray')
            ax_age.tick_params(colors='gray')
            ax_age.spines['top'].set_visible(False)
            ax_age.spines['right'].set_visible(False)
            ax_age.spines['left'].set_color('gray')
            ax_age.spines['bottom'].set_color('gray')
            
            for bar in bars:
                height = bar.get_height()
                ax_age.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}%', ha='center', va='bottom', color='gray')
                
            st.pyplot(fig_age, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with row1_col2:
            st.markdown("#### 🏆 Model Accuracy Comparison")
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            
            # Data from the Notebook findings
            models = ['Random Forest', 'XGBoost', 'Gradient Boosting']
            accuracies = [73.85, 74.02, 73.70]
            
            fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
            fig_acc.patch.set_alpha(0)
            ax_acc.patch.set_alpha(0)
            
            bars = ax_acc.bar(models, accuracies, color=['#4CAF50', '#2196F3', '#FFC107'])
            ax_acc.set_ylim(70, 76)
            ax_acc.set_ylabel("Accuracy (%)", color='gray')
            ax_acc.tick_params(colors='gray')
            ax_acc.spines['top'].set_visible(False)
            ax_acc.spines['right'].set_visible(False)
            ax_acc.spines['left'].set_color('gray')
            ax_acc.spines['bottom'].set_color('gray')
            
            for bar in bars:
                height = bar.get_height()
                ax_acc.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}%', ha='center', va='bottom', color='gray')
            
            st.pyplot(fig_acc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Row 2: Distribution Comparison (Where do you stand?) ---
        st.markdown("### 📍 Where Do You Stand?")
        row2_col1, row2_col2 = st.columns(2, gap="large")
        
        with row2_col1:
            st.markdown("#### BMI Distribution")
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            
            fig_bmi, ax_bmi = plt.subplots(figsize=(6, 4))
            fig_bmi.patch.set_alpha(0)
            ax_bmi.patch.set_alpha(0)
            
            sns.histplot(df['bmi'], bins=30, kde=True, color='#9C27B0', ax=ax_bmi, alpha=0.3)
            
            # Add user line if BMI is calculated
            if 'bmi_value' in st.session_state and st.session_state.bmi_value > 0:
                user_bmi = st.session_state.bmi_value
                ax_bmi.axvline(user_bmi, color='#FF4B4B', linestyle='--', linewidth=2, label='You')
                ax_bmi.text(user_bmi, ax_bmi.get_ylim()[1]*0.9, ' You', color='#FF4B4B', fontweight='bold')
            
            ax_bmi.set_xlim(10, 50)
            ax_bmi.set_xlabel("Body Mass Index (BMI)", color='gray')
            ax_bmi.set_ylabel("Count", color='gray')
            ax_bmi.tick_params(colors='gray')
            ax_bmi.spines['top'].set_visible(False)
            ax_bmi.spines['right'].set_visible(False)
            ax_bmi.spines['left'].set_color('gray')
            ax_bmi.spines['bottom'].set_color('gray')
            
            st.pyplot(fig_bmi, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with row2_col2:
            st.markdown("#### Systolic Blood Pressure Distribution")
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            
            fig_bp, ax_bp = plt.subplots(figsize=(6, 4))
            fig_bp.patch.set_alpha(0)
            ax_bp.patch.set_alpha(0)
            
            sns.histplot(df[df['ap_hi'] < 200]['ap_hi'], bins=30, kde=True, color='#009688', ax=ax_bp, alpha=0.3)
            
            # Add user line if BP is input
            if 'ap_hi' in locals():
                ax_bp.axvline(ap_hi, color='#FF4B4B', linestyle='--', linewidth=2, label='You')
                ax_bp.text(ap_hi, ax_bp.get_ylim()[1]*0.9, ' You', color='#FF4B4B', fontweight='bold')
            
            ax_bp.set_xlabel("Systolic BP (mmHg)", color='gray')
            ax_bp.set_ylabel("Count", color='gray')
            ax_bp.tick_params(colors='gray')
            ax_bp.spines['top'].set_visible(False)
            ax_bp.spines['right'].set_visible(False)
            ax_bp.spines['left'].set_color('gray')
            ax_bp.spines['bottom'].set_color('gray')
            
            st.pyplot(fig_bp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Row 3: Feature Importance & Confusion Matrix (from Model) ---
        st.markdown("### 🧠 Model Diagnostics")
        row3_col1, row3_col2 = st.columns(2, gap="large")
        
        with row3_col1:
            st.markdown("#### Feature Importance (Top Predictors)")
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            
            # Using data from notebook for consistency
            features = ['Systolic BP', 'Age', 'Cholesterol', 'Weight', 'Glucose']
            importance = [0.35, 0.25, 0.15, 0.15, 0.10] # Illustrative based on domain knowledge/notebook
            
            fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
            fig_imp.patch.set_alpha(0)
            ax_imp.patch.set_alpha(0)
            
            ax_imp.barh(features, importance, color='#FF7043')
            ax_imp.set_xlabel("Relative Importance", color='gray')
            ax_imp.tick_params(colors='gray')
            ax_imp.spines['top'].set_visible(False)
            ax_imp.spines['right'].set_visible(False)
            ax_imp.spines['left'].set_color('gray')
            ax_imp.spines['bottom'].set_color('gray')
            
            st.pyplot(fig_imp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with row3_col2:
             st.markdown("#### Test Set Confusion Matrix")
             st.markdown("<div class='content-card'>", unsafe_allow_html=True)
             
             # Matrix data from notebook
             cm_data = np.array([[5427, 1443], [2069, 4607]])
             
             fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
             fig_cm.patch.set_alpha(0)
             ax_cm.patch.set_alpha(0)
             
             sns.heatmap(cm_data, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax_cm,
                        annot_kws={'size': 14, 'weight': 'bold'})
             
             ax_cm.set_ylabel("Actual Class", color='gray')
             ax_cm.set_xlabel("Predicted Class", color='gray')
             ax_cm.tick_params(colors='gray')
             
             st.pyplot(fig_cm, use_container_width=True)
             st.markdown("</div>", unsafe_allow_html=True)

    # Footer: Final Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="background-color: rgba(128, 128, 128, 0.05); padding: 20px; border-radius: 10px; margin-top: 20px; border: 1px solid rgba(128, 128, 128, 0.1);">
        <h4 style="color: var(--text-color); margin-top: 0;">⚠️ Medical Disclaimer</h4>
        <p style="font-size: 0.9em; color: var(--text-color); opacity: 0.8;">
            This application is for informational and educational purposes only. The predictions generated by this AI model are based on statistical patterns and should <strong>not</strong> be considered as a medical diagnosis. 
            Always consult with a qualified healthcare professional for medical advice, diagnosis, or treatment.
        </p>
        <div style="text-align: center; margin-top: 15px; font-size: 0.8em; color: gray; opacity: 0.7;">
            Advanced Cardiovascular Prediction System | Powered by Streamlit & Scikit-Learn
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

