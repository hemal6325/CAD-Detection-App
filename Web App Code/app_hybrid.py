import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib
import os

# Page config
st.set_page_config(page_title="CAD Detection App", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0a0e27;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77e4;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1560c0;
    }
    .result-box {
        background-color: #1a1f3a;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77e4;
        margin: 20px 0;
    }
    .metric-box {
        background-color: #1a1f3a;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🫀 Coronary Artery Disease (CAD) Detection")
st.markdown("**Using Deep Learning on CT Images**")

# ===== LOAD MODEL & FEATURE EXTRACTOR =====
@st.cache_resource
def load_models():
    # Feature extractor (ResNet18)
    feature_extractor = models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.eval()
    
    # Random Forest model
    model_path = 'hybrid_cad_model.pkl'
    if os.path.exists(model_path):
        rf_model = joblib.load(model_path)
        return feature_extractor, rf_model
    else:
        st.error("❌ Model file not found!")
        return feature_extractor, None

feature_extractor, rf_model = load_models()

# ===== IMAGE PROCESSING =====
def extract_features(image):
    """CT image থেকে features extract করো"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = feature_extractor(img_tensor)
        features = features.squeeze().cpu().numpy()
    
    return features

# ===== MAIN APP =====
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload CT Image")
    uploaded_file = st.file_uploader("Choose a CT image (PNG, JPG)", 
                                     type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image")

with col2:
    st.subheader("📊 Prediction Results")
    
    if uploaded_file is not None and rf_model is not None:
        # Extract features
        with st.spinner("🔄 Processing image..."):
            features = extract_features(image)
            features_reshaped = features.reshape(1, -1)
        
        # Make prediction
        prediction = rf_model.predict(features_reshaped)[0]
        probability = rf_model.predict_proba(features_reshaped)[0]
        
        # Confidence score
        cad_probability = probability[1] * 100
        
        # ===== RISK LEVEL LOGIC (আপডেট করা) =====
        if cad_probability < 45:
            risk_level = "🟢 UNCERTAIN"
            risk_color = "#FFD700"  # Yellow
            risk_message = "Need more tests"
            risk_desc = "Confidence too low for diagnosis"
        elif cad_probability < 60:
            risk_level = "🟡 LOW-MODERATE RISK"
            risk_color = "#FFA500"  # Orange
            risk_message = "Requires Investigation"
            risk_desc = "Additional imaging recommended"
        elif cad_probability < 75:
            risk_level = "🔴 MODERATE-HIGH RISK"
            risk_color = "#FF6B6B"  # Red
            risk_message = "CAD Suspected"
            risk_desc = "Clinical follow-up needed"
        elif cad_probability < 90:
            risk_level = "🔴 HIGH RISK"
            risk_color = "#FF4444"  # Darker Red
            risk_message = "CAD Likely Present"
            risk_desc = "Urgent clinical intervention"
        else:
            risk_level = "⛔ VERY HIGH RISK"
            risk_color = "#CC0000"  # Dark Red
            risk_message = "Severe CAD Detected"
            risk_desc = "Critical - Immediate action required"
        
        # Display results
        st.markdown(f"""
        <div class="result-box" style="border-color: {risk_color}; border-width: 3px;">
            <h2 style="color: {risk_color}; text-align: center;">{risk_level}</h2>
            <h3 style="text-align: center;">{risk_message}</h3>
            <p style="text-align: center; font-size: 18px; color: {risk_color};">
                Confidence: {cad_probability:.2f}%
            </p>
            <p style="text-align: center; font-size: 14px; color: #cccccc;">
                {risk_desc}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed metrics
        st.markdown("### 📈 Detailed Analysis")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("CAD Probability", f"{probability[1]*100:.2f}%")
        
        with col_b:
            st.metric("Non-CAD Probability", f"{probability[0]*100:.2f}%")
        
        # Risk bar
        st.markdown("### 📊 Risk Spectrum")
        risk_percentage = cad_probability
        st.progress(risk_percentage / 100, text=f"Risk Level: {risk_percentage:.1f}%")

# ===== MODEL INFO =====
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Information")
st.sidebar.info("""
**Model**: Hybrid CNN + Random Forest
- **Features**: ResNet18 (512 features)
- **Classifier**: Random Forest
- **Accuracy**: 88.89%
- **Recall**: 100.00%
- **F1-Score**: 94.12%

**Dataset**: 42 CT Scans Patients & Total Images: 18,061
- CAD Positive: 38
- CAD Negative: 4

**Risk Levels**:
- 🟢 < 45% = UNCERTAIN
- 🟡 45-60% = LOW-MODERATE
- 🔴 60-75% = MODERATE-HIGH
- 🔴 75-90% = HIGH RISK
- ⛔ > 90% = VERY HIGH RISK
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Instructions or Working Process!")
st.sidebar.markdown("""
1. CT image upload করো
2. Model features extract করবে
3. Risk level দেখাবে
4. বিভিন্ন images test করো

**Note**: এটি একটি AI-based screening tool। চূড়ান্ত diagnosis এর জন্য cardiologist দেখান লাগবে।
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Threshold Ranges")
st.sidebar.markdown("""
| Risk Level | Confidence | Action |
|-----------|-----------|--------|
| UNCERTAIN | < 45% | Need more tests |
| LOW-MODERATE | 45-60% | Further investigation |
| MODERATE-HIGH | 60-75% | Clinical follow-up |
| HIGH RISK | 75-90% | Urgent intervention |
| VERY HIGH | > 90% | Critical action |
""")
