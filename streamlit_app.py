import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Load model and metadata
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("insect_classification_model.keras")
    with open("model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, metadata

model, metadata = load_model()
class_names = metadata["class_names"]

# ---------- GLOBAL STYLES ----------
st.set_page_config(
    page_title="Insect Classifier", 
    layout="centered",
    page_icon="🦋"
)

st.markdown("""
    <style>
    :root {
        --primary: #4f46e5;
        --primary-hover: #4338ca;
        --secondary: #f5f3ff;
        --accent: #10b981;
        --dark: #1e293b;
        --light: #f8fafc;
    }
    
    [data-testid="stAppViewContainer"] {
        background: #f9fafb;
    }
    
    .stButton button {
        width: 100%;
        background-color: white;
        color: black;
        border: none;
        border-radius: 8px;
        padding: 0.75em 1em;
        font-size: 1em;
        font-weight: 500;
        transition: all 0.2s ease;
        margin: 0.25em 0;
    }
    
    .stButton button:hover {
        background-color: var(--primary-hover);
        color: white;
    }
    
    .sidebar .stButton button {
        background-color: white;
        color: var(--primary);
        border: 1px solid var(--primary);
    }
    
    .sidebar .stButton button:hover {
        background-color: var(--secondary);
        color: white;
    }
    
    .sidebar .stButton button[kind="secondary"] {
        background-color: var(--primary);
        color: white;
    }
    
    .stFileUploader {
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 2rem;
        background-color: white;
    }
    
    .prediction-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-top: 1rem;
        color: black;
    }
    
    .confidence-meter {
        height: 8px;
        background: #e5e7eb;
        border-radius: 8px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
        border-radius: 8px;
    }
    
    .uploaded-image {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("""
        <h2 style='font-size: 50px; font-weight: bold; color: var(--primary);'>Insect Classifier</h2>
        <h5 style='margin-bottom: 1.5rem;'> by Group JEL.LY </h5>
    """, unsafe_allow_html=True)
    
    if st.button("Home", key="home_btn"):
        st.session_state.page = "Home"
    if st.button("About", key="about_btn"):
        st.session_state.page = "About"
    if st.button("Feedback", key="feedback_btn"):
        st.session_state.page = "Feedback"

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------- HOME PAGE ----------
if st.session_state.page == "Home":
    st.markdown("""
        <h1 style='color: var(--dark); margin-bottom: 0.5rem;'>Insect Classifier</h1>
        <p style='color: #64748b; margin-bottom: 2rem;'>
            Upload an insect image to identify its species
        </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized), axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("<div class='uploaded-image'>", unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            top_k = np.argsort(prediction[0])[::-1][:3]
            
            st.markdown(f"""
                <div class='prediction-card'>
                    <h3 style='margin-top: 0;'>Prediction: {predicted_class}</h3>
                    <p style='margin-bottom: 0.25rem;'>Confidence</p>
                    <div class='confidence-meter'>
                        <div class='confidence-fill' style='width: {confidence}%'></div>
                    </div>
                    <p style='text-align: right; margin-top: 0;'>{confidence:.1f}%</p>
                    
            """, unsafe_allow_html=True)

            st.markdown("<h4 style='color: black; margin-top: 0.5rem;'>Top Predictions</h4>", unsafe_allow_html=True)
            
            for i in top_k:
                label = class_names[i]
                conf = prediction[0][i] * 100
                st.markdown(f"""
                    <div style='color: black; display: flex; justify-content: space-between; margin: 0.5rem 0;'>
                        <span>{label}</span>
                        <span style='font-weight: 500;'>{conf:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
            <div style='text-align: center; margin: 3rem 0; color: #64748b;'>
                <svg xmlns='http://www.w3.org/2000/svg' width='48' height='48' viewBox='0 0 24 24' fill='none' 
                    stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' 
                    style='opacity: 0.5; margin-bottom: 1rem;'>
                    <path d='M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z'></path>
                    <circle cx='12' cy='10' r='3'></circle>
                </svg>
                <p>Upload an insect image to get started</p>
            </div>
        """, unsafe_allow_html=True)

# ---------- ABOUT PAGE ----------
elif st.session_state.page == "About":
    st.markdown("""
        <h1 style='color: black; margin-bottom: 1rem;'>About</h1>
        
        <div style='color: black; background: white; border-radius: 12px; padding: 1.5rem; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
            <p style='line-height: 1.6;'>
                This insect classifier uses deep learning to identify various insect species from images. 
                Upload a clear photo of an insect to get started.
            </p>
        </div>
        
        <h3 style='margin-bottom: 0.5rem;'>Features</h3>
        <ul style='margin-top: 0; padding-left: 1.2rem;'>
            <li>Identify insect species from images</li>
            <li>View confidence scores for predictions</li>
            <li>Simple and easy to use</li>
        </ul>
        
        <p style='margin-top: 2rem;'>
            Created by <strong>Team JEL.LY</strong>
        </p>
    """, unsafe_allow_html=True)

# ---------- FEEDBACK PAGE ----------
elif st.session_state.page == "Feedback":
    st.markdown("""
        <h1 style='color: var(--dark); margin-bottom: 1rem;'>Feedback</h1>
        <p style='color: #64748b; margin-bottom: 1.5rem;'>
            We appreciate your feedback to improve our classifier.
        </p>
    """, unsafe_allow_html=True)
    
    with st.form("feedback_form"):
        st.markdown("<span style='color: black;'>Your name (optional):</span>", unsafe_allow_html=True)
        name = st.text_input("", label_visibility="collapsed")
        
        st.markdown("<span style='color: black;'>Your feedback:</span>", unsafe_allow_html=True)
        feedback = st.text_area("", height=150, label_visibility="collapsed")
        
        submitted = st.form_submit_button("Submit Feedback")

# ---------- FOOTER ----------
st.markdown("""
    <div class='footer'>
        <p>© 2023 Insect Classifier | Team JEL.LY</p>
    </div>
""", unsafe_allow_html=True)