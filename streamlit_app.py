import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import time
import matplotlib.pyplot as plt  # ← Add this

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

st.markdown("""
    <style>
    :root {
        --primary: #4f46e5;
        --primary-hover: #4338ca;
        --secondary: #f5f3ff;
        --accent: #10b981;
        --dark: #00000;
        --light: #f8fafc;
    }
            
    body, p, h1, h2, h3, h4, h5, h6, div, span {
        color: black !important;
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

    st.subheader("Upload or Take a Photo")
    tab1, tab2 = st.tabs(["Upload Image", "Take Photo"])

    image = None
    with tab1:
        uploaded_file = st.file_uploader("Upload an insect image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    with tab2:
        camera_file = st.camera_input("Take a photo")
        if camera_file:
            image = Image.open(camera_file).convert("RGB")

    if image:
        st.markdown("<div class='image-preview'>", unsafe_allow_html=True)
        st.image(image, caption="Selected Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img), axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Progress bar simulation
        st.subheader("Processing...")
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)

        CONFIDENCE_THRESHOLD = 0.70

        with st.spinner("Analyzing image..."):
            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx]

            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class = class_names[predicted_class_idx]

                # Get top 3 predictions
                top_3_indices = prediction[0].argsort()[-3:][::-1]
                top_3 = [(class_names[i], prediction[0][i]) for i in top_3_indices]

                st.markdown(f"<div class='prediction'>Prediction: <strong>{predicted_class}</strong></div>", unsafe_allow_html=True)
                st.markdown("<div class='confidence-badge'>Top 3 guesses:</div>", unsafe_allow_html=True)

                for label, conf in top_3:
                    st.markdown(f"- {label}: **{conf*100:.2f}%**")
            else:
                predicted_class = class_names[predicted_class_idx]
                st.markdown(f"<div class='prediction'>Prediction: <strong>{predicted_class}</strong></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='confidence-badge'>Confidence: {confidence*100:.2f}%</div>", unsafe_allow_html=True)

        # Plot all class probabilities
        st.subheader("Prediction Probabilities for All Classes")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(class_names, prediction[0], color='#00b4d8')
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}', va='center', fontsize=9, color='white')
        st.pyplot(fig)

    st.markdown("""
    ---
    This app was built using TensorFlow and Streamlit.

    The model is based on work by [vencerlanz09 on Kaggle](https://www.kaggle.com/code/vencerlanz09/sea-animals-classification-using-efficeintnetb7), and the dataset includes 23 Insect classes such as clams, corals, crabs, dolphins, sharks, turtles, and more.
    """)

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