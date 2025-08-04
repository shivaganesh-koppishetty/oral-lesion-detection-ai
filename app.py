import streamlit as st
from PIL import Image
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
from cnnClassifier.pipeline.prediction import PredictionPipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Oral Lesion Detection",
    page_icon="🦷",
    layout="wide"
)

# --- HEADER ---
st.title("Oral Lesion Detection AI System")
st.caption("An intelligent assistant to help detect early signs of oral abnormalities.")

# --- TABS ---
tabs = st.tabs(["🏠 Home", "🔍 Prediction", "📈 Confidence Scores", "ℹ️ About"])

# --- HOME TAB ---
with tabs[0]:
    st.header("Welcome")
    st.write("""
    This application leverages deep learning to identify early-stage oral lesions using clinical images. 
    Upload a clear image of the oral cavity and the model will determine whether it is **Lesion** or **Non-Lesion**.
    """)
    st.markdown("---")
    st.subheader("Key Features")
    st.markdown("""
    - 🔬 Trained on clinical oral cavity datasets  
    - ⚙️ TensorFlow-based deep learning model  
    - 📤 Upload and get results instantly  
    - 📊 View detailed confidence scores
    """)

# --- PREDICTION TAB ---
with tabs[1]:
    st.header("Upload Image for Prediction")
    uploaded_file = st.file_uploader("Upload a clinical image of the mouth (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=150)

        extension = uploaded_file.type.split("/")[-1]
        format_ = "JPEG" if extension in ["jpg", "jpeg"] else extension.upper()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            image.save(temp_file.name, format=format_)

            pipeline = PredictionPipeline(temp_file.name)
            prediction_label = pipeline.predict()
            probabilities = pipeline.predict_proba()

            st.markdown("---")
            st.subheader("Prediction Result")
            if prediction_label == "Lesion":
                st.error(f"⚠️ **Detected**: {prediction_label}")
            else:
                st.success(f"✅ **Detected**: {prediction_label}")

            # Save prediction and probabilities for next tab
            st.session_state["class_names"] = list(probabilities.keys())
            st.session_state["class_probs"] = list(probabilities.values())

# --- CONFIDENCE SCORES TAB ---
with tabs[2]:
    st.header("Prediction Confidence")
    if "class_names" in st.session_state and "class_probs" in st.session_state:
        fig, ax = plt.subplots(figsize=(3, 1.5))  # Small size
        sns.barplot(
            x=st.session_state["class_probs"], 
            y=st.session_state["class_names"], 
            palette=["#ff4d4d" if name == "Lesion" else "#28a745" for name in st.session_state["class_names"]],
            ax=ax
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", fontsize=8)
        ax.set_title("Prediction Confidence", fontsize=10)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=False)
    else:
        st.info("⚠️ No predictions yet. Please upload an image from the 'Prediction' tab.")


# --- ABOUT TAB ---
with tabs[3]:
    st.header("About This Project")
    st.markdown("""
    - 🧠 **Model**: Custom CNN using TensorFlow/Keras  
    - 📁 **Artifacts**: Stored using DVC  
    - 💻 **Deployment**: Streamlit App  
    - 👨‍💻 **Developer**: *Shivaganesh Koppishetty*  
    
    **How It Works:**
    1. Upload a clinical oral image.
    2. The model classifies it as Lesion or Non-Lesion.
    3. Displays prediction and confidence.
    """)
    st.markdown("---")
    st.caption("🔍 For demonstration purposes only. Not intended for clinical diagnosis.")
