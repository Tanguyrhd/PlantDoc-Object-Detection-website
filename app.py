import streamlit as st
import requests
from PIL import Image
import io
import base64

# Page config
st.set_page_config(
    page_title="Recognize and detect plant diseases",
    page_icon="🌿",
    layout="wide"
)

# API URL
API_URL = "https://plantdoc-api-645106012666.europe-west1.run.app"

# Title
st.title("🌿 Recognize and Detect Plant Diseases")
st.markdown("""
This app automatically analyzes your plant image through three steps:
1. 🌱 **Identify** the plant species
2. 🔍 **Detect** if the plant is healthy or diseased
3. 🦠 **Diagnose** the specific disease (if diseased)
""")

# Sidebar with model info
with st.sidebar:
    st.header("📊 Model Information")

    st.subheader("🌱 Supported Species (13)")
    species = [
        "Apple", "Bell Pepper", "Blueberry", "Cherry", "Corn",
        "Grape", "Peach", "Potato", "Raspberry", "Soybean",
        "Squash", "Strawberry", "Tomato"
    ]
    st.markdown("- " + "\n- ".join(species))

    st.subheader("🦠 Detected Diseases (27 classes)")
    diseases = [
        "Rust leaf", "Scab Leaf",
        "Leaf early blight", "Leaf late blight", "Leaf yellow virus",
        "Leaf mosaic virus", "Leaf bacterial spot",
        "Septoria leaf spot",
        "Powdery mildew leaf",
    ]
    st.markdown("- " + "\n- ".join(diseases))

    st.info("💡 The model has been trained on a dataset with 2,598 annotated images")

# Main interface
st.markdown("---")

# Upload
uploaded_file = st.file_uploader(
    "📤 Choose a plant leaf image from the list of supported species",
    type=["jpg", "jpeg", "png"]
)

# Warm-up: Load models on first page load
if 'models_warmed_up' not in st.session_state:
    try:
        # Send a dummy request to wake up the API
        # You can use a small dummy image or just ping the health endpoint if you have one
        # Try to ping the API (adjust if you have a health check endpoint)
        requests.get(f"{API_URL}/", timeout=30)
        st.session_state.models_warmed_up = True
    except:
        # If warm-up fails, continue anyway
        st.session_state.models_warmed_up = True

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)

    st.subheader("🖼️ Original Image")
    st.image(image, use_container_width=True)

    st.markdown("---")

    # Step 1: Species Identification
    st.markdown("### Step 1: 🌱 Species Identification")

    with st.spinner("Identifying the species"):

        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

        try:
            response = requests.post(f"{API_URL}/predict/species", files=files)
            response.raise_for_status()
            species_result = response.json()

            if "annotated_image" in species_result:
                img_data = base64.b64decode(species_result["annotated_image"].split(",")[1])
                img = Image.open(io.BytesIO(img_data))
                st.image(img, use_container_width=True)

            if species_result["predictions"]:
                st.markdown("Identification complete")
            else:
                st.warning("No species detected, are you sure that the species is in the list in the sidebar ?")
                st.stop()

        except Exception as e:
            st.error(f"Error during species identification: {str(e)}")
            st.stop()

    st.markdown("---")

    # Step 2: Binary Classification (Healthy or Diseased)
    st.markdown("### Step 2: 🔍 Health Status Detection")

    is_diseased = False

    with st.spinner("Detecting if plant is healthy or diseased..."):

        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

        try:
            response = requests.post(f"{API_URL}/predict/binary", files=files)
            response.raise_for_status()
            binary_result = response.json()

            if binary_result["predictions"]:
                for pred in binary_result["predictions"]:
                    class_name = pred['class_name']

                    if class_name.lower() == "disease":
                        is_diseased = True

        except Exception as e:
            st.error(f"Error during health status detection: {str(e)}")
            st.stop()

    if not is_diseased:
        st.success("✅ Your plant appears to be healthy! No further analysis needed.")
        st.stop()
    else:
        st.markdown("✅ Disease detected - proceeding to diagnosis...")

    st.markdown("---")

    # Step 3: Disease Diagnosis (only if diseased)
    st.markdown("### Step 3: 🦠 Disease Diagnosis")
    with st.spinner("Diagnosing specific disease..."):

        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

        try:
            response = requests.post(f"{API_URL}/predict/diseases", files=files)
            response.raise_for_status()
            disease_result = response.json()

            if "annotated_image" in disease_result:
                img_data = base64.b64decode(disease_result["annotated_image"].split(",")[1])
                img = Image.open(io.BytesIO(img_data))
                st.image(img, use_container_width=True)

            if disease_result["predictions"]:
                st.markdown("✅ Disease diagnosis complete!")
            else:
                st.warning("No specific disease identified")
                st.markdown("✅ Analysis complete!")

        except Exception as e:
            st.error(f"Error during disease diagnosis: {str(e)}")
            st.markdown("⚠️ Analysis completed with errors")

else:
        st.info("👆 Please upload an image to start the automatic analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔬 Based on the PlantDoc dataset with 2,598 annotated images</p>
    <p>📄 <a href='https://arxiv.org/abs/1911.10317'>Paper: PlantDoc: A Dataset for Visual Plant Disease Detection</a></p>
</div>
""", unsafe_allow_html=True)
