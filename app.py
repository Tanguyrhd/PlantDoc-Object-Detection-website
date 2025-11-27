import streamlit as st
import requests
from PIL import Image
import io
import base64

# Configuration de la page
st.set_page_config(
    page_title="PlantDoc - Détection de Maladies",
    page_icon="🌿",
    layout="wide"
)

# URL de l'API
API_URL = "https://plantdoc-api-645106012666.europe-west1.run.app"

# Titre principal
st.title("🌿 PlantDoc - Détection de Maladies des Plantes")
st.markdown("""
Cette application permet de :
- 🔍 **Détecter** si une plante est saine ou malade
- 🌱 **Identifier** l'espèce de la plante
- 🦠 **Diagnostiquer** la maladie spécifique si présente
""")

# Sidebar avec les informations sur les modèles
with st.sidebar:
    st.header("📊 Informations sur les modèles")

    st.subheader("🌱 Espèces supportées (13)")
    species = [
        "Apple", "Bell Pepper", "Blueberry", "Cherry", "Corn",
        "Grape", "Peach", "Potato", "Raspberry", "Soybean",
        "Squash", "Strawberry", "Tomato"
    ]
    st.markdown("- " + "\n- ".join(species))

    st.subheader("🦠 Maladies détectées (27 classes)")
    diseases = [
        "Apple rust leaf", "Apple Scab Leaf",
        "Bell pepper leaf spot",
        "Corn leaf blight", "Corn rust leaf", "Corn Gray leaf spot",
        "Potato leaf early blight", "Potato leaf late blight",
        "Tomato Early blight leaf", "Tomato leaf late blight",
        "Tomato mold leaf", "Tomato leaf yellow virus",
        "Tomato leaf mosaic virus", "Tomato leaf bacterial spot",
        "Tomato Septoria leaf spot", "Tomato two spotted spider mites leaf",
        "Squash Powdery mildew leaf",
        "Grape leaf black rot"
    ]
    st.markdown("- " + "\n- ".join(diseases))

    st.info("💡 Le dataset contient 2,598 images annotées pour l'entraînement")

# Interface principale
st.markdown("---")

# Upload d'image
uploaded_file = st.file_uploader(
    "📤 Choisissez une image de feuille de plante",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Afficher l'image originale
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Image originale")
        st.image(image, use_container_width=True)

    # Choix du type de prédiction
    st.markdown("---")
    st.subheader("🎯 Choisissez le type d'analyse")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("🔍 Saine ou Malade", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                # Préparer le fichier pour l'API
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

                try:
                    response = requests.post(f"{API_URL}/predict/binary", files=files)
                    response.raise_for_status()
                    result = response.json()

                    with col2:
                        st.subheader("📊 Résultats - Classification Binaire")

                        # Afficher l'image annotée
                        if "annotated_image" in result:
                            img_data = base64.b64decode(result["annotated_image"].split(",")[1])
                            img = Image.open(io.BytesIO(img_data))
                            st.image(img, use_container_width=True)

                        # Afficher les prédictions
                        if result["predictions"]:
                            for pred in result["predictions"]:
                                st.success(f"**{pred['class_name']}** - Confiance: {pred['confidence']:.2%}")
                        else:
                            st.warning("Aucune détection")

                except Exception as e:
                    st.error(f"Erreur: {str(e)}")

    with col_btn2:
        if st.button("🌱 Identifier l'espèce", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

                try:
                    response = requests.post(f"{API_URL}/predict/species", files=files)
                    response.raise_for_status()
                    result = response.json()

                    with col2:
                        st.subheader("📊 Résultats - Identification d'espèce")

                        if "annotated_image" in result:
                            img_data = base64.b64decode(result["annotated_image"].split(",")[1])
                            img = Image.open(io.BytesIO(img_data))
                            st.image(img, use_container_width=True)

                        if result["predictions"]:
                            for pred in result["predictions"]:
                                st.success(f"**{pred['class_name']}** - Confiance: {pred['confidence']:.2%}")
                        else:
                            st.warning("Aucune détection")

                except Exception as e:
                    st.error(f"Erreur: {str(e)}")

    with col_btn3:
        if st.button("🦠 Diagnostiquer la maladie", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

                try:
                    response = requests.post(f"{API_URL}/predict/diseases", files=files)
                    response.raise_for_status()
                    result = response.json()

                    with col2:
                        st.subheader("📊 Résultats - Diagnostic de maladie")

                        if "annotated_image" in result:
                            img_data = base64.b64decode(result["annotated_image"].split(",")[1])
                            img = Image.open(io.BytesIO(img_data))
                            st.image(img, use_container_width=True)

                        if result["predictions"]:
                            for pred in result["predictions"]:
                                st.success(f"**{pred['class_name']}** - Confiance: {pred['confidence']:.2%}")
                        else:
                            st.warning("Aucune détection")

                except Exception as e:
                    st.error(f"Erreur: {str(e)}")

else:
    st.info("👆 Veuillez télécharger une image pour commencer l'analyse")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔬 Basé sur le dataset PlantDoc avec 2,598 images annotées</p>
    <p>📄 <a href='https://arxiv.org/abs/1911.10317'>Paper: PlantDoc: A Dataset for Visual Plant Disease Detection</a></p>
</div>
""", unsafe_allow_html=True)
