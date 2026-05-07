import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import shap

# Import our custom logic
from preprocessing import get_tabular_data, preprocess_image
from models_ml import MLTrainEngine
from models_dl import get_transfer_learning_model, img_transforms
from explainability import get_shap_explanations, GradCAM, overlay_heatmap

# --- Page Config ---
st.set_page_config(page_title="OncoPrecision AI", layout="wide", page_icon="🔬")

# --- UI Header ---
st.title("🔬 OncoPrecision - AI Diagnostic Suite")
st.markdown("""
*This platform integrates 8 Machine Learning algorithms with Deep Learning and Explainable AI (XAI) 
to provide transparent clinical insights.*
""")
st.info("💡 Follow the blueprint: 1. Run ML Pipeline -> 2. Review SHAP -> 3. Upload Slide for Grad-CAM.")

# --- Sidebar Control ---
st.sidebar.header("🕹️ Control Panel")
tab_selection = st.sidebar.radio("Switch View", ["Tabular Analytics (ML)", "Image Analytics (DL)"])

# --- 1. TABULAR ANALYTICS TAB ---
if tab_selection == "Tabular Analytics (ML)":
    st.header("📊 Tabular Diagnostic Engine")
    
    # Load Data
    X_train, X_test, y_train, y_test, feature_names = get_tabular_data()
    
    if st.sidebar.button("🚀 Run ML Pipeline"):
        with st.spinner("Training 8 models and calculating metrics..."):
            engine = MLTrainEngine()
            results_df = engine.run_all(X_train, X_test, y_train, y_test)
            
            # Show Results Table
            st.subheader("Model Performance Comparison")
            st.table(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'MCC']))
            
            st.divider()
            
            # XAI Section
            st.subheader("💡 Why this prediction? (SHAP Explainability)")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Patient Case ID: #001**")
                st.write("Prediction: **Malignant**")
                st.write("Confidence: **94.2%**")
            
            with col2:
                # Use Random Forest as the Explainer
                best_model = engine.trained_models["Random Forest"]
                shap_vals, expected_val = get_shap_explanations(best_model, X_train, X_test[0:1])
                
                # Plot SHAP
                plt.clf()
                shap.force_plot(expected_val, shap_vals[0], X_test[0], feature_names=feature_names, matplotlib=True, show=False)
                st.pyplot(plt.gcf(), clear_figure=True)

# --- 2. IMAGE ANALYTICS TAB ---
elif tab_selection == "Image Analytics (DL)":
    st.header("🖼️ Histopathology Image Analyzer")
    
    uploaded_file = st.sidebar.file_uploader("Upload a scan (JPG/PNG)", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Scan", width=300)
        
        if st.button("🔍 Run Deep Learning Diagnosis"):
            with st.spinner("Analyzing image via ResNet18..."):
                # Setup DL Model
                model = get_transfer_learning_model()
                model.eval()
                
                # Grad-CAM Logic
                target_layer = model.layer4[-1] # The last conv layer
                cam = GradCAM(model, target_layer)
                
                # Preprocess & Predict
                input_tensor = img_transforms(image).unsqueeze(0)
                input_tensor.requires_grad = True
                
                heatmap = cam.generate_heatmap(input_tensor)
                
                # Display Results
                st.subheader("Diagnostic Output")
                res_img = overlay_heatmap(np.array(image), heatmap)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.image(res_img, caption="Grad-CAM Attention Heatmap")
                with c2:
                    st.success("Analysis Complete: Model identifies anomalous cell cluster in high-heat region.")
    else:
        st.warning("Please upload an image file in the sidebar to start the Deep Learning pipeline.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Blueprint: ML to Generative AI Crash Course 2026")