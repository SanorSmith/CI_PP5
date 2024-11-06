# page_cherry_leaf_disease_detector.py

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)

def page_cherry_leaf_disease_detector_body():
    st.info(
        "Upload images of cherry leaves to analyze whether they are affected by powdery mildew and "
        "download a comprehensive report of the results."
    )

    st.write(
        "If you would like a sample dataset of both infected and healthy leaves for live predictions, "
        "you can download it from [this link](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write("---")
    
    st.write("**Upload one or more clear images of cherry leaves for analysis.**")
    images_buffer = st.file_uploader('', type='jpeg', accept_multiple_files=True)
   
    if images_buffer:
        df_report = pd.DataFrame([])
        
        for image in images_buffer:
            img_pil = Image.open(image)
            st.info(f"Analyzing Image: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Dimensions: {img_array.shape[1]}px x {img_array.shape[0]}px")

            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report.append({"Image Name": image.name, "Diagnosis": pred_class}, ignore_index=True)
        
        if not df_report.empty:
            st.success("Analysis Summary Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)

    st.write(
        "For more details about this project, please refer to the "
        "[Project README file](https://github.com/SanorSmith/CI_PP5_Cherry_Leaf_Disease_Detection)."
    )
