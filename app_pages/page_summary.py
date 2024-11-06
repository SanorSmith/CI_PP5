# page_summary.py

import streamlit as st
import matplotlib.pyplot as plt

def page_summary_body():
    st.write("### Project Summary")

    st.info(
        f"**Project Background**\n"
        f"* This project focuses on distinguishing between parasitized and uninfected cells "
        f"from cell images.\n"
        f"* Visual inspection of cell images is commonly used to detect certain types of infections, "
        f"which can provide insights into health conditions based on cell morphology and other features.\n\n"
        
        f"**Dataset Overview**\n"
        f"* The dataset contains thousands of cell images categorized into 'parasitized' and 'uninfected'.\n"
        f"* These images will be analyzed to uncover patterns and develop models capable of "
        f"automatically identifying infected cells.\n\n"
        
        f"**Data Source and Relevance**\n"
        f"* The dataset was collected from medical sources focused on identifying parasitic infections "
        f"through visual means.\n"
        f"* By automating cell classification, this project aims to support quicker diagnostics and "
        f"potentially reduce the reliance on manual inspections in certain contexts.\n"
    )

    st.write(
        f"* For more detailed information, please refer to the "
        f"[Project README](https://github.com/your-repo/Project-README) "
        f"for background, methodology, and results."
    )
    
    st.success(
        f"The project addresses two main business objectives:\n"
        f"* 1 - Conduct a visual study to differentiate between infected and uninfected cells.\n"
        f"* 2 - Develop a predictive model capable of identifying infected cells from new images with "
        f"a high degree of accuracy.\n"
    )
