# page_cells_visualizer.py

import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import itertools
import random

def page_cells_visualizer_body():
    st.write("### Cells Visualizer")
    st.info(
        "* This visual study allows comparison between parasitized and uninfected cells, "
        "helping to identify visual markers for each type.")

    version = 'v1'
    if st.checkbox("Difference between Average and Variability Images"):
        avg_parasitized = imread(f"outputs/{version}/avg_var_powdery_mildew.png")
        avg_uninfected = imread(f"outputs/{version}/avg_var_healthy.png")

        st.warning(
            "* Average and variability images may not intuitively differentiate parasitized from uninfected cells, "
            "but slight color differences might be observed.")

        st.image(avg_parasitized, caption='Average and Variability - Parasitized Cell')
        st.image(avg_uninfected, caption='Average and Variability - Uninfected Cell')
        st.write("---")

    if st.checkbox("Differences Between Average Parasitized and Average Uninfected Cells"):
        diff_between_avgs = imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            "* This comparison may not intuitively reveal clear differences between the two types of cells.")
        st.image(diff_between_avgs, caption='Difference Between Average Images - Parasitized vs Uninfected')

    if st.checkbox("Image Montage"): 
        st.write("* Click 'Create Montage' to refresh the displayed montage.")
        my_data_dir = 'inputs/cherry-leaves'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(label="Select Cell Type", options=labels, index=0)
        
        if st.button("Create Montage"):      
            image_montage(
                dir_path=my_data_dir + '/validation',
                label_to_display=label_to_display,
                nrows=8, ncols=3, figsize=(10, 25)
            )
        st.write("---")

def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    sns.set_style("white")
    labels = os.listdir(dir_path)

    # Display selected label's images if it exists
    if label_to_display in labels:
        images_list = os.listdir(f"{dir_path}/{label_to_display}")
        
        # Verify montage grid space availability
        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            st.warning(
                f"Decrease 'nrows' or 'ncols' to fit available images. "
                f"Only {len(images_list)} images in this subset; requested montage with {nrows * ncols} spaces."
            )
            return
        
        # Prepare the montage grid
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        plot_idx = list(itertools.product(range(nrows), range(ncols)))

        # Populate the grid with selected images
        for i in range(nrows * ncols):
            img_path = f"{dir_path}/{label_to_display}/{img_idx[i]}"
            img = imread(img_path)
            axes[plot_idx[i][0], plot_idx[i][1]].imshow(img)
            axes[plot_idx[i][0], plot_idx[i][1]].set_title(
                f"{img.shape[1]}x{img.shape[0]} px"
            )
            axes[plot_idx[i][0], plot_idx[i][1]].set_xticks([])
            axes[plot_idx[i][0], plot_idx[i][1]].set_yticks([])
        
        plt.tight_layout()
        st.pyplot(fig=fig)
    else:
        st.error(f"The selected label '{label_to_display}' does not exist. Available labels: {labels}")

