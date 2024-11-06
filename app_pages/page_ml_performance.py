# page_ml_performance.py

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation

def page_ml_performance_metrics():
    version = 'v1'
    st.info(
        "This page presents an overview of the model's performance on the dataset, "
        "explaining the dataset split, evaluation metrics, and model accuracy."
    )
    
    st.write("### Image Distribution Across Sets and Labels")

    labels_distribution = imread(f"outputs/{version}/image_distribution_bar.png")
    st.image(labels_distribution, caption='Label Distribution: Train, Validation, and Test Sets')

    sets_distribution = imread(f"outputs/{version}/set_distribution_pie.png")
    st.image(sets_distribution, caption='Dataset Set Distribution')

    st.warning(
        "The cherry leaves dataset is divided into three parts:\n\n"
        "- **Train Set (70%)**: Used to fit the model, where it learns to generalize and make predictions.\n"
        "- **Validation Set (10%)**: Aids in fine-tuning the model after each training epoch to improve performance.\n"
        "- **Test Set (20%)**: Final data for evaluating the model’s accuracy, containing data unseen during training."
    )
    st.write("---")

    st.write("### Model Performance")

    classification_report_img = imread(f"outputs/{version}/classification_report.png")
    st.image(classification_report_img, caption='Classification Report')  

    st.warning(
        "**Classification Report Details**\n\n"
        "- **Precision**: The proportion of correctly predicted positives out of total predicted positives.\n"
        "- **Recall**: The proportion of actual positives identified by the model.\n"
        "- **F1 Score**: The weighted harmonic mean of precision and recall, balancing the two.\n"
        "- **Support**: The actual number of samples per class."
    )

    roc_curve_img = imread(f"outputs/{version}/roccurve.png")
    st.image(roc_curve_img, caption='ROC Curve')

    st.warning(
        "**ROC Curve Explanation**\n\n"
        "The ROC curve shows the model's ability to distinguish between classes by plotting true positive rate (sensitivity) against false positive rate (1 - specificity).\n"
        "- **True Positive Rate (Sensitivity)**: Proportion of actual positives correctly predicted.\n"
        "- **False Positive Rate**: Proportion of negatives incorrectly predicted as positive."
    )

    confusion_matrix_img = imread(f"outputs/{version}/confusion_matrix.png")
    st.image(confusion_matrix_img, caption='Confusion Matrix')

    st.warning(
        "**Confusion Matrix Interpretation**\n\n"
        "- **True Positive/True Negative**: Correct predictions where the prediction matches reality.\n"
        "- **False Positive/False Negative**: Incorrect predictions where the prediction is the opposite of reality.\n"
        "A strong model maximizes True Positives and True Negatives while minimizing False Positives and False Negatives."
    )

    model_performance_img = imread(f"outputs/{version}/model_history.png")
    st.image(model_performance_img, caption='Model Training Performance')

    st.warning(
        "**Model Performance Overview**\n\n"
        "- **Loss**: Represents the sum of errors made during training (loss) or validation (val_loss) over each iteration.\n"
        "- **Accuracy**: The ratio of correct predictions over total predictions for training (accuracy) or validation (val_accuracy) sets.\n"
        "Good generalization indicates the model is effective with new data, not overfitting to the training data."
    )

    st.write("### Test Set Performance Metrics")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    
    st.write(
        "For further information, please refer to the "
        "[Project README file](https://github.com/SanorSmith/CI_PP5_Cherry_Leaf_Disease_Detection)."
    )
