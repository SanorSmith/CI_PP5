# Cherry Leaf Powdery Mildew Detector

![Banner](https://github.com/SanorSmith/CI_PP5_Cherry_Leaf_Disease_Detection/blob/main/assets/banner.jpg)
## Introduction

The **Cherry Leaf Disease Detector** is a machine learning solution developed to assist the agricultural industry, specifically Farmy & Foods, in early detection of powdery mildew on cherry leaves. The project leverages Convolutional Neural Networks (CNNs) for accurate image classification, helping farmers identify infections early to protect crop yield and quality. This project is part of the Code Institute's Full Stack Software Development Bootcamp.

The application is integrated into an intuitive Streamlit dashboard where users can upload images of cherry leaves to receive real-time predictions. The solution is scalable, making it adaptable for detecting other plant diseases in the future.

![Cherry Leaf Powdery Mildew Detector – Responsive Design Preview](https://github.com/SanorSmith/CI_PP5_Cherry_Leaf_Disease_Detection/blob/main/assets/amiresponsive.jpg)  
*Illustration: The Cherry Leaf Powdery Mildew Detector app showcasing its responsive design across different devices.*

### Deployed Version
[Cherry Leaf Disease Detector](https://cherry-leaf-disease-detection-63b42bf0104d.herokuapp.com/)

---

## Table of Contents

1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and Validation](#hypothesis-and-validation)
4. [Rationale for the Model](#rationale-for-the-model)
5. [Implementation of the Business Requirements](#implementation-of-the-business-requirements)
6. [Machine Learning Business Case](#machine-learning-business-case)
7. [Dashboard Design (Streamlit UI)](#dashboard-design-streamlit-ui)
8. [CRISP-DM Process](#crisp-dm-process)
9. [Bugs](#bugs)
10. [Deployment](#deployment)
11. [Technologies Used](#technologies-used)
12. [Credits](#credits)

---

## Dataset Content

The dataset used in this project comprises over 4,000 high-resolution images of cherry leaves, meticulously categorized into two distinct classes: **healthy leaves** and **leaves infected with powdery mildew**. This dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves), serves as the backbone for building a machine learning model capable of detecting fungal infections by analyzing visual symptoms. Infected leaves display characteristic signs such as white powdery growth and circular lesions, making them visually distinguishable from healthy ones. The dataset's comprehensive collection ensures that the model is trained with a wide range of examples for enhanced learning.

To achieve optimal model performance, the dataset was divided into three subsets: 70% for training, 10% for validation, and 20% for testing. Training data is used to teach the model patterns associated with the two classes, while validation data helps fine-tune parameters and prevent overfitting. The test set, comprising unseen data, is used to evaluate the model's generalization capabilities. Before feeding images into the model, several preprocessing steps were applied, including resizing all images to a uniform dimension to ensure compatibility with the neural network, normalizing pixel values for better pattern recognition, and applying data augmentation techniques such as flipping, rotation, and zooming to artificially increase dataset diversity. These steps collectively enhance the robustness and accuracy of the model, enabling reliable predictions for real-world agricultural applications.


