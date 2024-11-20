# Cherry Leaf Disease Detector

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

## Business Requirements

A prominent agricultural company has requested the development of a Machine Learning-based solution to efficiently identify powdery mildew on cherry tree leaves. With thousands of cherry trees distributed across multiple farms, the manual inspection process currently in use is highly time-consuming and lacks scalability for their expansive operations.The current manual inspection process is time-intensive and unsustainable given the scale of their operations, which span thousands of cherry trees across multiple farms nationwide.


1. **Identify Visual Distinctions:**  
   Perform a detailed analysis to visually distinguish between healthy leaves and those affected by powdery mildew, helping Farmy & Foods gain insights into visible symptoms.

2. **Precision Detection System:**  
   Create a reliable and accurate predictive model to classify cherry tree leaves as either healthy or infected. The system must achieve a **minimum accuracy of 97%**, ensuring minimal errors in predictions to optimize resource use and protect crops effectively.

3. **Intuitive and Accessible Dashboard:**  
   Design a user-friendly, web-based interface that simplifies the process of uploading images and obtaining results. The dashboard must cater to non-technical users while ensuring compatibility with various devices, including smartphones and desktops.

4. **Generate Actionable Reports:**  
   Provide a comprehensive analysis of the predictions in the form of reports. These reports will assist Farmy & Foods in making informed decisions about crop treatments and disease management.

5. **Future Scalability:**  
   Ensure the solution is adaptable, allowing it to extend to other plant diseases or crop types in the future without significant redevelopment effort.

### Key Deliverables:
- **Accurate Disease Classification** for cherry leaves.
- **High Model Performance** to meet business standards with reliable predictions.
- **Ease of Use** through an intuitive dashboard for disease detection.
- **Detailed Prediction Reports** to support operational decisions.
- **Scalable Architecture** to accommodate additional crops and diseases in the future.

This solution aims to empower Farmy & Foods with cutting-edge technology, enabling them to optimize their processes, improve productivity, and maintain healthy crop yields with minimal manual intervention.

[Back to top ⇧](#table-of-contents)


## Hypothesis and Validation

### 1. Visual Characteristics of Infected Cherry Leaves

**Hypothesis**: Infected cherry leaves exhibit distinctive symptoms that can be identified through image analysis.

Powdery mildew infection in cherry leaves causes visually identifiable symptoms, such as light-green circular lesions that develop into a cotton-like white fungal growth. These symptoms are distinguishable by the human eye and are equally well-suited for machine learning-based image analysis. By leveraging features like texture, color distribution, and lesion patterns, the model can effectively differentiate healthy leaves from infected ones.

To ensure effective learning, images are normalized to standardize pixel values, enhancing consistency across datasets. This process not only improves feature extraction but also helps the model generalize well, even with variations in lighting, angles, or leaf orientation. 

**Visual Aids**:
- Image montages highlight evident differences between healthy and infected leaves:
  ![montage_healthy](./assets/healthy_leaves.png)
  ![montage_infected](./assets/infected_leaves.png)

- Average and variability images showcase distinguishing features and inconsistencies between healthy and infected leaves:
  ![average_variability](./assets/infected_average_variability.png)

- Comparing average images of healthy and infected leaves underscores critical visual differences:
  ![average_healthy_average_infected](./assets/healthy_average_variability.png)

These visualizations validate the hypothesis that infected leaves possess unique patterns, which the model can learn to generalize and accurately predict.

---

### 2. Impact of Image Normalization on Model Performance

**Hypothesis**: Normalizing images enhances the model's ability to differentiate between healthy and infected leaves.

Normalization scales pixel values within a consistent range, reducing noise and improving training stability. This ensures the model focuses on relevant features rather than inconsistencies in the data. The normalization process contributes to improved generalization, enabling accurate predictions on unseen images.

**Model Performance**:
- Training and validation accuracies stabilize quickly, indicating effective generalization:
  ![normalization_image](./outputs/v1/model_history.png)

- Consistently low validation loss reflects the model's ability to avoid overfitting, further validating the effectiveness of normalization techniques.

---

### 3. Advantages of Data Augmentation

**Hypothesis**: Augmented training data enhances model robustness and adaptability.

Data augmentation techniques, such as rotation, flipping, and zooming, simulate diverse real-world conditions, enabling the model to generalize effectively. This approach ensures the model learns features that remain invariant to changes in orientation, lighting, or scale, improving performance on unseen data. 

**Impact of Augmentation**:
- Augmented data prevents overfitting and enhances resilience to variations in test scenarios.
- Validation accuracy remains stable, even when exposed to diverse input conditions, showcasing the robustness of the augmented model.

---

## Rationale for the Model

The model addresses a critical challenge in agriculture: detecting powdery mildew in cherry leaves, a fungal disease that severely affects crop yield. By leveraging Convolutional Neural Networks (CNNs), the model extracts subtle visual features from leaf images to identify signs of infection. This automated approach reduces reliance on time-intensive manual inspections, offering an efficient and scalable solution for large-scale agricultural operations.

**Key Objectives**:
- Accurately classify leaves as healthy or infected.
- Enable early detection and intervention to mitigate crop damage.
- Provide a scalable, user-friendly tool for real-world agricultural scenarios.

---


[Back to top ⇧](#table-of-contents)
