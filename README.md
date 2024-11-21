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
5. [The Rationale to Map the Business Requirements to Data Visualizations and ML Tasks](#the-rationale-to-map-the-business-requirements-to-data-visualizations-and-ml-tasks)
6. [Strategic Vision for Machine Learning Integration](#strategic-vision-for-machine-learning-integration)
7. [Project Dashboard](#project-dashboard)
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

## The Rationale to Map the Business Requirements to Data Visualizations and ML Tasks

The project aligns its business requirements with tailored data visualizations and machine learning (ML) tasks to deliver an intuitive and effective solution. A key focus is on creating an interactive dashboard that allows clients to easily navigate and explore various insights. The dashboard includes side-by-side visual graphs showcasing average images, differences, and variabilities between healthy and infected leaves. These visualizations enable clients to identify distinguishing features effortlessly. Additionally, the dashboard provides an image montage for a direct comparison of healthy and infected leaves, alongside a confusion matrix to assess the model's performance in classification tasks.

For classification, the project emphasizes dynamic plots to track the model's accuracy and loss over epochs, ensuring progress towards achieving the target accuracy of 97% or higher. The impact of preprocessing techniques like image normalization and data augmentation is also visualized, helping clients understand how these methods improve the model's reliability. Furthermore, performance metrics on a validation dataset confirm the model's ability to generalize effectively to unseen data.

The system also prioritizes generating actionable insights through user-friendly reporting. Clients can upload cherry leaf images to receive immediate health status feedback, aiding quick decision-making. Dataset distribution visualizations, including proportions of healthy versus infected leaves, provide deeper insights into the data used for training. Clear instructions and contextual explanations alongside these visualizations ensure the dashboard remains accessible and interpretable, even for users without technical expertise.

[Back to top ⇧](#table-of-contents)

## Strategic Vision for Machine Learning Integration

The primary objective of this project is to develop a machine learning model capable of predicting whether a cherry leaf is infected with powdery mildew, using a dataset of labeled leaf images. This supervised learning task focuses on binary classification, where the model determines if a leaf is "healthy" or "infected." The desired outcome is to provide a faster, more reliable method for early detection of powdery mildew, enabling timely intervention and minimizing crop damage.

The success of the project will be measured by achieving a minimum accuracy of 97% on the test set. Additionally, the solution is designed to deliver real-time results, ensuring users receive diagnostic outcomes immediately after uploading images, eliminating the need for batch processing. The model’s output will include a binary flag indicating whether a leaf is healthy or infected, accompanied by probability scores for each prediction. This output will be accessible through a user-friendly dashboard, allowing seamless interaction for users with minimal technical expertise.

Traditionally, detecting powdery mildew involved manual inspection by employees, which could take up to 30 minutes per tree and was prone to human error. The proposed image-based detection system enhances speed, accuracy, and consistency, addressing these inefficiencies. The training data for the model comprises a labeled dataset of 4,208 images, sourced from Kaggle, containing both healthy and infected cherry leaves. This dataset plays a pivotal role in training and validating the model.

The business benefits of this solution are significant. It streamlines the inspection process, saving Farmy & Foods considerable time compared to manual methods. Furthermore, the model reduces human error, ensuring more consistent and reliable diagnoses. The solution is scalable, allowing for application to larger datasets or adaptation to detect diseases in other crops, broadening its impact within the agricultural sector.

[Back to top ⇧](#table-of-contents)

## Project Dashboard

The **Cherry Leaf Disease Detector Dashboard** is a powerful tool designed to streamline the process of detecting powdery mildew infections on cherry leaves. This dashboard leverages machine learning to provide actionable insights, enabling early intervention and minimizing crop yield losses. With an emphasis on usability, the system integrates features to ensure accurate predictions, robust model performance, and a user-friendly interface for diverse audiences.

## Visual Insights into Disease Symptoms

A key feature of the dashboard is its ability to visualize the differences between healthy and infected leaves. Using tools such as average and variability images, as well as montages, the system highlights distinguishing patterns like white fungal growth and discoloration. These visualizations not only enhance understanding but also validate the model's training process by showcasing the critical features used for classification.

## Model Performance Analysis

The dashboard includes a section dedicated to evaluating the machine learning model's performance. Metrics such as accuracy and loss over training epochs are presented, offering transparency and assurance of the model’s reliability. This analysis ensures the system meets the high-performance standards necessary for practical deployment, targeting an accuracy rate of 97% to minimize errors in prediction.

## Real-Time Disease Detection

The core functionality of the dashboard allows users to upload images of cherry leaves and receive instant predictions about their health. This feature supports batch uploads and generates downloadable reports, making it easier for users to monitor large-scale operations efficiently. The intuitive design ensures accessibility for users with varying levels of technical expertise, promoting widespread adoption.

## Validating Hypotheses for Model Development

The project incorporates several hypotheses to refine the model’s robustness and accuracy. By analyzing visual symptoms, utilizing image normalization, and training with augmented data, the system ensures reliable generalization to unseen scenarios. Each hypothesis is supported by data and visual outputs, reinforcing the system’s ability to accurately identify powdery mildew in cherry leaves and address real-world agricultural challenges.

[Back to top ⇧](#table-of-contents)


