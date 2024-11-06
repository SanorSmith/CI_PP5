# page_project_hypothesis.py

import streamlit as st
import matplotlib.pyplot as plt

def page_project_hypothesis_body():
    st.write("### Hypothesis 1 and Validation")

    st.success(
        "Infected cherry leaves have distinguishable marks that set them apart from healthy leaves."
    )
    st.info(
        "It is hypothesized that cherry leaves affected by powdery mildew exhibit clear symptoms, "
        "beginning with light-green, circular lesions on the leaf surface, followed by a subtle white "
        "cotton-like growth in the infected area."
    )
    st.write(
        "For a detailed visualization and investigation of infected vs. healthy leaves, please visit the Leaves Visualizer tab."
    )

    st.warning(
        "The model successfully identifies these differences and learns to generalize, enabling accurate predictions. "
        "A well-trained model achieves this by not overfitting to training data but rather learning generalized patterns "
        "between features and labels. This allows the model to reliably predict future cases without simply 'memorizing' "
        "the training dataset."
    )

    st.write("### Hypothesis 2 and Validation")

    st.success(
        "The `softmax` activation function performs better than `sigmoid` for the CNN's output layer."
    )
    st.info(
        "Both `softmax` and `sigmoid` are commonly used in binary and multiclass classification tasks. "
        "The efficacy of an activation function can be evaluated by plotting the model's prediction performance. "
        "Learning curves, which display accuracy and error rates on training and validation datasets during training, "
        "can reveal how well the model is learning.\n\n"
        "In our model, using `softmax` led to a smaller gap between training and validation accuracy, "
        "indicating more consistent learning beyond the 5th epoch compared to using `sigmoid`."
    )
    st.warning(
        "In this project, the `softmax` function demonstrated superior performance."
    )
    model_perf_softmax = plt.imread("outputs/v1/model_history.png")
    st.image(model_perf_softmax, caption='Softmax CNN Loss/Accuracy Performance') 
    model_perf_sigmoid = plt.imread("outputs/v1/model_loss_acc.png")
    st.image(model_perf_sigmoid, caption='Sigmoid CNN Loss/Accuracy Performance')

   
    st.write(
        "For additional information, please visit the "
        "[Project README file](https://github.com/SanorSmith/CI_PP5_Cherry_Leaf_Disease_Detection)."
    )
