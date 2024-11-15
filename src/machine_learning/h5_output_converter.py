from tensorflow.keras.models import load_model

# Load the .h5 model
model = load_model('outputs/v1/powdery_mildew_model.h5')

# Save it in the SavedModel format
model.save('outputs/v1/powdery_mildew_model')
print("Model successfully saved in the SavedModel format")
