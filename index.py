import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.models import load_model

# Load the trained CNN model
# @st.cache(allow_output_mutation=True)
@st.cache_resource
def loads_model():
    cnn = load_model("ml_model")
    return cnn

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image to the input size of the model
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Mapping class indices to class labels
class_labels = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}

# Streamlit app
def main():
    st.title('Brain Tumor Image Classifier :')
    st.write('Upload an MRI image of the Tumor and the model will predict the type of the Tumor')
    st.write('Tumor Types : 1. Glioma, 2. Meningioma, 3. Pituitary')


    # Upload image
    uploaded_image = st.file_uploader("Upload Your MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess image and make prediction
        model = loads_model()

        processed_image = preprocess_image(image)
        # st.write('Image shape : ', processed_image.shape)


        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]

        st.write('Prediction : ', predicted_class)

if __name__ == '__main__':
    main()
