#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[ ]:





# In[ ]:


import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the Keras model from a Google Colab directory
model = tf.keras.models.load_model("MODEL.h5")

# Define the Streamlit web app
def main():
    # Set the title and description
    st.title("Brain Tumor Classification")
    st.write("Upload an MRI image for classification.")

    # Input field for image upload
    uploaded_image = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

    # Check if an image is uploaded
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption='Uploaded MRI Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Perform prediction when the 'Predict' button is clicked
        if st.button("Predict"):
            prediction_result = predict_image(uploaded_image)
            st.markdown(f"<h2>Prediction:</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 24px;'>{prediction_result}</p>", unsafe_allow_html=True)

# Function to make a prediction
def predict_image(uploaded_image):
    # Load and preprocess the image
    image = Image.open(uploaded_image)
    image = image.resize((280, 280))
    image = np.array(image) / 255.0  # Normalize pixel values to the range [0, 1]
    image = tf.expand_dims(image, axis=0)

    # Make a prediction using the loaded model
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    class_names = ['Glioma', 'Meningioma','No','Pituitary']  # Replace with your actual class names
    return f"Predicted Class: {class_names[predicted_class]}"

if __name__ == "__main__":
    main()

