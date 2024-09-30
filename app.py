import streamlit as st
from PIL import Image
import easyocr
import numpy as np

# Set up EasyOCR reader (you can specify the language here, e.g., ['en'] for English)
reader = easyocr.Reader(['ar'], gpu=False)

# Streamlit app
st.title("Image Text Extraction with EasyOCR")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Process the image if uploaded
if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the PIL image to numpy array
    image_np = np.array(image)

    # Extract text button
    if st.button("Extract Text"):
        with st.spinner("Extracting text..."):
            # Pass the numpy array to EasyOCR
            result = reader.readtext(image_np)

            # Display the extracted text
            st.subheader("Extracted Text:")
            extracted_text = "\n".join([text[1] for text in result])
            st.text(extracted_text)
