import streamlit as st
from PIL import Image
import easyocr

# Set up EasyOCR reader (you can specify the language here, e.g., ['en'] for English)
reader = easyocr.Reader(['ar'])

# Streamlit app
st.title("Image Text Extraction with EasyOCR")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Process the image if uploaded
if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Extract text button
    if st.button("Extract Text"):
        with st.spinner("Extracting text..."):
            # Convert image to byte array for OCR processing
            result = reader.readtext(uploaded_image)

            # Display the extracted text
            st.subheader("Extracted Text:")
            extracted_text = "\n".join([text[1] for text in result])
            st.text(extracted_text)
