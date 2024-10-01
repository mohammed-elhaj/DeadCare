import streamlit as st
from PIL import Image
import easyocr
import numpy as np

# Set up EasyOCR reader (Arabic language included)
reader = easyocr.Reader(['ar'])

# Streamlit app
st.title("Specific Text Extraction with EasyOCR")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Function to get all text to the left of reference indicators, within the same line and x-pixel range
def find_texts_to_left(results, indicator_text_list, x_range=200):
    indicator_box = None
    for result in results:
        text = result[1]  # Extracted text
        bbox = result[0]  # Bounding box

        # Find the indicator bounding box if any of the possible indicators are found
        if any(indicator_text in text for indicator_text in indicator_text_list):
            indicator_box = bbox
            break

    if indicator_box:
        # Get the Y range of the indicator (same line check)
        indicator_y_top = min(indicator_box[0][1], indicator_box[1][1])  # Top y-coordinate
        indicator_y_bottom = max(indicator_box[2][1], indicator_box[3][1])  # Bottom y-coordinate

        # Store all valid texts within x-range and same line
        valid_texts = []
        
        for result in results:
            bbox = result[0]
            text = result[1]

            # Calculate the Y range of the current text (to check if it is on the same line)
            text_y_top = min(bbox[0][1], bbox[1][1])
            text_y_bottom = max(bbox[2][1], bbox[3][1])

            # Check if text is on the same horizontal line (Y range overlaps with indicator)
            same_line = (text_y_top >= indicator_y_top - 20) and (text_y_bottom <= indicator_y_bottom + 30)

            # Check if the text is within the x-pixel range to the left of the indicator
            if bbox[0][0] < indicator_box[0][0] and (indicator_box[0][0] - bbox[1][0] <= x_range) and same_line:
                valid_texts.append((bbox[0][0], text))  # Store x-coordinate and the text

        # Sort the valid texts by their x-coordinates (from left to right)
        sorted_texts = sorted(valid_texts, key=lambda x: x[0], reverse=True)

        # Combine all sorted texts into a single string
        final_text = ' '.join([text for _, text in sorted_texts])
        return final_text

    return None

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
            results = reader.readtext(image_np)

            # Define the x-range for searching (in pixels)
            x_range = 300

            # List of possible indicator text variations
            name_1_indicators = ["نعم انا"]
            nationality_1_indicators = ["الجنسية"]
            name_2_indicators = ["اقر بانني استلمت", "اقر باننى استلمت"]
            nationality_2_indicators = ["الجنسية"]

            # Find the specific texts using the new function with indicator lists
            name_1 = find_texts_to_left(results, name_1_indicators, x_range)
            nationality_1 = find_texts_to_left(results, nationality_1_indicators, x_range)
            name_2 = find_texts_to_left(results, name_2_indicators, x_range)
            nationality_2 = find_texts_to_left(results, nationality_2_indicators, x_range)

            # Display the extracted specific sentences
            st.subheader("Extracted Information:")
            st.text(f"1. Name (left of {name_1_indicators}): {name_1}")
            st.text(f"2. Nationality (left of {nationality_1_indicators}): {nationality_1}")
            st.text(f"3. Name (left of {name_2_indicators}): {name_2}")
            st.text(f"4. Nationality (left of {nationality_2_indicators}): {nationality_2}")

            st.subheader("All Information:")
            for result in results:
                st.text(result[1])

