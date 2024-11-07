import streamlit as st
import torch
import pickle

# Load the results from the file
with open('results.pkl', 'rb') as f:
    data = pickle.load(f)
    results = data['results']
    val_results = data['val_results']

# Title of the web app
st.title('YOLO Model Training and Evaluation')

# Display CUDA availability and version
st.write(f"CUDA Available: {torch.cuda.is_available()}")
st.write(f"CUDA Version: {torch.version.cuda}")

# Display training results
st.write('## Training Results')
st.write(results)

# Display evaluation results
st.write('## Evaluation Results')
st.write(val_results)
# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Perform detection (assuming you have a detect function)
    # Example: detected_image = detect(uploaded_file)
    # For demonstration, let's assume detect function returns the same image
    detected_image = uploaded_file  # Replace this with actual detection logic
    
    # Display the detected image
    st.image(detected_image, caption='Detected Image.', use_column_width=True)
    # Assuming you have a YOLO model loaded and a detect function defined
    # Example:
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # def detect(image):
    #     results = model(image)
    #     return results.render()[0]

    # Load YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Perform detection
    results = model(uploaded_file)
    detected_image = results.render()[0]

    # Display the detected image
    st.image(detected_image, caption='Detected Image.', use_column_width=True)