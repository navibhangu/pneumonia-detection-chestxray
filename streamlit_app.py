import streamlit as st
import numpy as np
from PIL import Image
import boto3
import json
import io
import os 
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

# Initialize the SageMaker runtime client
client = boto3.client(
    'runtime.sagemaker',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
#Streamlit app title and description
st.title("Predictive Risk Assessment for Pneumonia using Chest X-Ray")
st.write("Upload your chest X-ray to the machine learning model.")

#ile uploader for chest X-ray images
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type="jpg")

#If an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image to match the input size expected by the model
    image = image.resize((224, 224))

    # Convert the image to bytes in JPEG format
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    # Invoke the SageMaker endpoint
    response = client.invoke_endpoint(
        EndpointName="canvas-new-deployment-08-18-2024-5-56-PM",  # Your SageMaker endpoint name without colon
        ContentType="image/jpeg",  # The content type
        Body=img_bytes,            # The image data as bytes
        Accept="application/json"  # The expected response format
    )

    # Read the response body once
    response_body = response['Body'].read().decode('utf-8')

    # Print the raw response for debugging
    st.write("Raw response from the endpoint:")
    st.write(response_body)
try:
        # Parse the JSON response
        prediction = json.loads(response_body)

        # Display the prediction result
        predicted_label = prediction.get('predicted_label')
        probability = prediction.get('probability')

        st.write(f"Prediction: {predicted_label}")
        st.write(f"Confidence: {probability:.2f}")
except json.JSONDecodeError:
        st.write("Error decoding the JSON response.")



