# 🩺 Predictive Risk Assessment for Pneumonia — Chest X-Ray Classifier

This Streamlit app lets you upload a chest X-ray image and get a **pneumonia risk prediction** from an **AWS SageMaker** machine learning model.  
It returns the predicted label (e.g., "Pneumonia" / "Normal") and confidence score.

> ⚠ **Disclaimer:** This app is for **educational and demonstration purposes only**. It is **not** a medical diagnostic tool. Always seek professional medical advice.

---

## Features
- Upload `.jpg` chest X-ray images
- Preprocess image to model’s expected format (`224×224`)
- Send the image to a **SageMaker real-time endpoint**
- Display:
  - Raw JSON response from the model
  - Predicted label
  - Model confidence score

---

## Requirements
- Python 3.9–3.11
- AWS account with:
  - Deployed SageMaker endpoint
  - IAM permission: `sagemaker:InvokeEndpoint`
- Endpoint must accept `image/jpeg` and return JSON like:
```json
{
  "predicted_label": "Pneumonia",
  "probability": 0.93
}
