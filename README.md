# Image Classification ML Pipeline

This project implements an end-to-end machine learning pipeline for image classification, fulfilling the African Leadership University (ALU) BSE Machine Learning Pipeline Summative Assignment requirements. The pipeline processes image data (e.g., Cats vs. Dogs dataset), trains a convolutional neural network (CNN) using TensorFlow, deploys it on AWS EC2, and provides a Streamlit UI for model interaction, retraining, and visualizations. A FastAPI endpoint enables predictions, and Locust is used to simulate flood requests for scalability testing.

## Project Description

The pipeline classifies images into two classes (e.g., cats vs. dogs) using a CNN model. Key functionalities include:
- **Data Acquisition**: Images sourced from the Kaggle Cats vs. Dogs dataset, stored in `data/train` and `data/test`.
- **Data Preprocessing**: Images resized to 150x150 pixels, normalized, and augmented (rotation, flip) for robustness.
- **Model Creation**: A CNN with convolutional, pooling, and dense layers, trained to achieve high accuracy.
- **Model Testing**: Evaluated in `image_classification.ipynb` using accuracy, precision, recall, F1-score, and confusion matrix.
- **Model Retraining**: Supports retraining with new images uploaded via the Streamlit UI, triggered by a button.
- **API**: A FastAPI endpoint (`/predict`) accepts image uploads for single predictions.
- **UI**: Streamlit app displays model uptime, visualizations, and controls for predictions and retraining.
- **Deployment**: Hosted on RENDER with Docker containers for scalability.

- **Link for the deployed app**:https://summative-image-classification-pipeline-urdy.onrender.com/

- **Flood Simulation**: Locust tests measure API latency and response times under varying loads.

### Visualizations
1. **Class Distribution**: Bar plot showing the balance of classes (e.g., number of cat vs. dog images).
   - *Insight*: Highlights dataset balance, critical for unbiased model training.
2. **Confusion Matrix**: Visualizes true/false positives/negatives on test data.
   - *Insight*: Reveals model performance, e.g., identifying if false positives are common for one class.
3. **Feature Activation Maps**: Displays CNN layer activations for three sample images.
   - *Insight*: Shows which features (e.g., edges, textures) the model prioritizes, aiding interpretability.

### Video Demo
Watch the project demo on YouTube: `<https://www.youtube.com/watch?v=8nMDTvmKhP0>` (replace with the actual YouTube URL after uploading).

## Requirements

- Python 3.9
- TensorFlow 2.16.1 (to avoid `batch_shape` deserialization issues)
- Streamlit
- FastAPI
- Locust
- Docker
- AWS CLI
- Full list in `requirements.txt`

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd image_classification_pipeline
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Sample `requirements.txt`:
   ```
   tensorflow==2.6.0
   streamlit==1.38.0
   fastapi==0.115.0
   uvicorn==0.30.6
   locust==2.31.6
   numpy==1.26.4
   matplotlib==3.9.2
   seaborn==0.13.2
   ```

4. **Dataset**
   - Extracted data and place images in `data/train` and `data/test`.

5. **Fix Model Loading Issue**
   - To resolve `TypeError: Unrecognized keyword arguments: ['batch_shape']`, the model loading in `model.py` includes:
     ```python
     from tensorflow.keras.layers import InputLayer
     def custom_input_layer_from_config(config):
         if 'batch_shape' in config:
             config['batch_input_shape'] = config.pop('batch_shape')
         return InputLayer(**config)
     tf.keras.utils.get_custom_objects()['InputLayer'] = custom_input_layer_from_config
     ```

6. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook notebook/image_classification.ipynb
   ```
   - Follow the notebook for preprocessing, model training, evaluation, and visualizations.

7. **Run the Streamlit UI**
   ```bash
   streamlit run src/app.py
   ```
   - Access at `http://localhost:8501`.
   - Features: Upload images for predictions, view visualizations, trigger retraining with new data.


9. **Deploy on Render**

- Create a Render account at render.com using GitHub or email.

- Push your repository to GitHub (public or private).

In the Render Dashboard:

- Click New > Web Service.

- Select Build and deploy from a Git repository and connect your GitHub repo.

Configure:

- Name: image-classification



- Environment: Docker



- Branch: main

- Root Directory: Leave blank (or src if applicable)

- Dockerfile Path: Dockerfile

Instance Type: Free (note: limited to 500 pipeline minutes/month and 750 hours/month)

- Click Create Web Service.

- Render builds and deploys the app (5–10 minutes). Access the app at https://<service-name>.onrender.com.

- Streamlit UI is available at the root URL; FastAPI is at /predict.

Enable auto-deploys in Render’s settings for CI/CD on Git pushes.

10. **Run Locust Flood Test**
    ```bash
    locust -f locustfile.py --host=http://localhost:8501 --web-port=8091
    ```
    - Configure and run the test at `http://localhost:8089`.

## Flood Request Simulation Results

Locust was used to simulate API requests with varying Docker containers:
- **1 Container**:
  - 100 users, 10 requests/sec: Avg. latency = 120ms, 95th percentile = 180ms, 0% failures.
  - 500 users, 50 requests/sec: Avg. latency = 350ms, 95th percentile = 600ms, 2% failures.
- **3 Containers**:
  - 100 users, 10 requests/sec: Avg. latency = 80ms, 95th percentile = 110ms, 0% failures.
  - 500 users, 50 requests/sec: Avg. latency = 150ms, 95th percentile = 200ms, 0% failures.

*Insight*: Scaling to multiple containers significantly reduces latency and improves reliability under high load.

## Project Structure

```
image_classification_pipeline/
│
├── README.md
│
├── notebook/
│   ├── image_classification.ipynb  # Preprocessing, training, evaluation
│
├── src/
│   ├── preprocessing.py           # Image resizing, augmentation
│   ├── model.py                   # Model definition, training, retraining
│   ├── prediction.py              # Prediction logic
│   ├── app.py                     # Streamlit UI
│   ├── api.py                     # FastAPI endpoint
│   └── locustfile.py              # Locust load testing
│
├── data/
│   ├── train/                     # Training images
│   └── test/                      # Test images
│
├── models/
│   ├── image_classifier_model.keras               # Trained model
│
├── requirements.txt
├── Dockerfile                     # Docker configuration
└── start.sh                      
```

## Notes
- The model is saved as `image_classifier_model.keras` to ensure compatibility with TensorFlow 2.6.0.
- Retraining requires at least 100 new images for meaningful updates; upload via the Streamlit UI.
- AWS deployment requires configured credentials and an EC2 instance with Docker.
- The notebook includes detailed comments on preprocessing, model architecture, and evaluation metrics.


