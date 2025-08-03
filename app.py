import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from PIL import Image
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Image Classification ML Pipeline",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin: 1rem 0;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 20px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def create_sample_files():
    """Create sample files for demonstration if they don't exist"""
    os.makedirs('models', exist_ok=True)
    
    # Create sample label encoder file
    if not os.path.exists('models/label_encoder.json'):
        sample_classes = ['cats', 'dogs', 'birds', 'cars', 'flowers']
        with open('models/label_encoder.json', 'w') as f:
            json.dump({'classes': sample_classes}, f)
        st.info("Created sample label_encoder.json file")
    
    # Create sample metrics file
    if not os.path.exists('models/evaluation_metrics.txt'):
        sample_metrics = """Accuracy: 0.8542
Precision: 0.8321
Recall: 0.8456
F1-Score: 0.8387"""
        with open('models/evaluation_metrics.txt', 'w') as f:
            f.write(sample_metrics)
        st.info("Created sample evaluation_metrics.txt file")

@st.cache_data
def load_model_and_encoder():
    """Load the trained model and label encoder"""
    try:
        # Load model
        if not os.path.exists('models/image_classifier.h5'):
            st.error("Model file 'models/image_classifier.h5' not found.")
            return None, None
            
        model = tf.keras.models.load_model('models/image_classifier.h5')
        
        # Load class names
        class_names = None
        
        if os.path.exists('models/label_encoder.json'):
            with open('models/label_encoder.json', 'r') as f:
                label_data = json.load(f)
                
            # Try different possible structures
            if 'classes' in label_data:
                class_names = label_data['classes']
            elif isinstance(label_data, list):
                class_names = label_data
            elif isinstance(label_data, dict) and len(label_data) > 0:
                # Check if it's a dict mapping string indices to class names (like "0": "airplane")
                if all(str(i) in label_data for i in range(len(label_data))):
                    # Sort by index to maintain order
                    class_names = [label_data[str(i)] for i in sorted([int(k) for k in label_data.keys()])]
                else:
                    # If it's a dict mapping class names to indices
                    class_names = list(label_data.keys())
            else:
                st.warning("Unexpected label encoder format. Using default class names.")
                
        # If no label encoder file or class names found, create default ones
        if class_names is None:
            num_classes = model.output_shape[-1] if model else 5
            class_names = [f"Class_{i}" for i in range(num_classes)]
            st.info(f"Using default class names: {class_names}")
            
        return model, class_names
        
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None
    except json.JSONDecodeError as e:
        st.error(f"Error reading JSON file: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image_for_prediction(image):
    """Preprocess image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize to 224x224
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_image(model, image, class_names):
    """Make prediction on uploaded image"""
    processed_img = preprocess_image_for_prediction(image)
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]

@st.cache_data
def load_evaluation_metrics():
    """Load evaluation metrics from file"""
    try:
        metrics = {}
        with open('models/evaluation_metrics.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                metrics[key] = float(value)
        return metrics
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return {}

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Image Classification ML Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("**African Leadership University - Machine Learning Project**")
    st.markdown("*Author: Amandine Irakoze | Date: 30-07-2025*")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Setup section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Setup")
    if st.sidebar.button("Create Sample Files", help="Create sample files for demonstration"):
        create_sample_files()
        st.rerun()
    
    # Check if model exists
    model_exists = os.path.exists('models/image_classifier.h5')
    if model_exists:
        st.sidebar.success("✅ Model loaded")
    else:
        st.sidebar.error("❌ Model not found")
        st.sidebar.markdown("Upload your trained model to `models/image_classifier.h5`")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Prediction", "📊 Model Performance", "📈 Data Analysis", "🧠 Model Architecture"]
    )
    
    # Load model and classes
    model, class_names = load_model_and_encoder()
    
    if page == "🏠 Home":
        home_page()
    elif page == "🔮 Prediction":
        prediction_page(model, class_names)
    elif page == "📊 Model Performance":
        performance_page()
    elif page == "📈 Data Analysis":
        data_analysis_page()
    elif page == "🧠 Model Architecture":
        model_architecture_page(model)

def home_page():
    """Home page with project overview"""
    st.markdown('<h2 class="sub-header">Project Overview</h2>', unsafe_allow_html=True)
    
    # Check if model exists
    if not os.path.exists('models/image_classifier.h5'):
        st.warning("⚠️ No trained model found. Please upload your model or create sample files for demonstration.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔧 Create Sample Files", type="primary"):
                create_sample_files()
                st.success("Sample files created! Refresh the page to see changes.")
                st.rerun()
        
        with col2:
            st.markdown("**Or upload your trained model files:**")
            model_file = st.file_uploader("Upload model (.h5)", type=['h5'])
            if model_file:
                os.makedirs('models', exist_ok=True)
                with open('models/image_classifier.h5', 'wb') as f:
                    f.write(model_file.read())
                st.success("Model uploaded successfully!")
                st.rerun()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This project demonstrates an **end-to-end machine learning pipeline** for image classification, featuring:
        
        - 🎯 **Data Acquisition & Exploration**: Comprehensive dataset analysis
        - 🔧 **Data Preprocessing**: Image resizing, normalization, and augmentation
        - 🏗️ **Model Architecture**: CNN with ResNet50 transfer learning
        - 🎓 **Model Training**: Training with validation and performance monitoring
        - 📊 **Model Evaluation**: Comprehensive metrics and analysis
        - 🚀 **Model Deployment**: Ready for production deployment
        - 🧪 **Performance Testing**: Load testing and optimization
        """)
        
        st.markdown("### 🎯 Key Features")
        st.markdown("""
        - **Interactive Prediction Interface**: Upload images and get instant predictions
        - **Comprehensive Model Analysis**: View training history, metrics, and feature analysis
        - **Real-time Performance Monitoring**: Track model performance and accuracy
        - **Deployment Ready**: Containerized and cloud-ready architecture
        """)
    
    with col2:
        st.markdown("### 📋 Project Structure")
        st.code("""
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── image_classifier.h5
│   ├── label_encoder.json
│   └── evaluation_metrics.txt
├── src/
│   ├── preprocessing.py
│   └── prediction.py
└── streamlit_app.py
        """)
        
        # File status
        st.markdown("### 📁 File Status")
        files_status = {
            "Model": "✅" if os.path.exists('models/image_classifier.h5') else "❌",
            "Label Encoder": "✅" if os.path.exists('models/label_encoder.json') else "❌",
            "Metrics": "✅" if os.path.exists('models/evaluation_metrics.txt') else "❌"
        }
        
        for file_name, status in files_status.items():
            st.write(f"{status} {file_name}")


def prediction_page(model, class_names):
    """Image prediction page"""
    st.markdown('<h2 class="sub-header">🔮 Image Prediction</h2>', unsafe_allow_html=True)
    
    if model is None or class_names is None:
        st.error("⚠️ Model not loaded. Please ensure the model file exists in the 'models' directory.")
        return
    
    st.markdown("Upload an image to get a prediction from the trained model.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG, JPG, or JPEG image file"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown("**Image Details:**")
            st.write(f"- **Size**: {image.size}")
            st.write(f"- **Mode**: {image.mode}")
            st.write(f"- **Format**: {image.format}")
        
        with col2:
            st.markdown("### 🎯 Prediction Results")
            
            with st.spinner("Making prediction..."):
                try:
                    predicted_class, confidence, all_predictions = predict_image(model, image, class_names)
                    
                    # Display main prediction
                    st.success(f"**Predicted Class**: {predicted_class}")
                    st.info(f"**Confidence**: {confidence:.2%}")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # All predictions
                    st.markdown("### 📊 All Class Probabilities")
                    
                    # Create DataFrame for predictions
                    pred_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': all_predictions
                    }).sort_values('Probability', ascending=False)
                    
                    # Display as bar chart
                    fig = px.bar(
                        pred_df, 
                        x='Probability', 
                        y='Class',
                        orientation='h',
                        title="Class Prediction Probabilities",
                        color='Probability',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display as table
                    st.markdown("### 📋 Detailed Probabilities")
                    pred_df['Probability'] = pred_df['Probability'].apply(lambda x: f"{x:.4f}")
                    st.dataframe(pred_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

def performance_page():
    """Model performance and metrics page"""
    st.markdown('<h2 class="sub-header">📊 Model Performance</h2>', unsafe_allow_html=True)
    
    # Load metrics
    metrics = load_evaluation_metrics()
    
    if not metrics:
        st.error("⚠️ No evaluation metrics found. Please ensure the model has been trained and evaluated.")
        return
    
    # Display metrics in cards
    st.markdown("### 🎯 Model Evaluation Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Accuracy",
            value=f"{metrics.get('Accuracy', 0):.4f}",
            delta=f"{(metrics.get('Accuracy', 0) - 0.5):.4f}"
        )
    
    with col2:
        st.metric(
            label="⚡ Precision",
            value=f"{metrics.get('Precision', 0):.4f}",
            delta=f"{(metrics.get('Precision', 0) - 0.5):.4f}"
        )
    
    with col3:
        st.metric(
            label="🔄 Recall",
            value=f"{metrics.get('Recall', 0):.4f}",
            delta=f"{(metrics.get('Recall', 0) - 0.5):.4f}"
        )
    
    with col4:
        st.metric(
            label="🎯 F1-Score",
            value=f"{metrics.get('F1-Score', 0):.4f}",
            delta=f"{(metrics.get('F1-Score', 0) - 0.5):.4f}"
        )
    
    # Metrics visualization
    st.markdown("### 📈 Metrics Comparison")
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [metrics.get('Accuracy', 0), metrics.get('Precision', 0), 
                 metrics.get('Recall', 0), metrics.get('F1-Score', 0)]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig_bar = px.bar(
            metrics_df, 
            x='Metric', 
            y='Value',
            title="Model Performance Metrics",
            color='Value',
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Radar chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=metrics_df['Value'],
            theta=metrics_df['Metric'],
            fill='toself',
            name='Model Performance'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Performance Radar Chart",
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Performance interpretation
    st.markdown("### 💡 Performance Interpretation")
    
    avg_performance = sum(metrics.values()) / len(metrics)
    
    if avg_performance > 0.9:
        st.success("🎉 **Excellent Performance**: The model shows outstanding performance across all metrics!")
    elif avg_performance > 0.8:
        st.info("✅ **Good Performance**: The model performs well with room for minor improvements.")
    elif avg_performance > 0.7:
        st.warning("⚠️ **Fair Performance**: The model shows decent performance but may need optimization.")
    else:
        st.error("❌ **Poor Performance**: The model needs significant improvements.")

def data_analysis_page():
    """Data analysis and feature visualization page"""
    st.markdown('<h2 class="sub-header">📈 Data Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This section provides insights into the dataset characteristics and feature analysis.
    """)
    
    # Simulated data for demonstration (replace with actual data loading)
    st.markdown("### 📊 Dataset Overview")
    
    # Dataset statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📁 Dataset Statistics")
        dataset_stats = {
            "Total Images": 1000,
            "Training Images": 800,
            "Test Images": 200,
            "Number of Classes": 5,
            "Image Resolution": "224x224",
            "Color Channels": 3
        }
        
        for key, value in dataset_stats.items():
            st.metric(key, value)
    
    with col2:
        st.markdown("#### 🏷️ Class Distribution")
        # Simulated class distribution
        class_dist = pd.DataFrame({
            'Class': ['Class A', 'Class B', 'Class C', 'Class D', 'Class E'],
            'Count': [180, 220, 200, 190, 210]
        })
        
        fig_pie = px.pie(
            class_dist, 
            values='Count', 
            names='Class',
            title="Training Data Class Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Feature analysis
    st.markdown("### 🔍 Feature Analysis")
    
    tab1, tab2, tab3 = st.tabs(["RGB Analysis", "Image Properties", "Feature Maps"])
    
    with tab1:
        st.markdown("#### 🌈 RGB Channel Analysis")
        st.markdown("Analysis of color distribution across RGB channels in the dataset.")
        
        # Simulated RGB histogram data
        rgb_data = pd.DataFrame({
            'Pixel_Value': list(range(256)) * 3,
            'Frequency': np.random.gamma(2, 2, 768),
            'Channel': ['Red'] * 256 + ['Green'] * 256 + ['Blue'] * 256
        })
        
        fig_rgb = px.line(
            rgb_data, 
            x='Pixel_Value', 
            y='Frequency', 
            color='Channel',
            title="RGB Channel Distribution"
        )
        st.plotly_chart(fig_rgb, use_container_width=True)
    
    with tab2:
        st.markdown("#### 📐 Image Properties Analysis")
        
        # Simulated image size distribution
        sizes_data = pd.DataFrame({
            'Width': np.random.normal(300, 50, 100),
            'Height': np.random.normal(300, 50, 100)
        })
        
        fig_scatter = px.scatter(
            sizes_data, 
            x='Width', 
            y='Height',
            title="Original Image Size Distribution",
            opacity=0.6
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.markdown("#### 🧠 Feature Maps Visualization")
        st.markdown("Visualization of learned features from the convolutional layers.")
        
        # Simulated feature map data
        feature_map = np.random.rand(64, 64)
        
        fig_heatmap = px.imshow(
            feature_map,
            title="Sample Feature Map from Conv2D Layer",
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

def model_architecture_page(model):
    """Model architecture visualization page"""
    st.markdown('<h2 class="sub-header">🧠 Model Architecture</h2>', unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure the model file exists.")
        return
    
    # Model summary
    st.markdown("### 📋 Model Summary")
    
    # Capture model summary
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    summary_text = '\n'.join(summary_list)
    
    st.code(summary_text, language='text')
    
    # Model architecture details
    st.markdown("### 🏗️ Architecture Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Layer Information")
        layer_info = []
        for i, layer in enumerate(model.layers):
            layer_info.append({
                'Layer': i + 1,
                'Name': layer.name,
                'Type': type(layer).__name__,
                'Output Shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A'
            })
        
        layer_df = pd.DataFrame(layer_info)
        st.dataframe(layer_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Model Statistics")
        
        # Count parameters
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        stats = {
            "Total Parameters": f"{total_params:,}",
            "Trainable Parameters": f"{trainable_params:,}",
            "Non-trainable Parameters": f"{non_trainable_params:,}",
            "Number of Layers": len(model.layers),
            "Input Shape": str(model.input_shape),
            "Output Shape": str(model.output_shape)
        }
        
        for key, value in stats.items():
            st.metric(key, value)
    
    # Architecture visualization
    st.markdown("### 🎨 Architecture Visualization")
    
    # Layer type distribution
    layer_types = [type(layer).__name__ for layer in model.layers]
    layer_counts = pd.Series(layer_types).value_counts()
    
    fig_layers = px.pie(
        values=layer_counts.values,
        names=layer_counts.index,
        title="Layer Type Distribution"
    )
    st.plotly_chart(fig_layers, use_container_width=True)
    
    # Model configuration
    st.markdown("### ⚙️ Model Configuration")
    
    config_info = {
        "Optimizer": "Adam",
        "Loss Function": "Sparse Categorical Crossentropy",
        "Metrics": "Accuracy",
        "Input Size": "224x224x3",
        "Batch Size": "32",
        "Training Epochs": "20"
    }
    
    config_df = pd.DataFrame(list(config_info.items()), columns=['Parameter', 'Value'])
    st.table(config_df)

if __name__ == "__main__":
    st.set_page_config(page_title="My App")