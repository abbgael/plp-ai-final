# 🫁 Pneumonia Detection in Chest X-rays

A comprehensive deep learning solution for automated pneumonia detection in chest X-ray images using transfer learning and advanced data visualization.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Results & Visualization](#results--visualization)
- [File Outputs](#file-outputs)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a state-of-the-art convolutional neural network for detecting pneumonia in chest X-ray images. The system uses transfer learning with a pre-trained VGG16 model and provides comprehensive analytics, visualizations, and performance metrics.

### Key Highlights
- **High Accuracy**: Achieves excellent performance through transfer learning
- **Comprehensive Analytics**: Detailed dataset demographics and model evaluation
- **Rich Visualizations**: 12+ different charts and graphs for insights
- **Production Ready**: Includes model checkpointing and proper evaluation metrics
- **Easy to Use**: Single script execution with minimal configuration

## ✨ Features

### 🔍 Data Analysis
- **Dataset Demographics**: Automatic analysis of train/validation/test splits
- **Class Distribution**: Visual representation of normal vs pneumonia cases
- **Imbalance Detection**: Identifies and handles class imbalance issues

### 🤖 Model Training
- **Transfer Learning**: Uses pre-trained VGG16 for feature extraction
- **Data Augmentation**: Advanced image augmentation for better generalization
- **Class Weighting**: Automatic handling of imbalanced datasets
- **Smart Callbacks**: Early stopping, learning rate reduction, model checkpointing

### 📊 Visualization & Evaluation
- **Training Metrics**: Real-time tracking of accuracy, loss, precision, recall
- **Confusion Matrix**: Detailed performance breakdown
- **ROC Curve**: Area under curve analysis
- **Sample Predictions**: Visual inspection of model predictions
- **Probability Distributions**: Understanding model confidence

## 📁 Dataset Structure

Your dataset should be organized as follows:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── PNEUMONIA/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Supported Image Formats
- `.jpg`, `.jpeg`, `.png`, `.bmp`
- Recommended size: Any size (will be resized to 224x224)
- Color: RGB or Grayscale (will be converted to RGB)

## 🚀 Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.0+
- CUDA GPU (recommended but not required)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection

# Install required packages
pip install tensorflow>=2.8.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install opencv-python>=4.5.0
pip install pillow>=8.0.0
pip install pandas>=1.3.0
pip install numpy>=1.21.0
```

### For Google Colab
```bash
# Most packages are pre-installed in Colab
!pip install opencv-python
```

## 🏃 Quick Start

### 1. Prepare Your Data
- Upload your chest X-ray dataset to Google Drive (if using Colab)
- Ensure the folder structure matches the required format

### 2. Update Configuration
```python
# In the Config class, update the base path
BASE_PATH = '/content/drive/MyDrive/your_chest_xray_folder'  # Google Colab
# or
BASE_PATH = '/path/to/your/chest_xray_folder'  # Local machine
```

### 3. Run the Model
```python
python pneumonia_detection.py
```

### 4. For Google Colab
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Run the main function
main()
```

## ⚙️ Configuration

### Key Parameters
```python
class Config:
    # Data path
    BASE_PATH = '/path/to/your/data'
    
    # Image parameters
    IMG_SIZE = (224, 224)      # Input image size
    BATCH_SIZE = 32            # Training batch size
    
    # Training parameters
    EPOCHS = 50                # Maximum training epochs
    LEARNING_RATE = 0.0001     # Initial learning rate
    
    # Model parameters
    MODEL_NAME = 'pneumonia_detector'  # Model save name
```

### Customization Options
- **IMG_SIZE**: Change input image dimensions
- **BATCH_SIZE**: Adjust based on GPU memory
- **EPOCHS**: Increase for potentially better performance
- **LEARNING_RATE**: Fine-tune learning rate
- **Base Model**: Switch from VGG16 to ResNet50 or other architectures

## 🏗️ Model Architecture

### Base Architecture
```
Input (224x224x3)
    ↓
VGG16 (Pre-trained, Frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.5)
    ↓
Dense (512, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (256, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (1, Sigmoid)
```

### Key Features
- **Transfer Learning**: Leverages ImageNet pre-trained weights
- **Regularization**: Multiple dropout layers prevent overfitting
- **Binary Classification**: Sigmoid activation for pneumonia/normal classification
- **Balanced Training**: Class weights handle dataset imbalance

## 📈 Results & Visualization

The script generates comprehensive visualizations:

### 1. Dataset Demographics (4 charts)
- Bar plot of image counts by split
- Overall dataset distribution pie chart
- Stacked distribution visualization
- Class imbalance ratios

### 2. Training History (4 plots)
- Training & validation accuracy
- Training & validation loss
- Precision metrics over time
- Recall metrics over time

### 3. Model Evaluation (4 visualizations)
- Confusion matrix heatmap
- ROC curve with AUC score
- Prediction probability distributions
- Per-class metrics heatmap

### 4. Sample Predictions
- Visual display of 8 test images
- True vs predicted labels
- Confidence scores
- Color-coded accuracy (green=correct, red=incorrect)

## 📂 File Outputs

After training completion, the following files are generated:

### Model Files
- `pneumonia_detector_best.h5`: Best model weights
- Model can be loaded with: `tf.keras.models.load_model('pneumonia_detector_best.h5')`

### Data Files
- `training_history.csv`: Complete training metrics log
- `dataset_demographics.csv`: Dataset statistics summary

### Generated Plots
All visualizations are displayed during execution and can be saved manually.

## 📊 Performance Metrics

The model is evaluated using multiple metrics:

### Primary Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Additional Metrics
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of predictions
- **Class-wise Performance**: Metrics for both normal and pneumonia classes

### Expected Performance
- **Training Accuracy**: 90-95%
- **Validation Accuracy**: 85-92%
- **Test Accuracy**: 85-90%
- **AUC Score**: 0.90-0.95

*Note: Actual performance depends on dataset quality and size*

## 🔧 Troubleshooting

### Common Issues

#### 1. "No such file or directory" Error
```python
# Solution: Check your BASE_PATH in Config class
BASE_PATH = '/correct/path/to/your/chest_xray'
```

#### 2. Google Drive Mount Issues
```python
# Ensure you mount the drive first
from google.colab import drive
drive.mount('/content/drive')
```

#### 3. Out of Memory Error
```python
# Reduce batch size in Config class
BATCH_SIZE = 16  # or even 8 for very limited memory
```

#### 4. Low Performance
- **Check data quality**: Ensure images are clear and properly labeled
- **Increase epochs**: Try EPOCHS = 100
- **Adjust learning rate**: Try LEARNING_RATE = 0.00005
- **More data**: Consider data augmentation or additional training data

#### 5. Training Too Slow
- **Use GPU**: Enable GPU in Google Colab or use local GPU
- **Reduce image size**: Try IMG_SIZE = (128, 128)
- **Increase batch size**: If you have enough memory

### Performance Optimization

#### For Better Accuracy
```python
# Fine-tune these parameters
LEARNING_RATE = 0.00005    # Lower learning rate
EPOCHS = 100               # More training epochs
BATCH_SIZE = 16            # Smaller batches for better gradient updates
```

#### For Faster Training
```python
# Speed optimization
BATCH_SIZE = 64           # Larger batches
IMG_SIZE = (128, 128)     # Smaller images
```

## 🧪 Advanced Usage

### Custom Model Architecture
```python
def build_custom_model():
    # Use ResNet50 instead of VGG16
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.IMG_SIZE, 3)
    )
    # ... rest of the model
```

### Custom Data Augmentation
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,        # Increased rotation
    width_shift_range=0.3,    # More shifting
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,       # Add vertical flip
    zoom_range=0.3,           # More zoom
    brightness_range=[0.8, 1.2],  # Brightness variation
    fill_mode='nearest'
)
```

### Inference on New Images
```python
# Load trained model
model = tf.keras.models.load_model('pneumonia_detector_best.h5')

# Predict on new image
def predict_pneumonia(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=config.IMG_SIZE
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)[0][0]
    result = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return result, confidence
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Contribution Ideas
- Support for additional image formats
- Integration with other pre-trained models
- Real-time prediction interface
- Mobile app development
- Performance optimizations
- Additional evaluation metrics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- **VGG16 Paper**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- **Transfer Learning**: [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
- **Medical Imaging**: [Deep learning for medical image analysis](https://www.nature.com/articles/s41591-018-0316-z)

## 📞 Support

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/pneumonia-detection/issues)
- **Email**: your.email@example.com
- **Documentation**: Check this README and code comments

---

## 🎉 Acknowledgments

- **Dataset**: Thanks to the chest X-ray dataset contributors
- **TensorFlow**: For the excellent deep learning framework
- **Medical Community**: For advancing AI in healthcare
- **Open Source Community**: For continuous improvements

---

**⭐ If you find this project helpful, please consider giving it a star!**

---

*Last updated: August 2025*
