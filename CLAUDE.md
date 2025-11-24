# CLAUDE.md - AI Assistant Guide for Projects Repository

## Repository Overview

This is a **confidential NDA project** associated with Northumbria University and Lockheed Martin, focusing on AI and Cybersecurity applications. The repository contains 9 independent machine learning and computer vision projects demonstrating various AI techniques for healthcare, food waste reduction, image processing, and biometric identification.

**Status**: NDA in effect - Confidential. Some components are not available for public use due to ethical considerations.

## Repository Structure

```
/home/user/Projects/
├── README.md                          # Main documentation
├── DataAugmentation/                  # Data augmentation utilities
│   ├── DataAugmentation.py            # Symptom data augmentation
│   └── CNN.py                         # Image transformation pipeline
├── FacialRecognition/                 # Face recognition system
│   ├── UserRegistration.py            # Register new users
│   └── UserIdentification.py          # Identify/verify users
├── GenderClassificationMatlab/        # MATLAB gender classification
│   ├── ANN_training.m                 # Training script
│   ├── ANN_testing.m                  # Testing script
│   ├── get_featureVector.m            # DWT feature extraction
│   ├── Perceptron.m                   # Perceptron classifier
│   ├── Male/                          # Male training images
│   └── Female/                        # Female training images
├── HamSpamMatlab/                     # MATLAB spam detection
│   ├── spam_detection_fitcsvm_fitctree.m
│   ├── spam_detection_fitcnb.m
│   └── *.txt                          # Algorithm documentation
├── RottenDetector/                    # Food ripeness detection
│   ├── SubSystem.py                   # High-level inference API
│   ├── resnet50/                      # ResNet50 implementation
│   │   ├── RottenDetectorCVTrain.py   # Training script
│   │   └── RottenDetectorCVTest.py    # Testing script
│   └── resnet101/                     # ResNet101 implementation
│       ├── train.py                   # Training script
│       └── test.py                    # Testing script
├── SelfServiceNurse/                  # Medical triage application
│   ├── SelfServiceNurse.py            # Main application
│   └── kivy.kv                        # UI definition
├── SkinDiseaseDetection/              # CNN-based skin disease detection
│   └── AllCNN/                        # Multiple CNN architectures
│       ├── CNN/Train.py               # Basic CNN
│       ├── VGG19/Train.py             # VGG19 transfer learning
│       ├── CNN50/Train.py & Test.py   # ResNet50
│       └── CNN101/Train.py & Test.py  # ResNet101
├── SymptomDiagnosisMapping/           # Multi-algorithm ML comparison
│   ├── ANN/ANN.py                     # Artificial Neural Network
│   ├── RandomForest/Train.py          # Random Forest classifier
│   ├── LightGBM/TrainLight.py         # LightGBM multi-label
│   └── SVM/trainn.py                  # Support Vector Machine
└── WatermarkingMatlab/                # Image watermarking
    ├── DCTEmbedding.m & DCTDetection.m
    └── DWTEmbedding.m & DWTDetection.m
```

## Project Descriptions

### 1. Self-Service Nurse (Healthcare AI)
**Location**: `SelfServiceNurse/`
**Purpose**: Autonomous medical assessment tool combining CNN for skin disease identification and ANN for symptom analysis.

**Key Features**:
- Multi-screen Kivy GUI (SignIn → Triage → Habits → Symptoms → SkinRashes → Report)
- IoT sensor integration (ultrasonic, heart rate, temperature)
- Real-time skin disease detection via camera
- Comprehensive symptom checklist (150+ medical symptoms)
- File-based data persistence

**Entry Point**: `SelfServiceNurse.py`
**Framework**: Kivy (cross-platform GUI), OpenCV

### 2. RottenDetector (Food Waste Reduction)
**Location**: `RottenDetector/`
**Purpose**: Fruit ripeness classification with bruise detection to minimize food waste.

**Architectures**:
- **ResNet50**: 99.4% test accuracy
- **ResNet101**: 99.89% train accuracy, 99.98% validation accuracy

**Classes**: OverRipe, Ripe, Rotten, UnRipe

**Unique Features**:
- `SubSystem.py`: Advanced inference with HSV color-based bruise detection
- Morphological operations for noise reduction
- Bruise area percentage calculation

**Entry Points**:
- Training: `resnet50/RottenDetectorCVTrain.py` or `resnet101/train.py`
- Testing: `resnet50/RottenDetectorCVTest.py` or `resnet101/test.py`
- Inference: `SubSystem.py`

**Framework**: PyTorch with torchvision

### 3. Skin Disease Detection
**Location**: `SkinDiseaseDetection/AllCNN/`
**Purpose**: Multi-architecture CNN comparison for classifying 19 skin disease types.

**Models Implemented**:
- **Basic CNN**: 3 Conv2D layers, 150x150 input, 50 epochs
- **VGG19**: Transfer learning from ImageNet
- **ResNet50**: 15 epochs with learning rate scheduler
- **ResNet101**: Same architecture as ResNet50

**Key Features**:
- Data augmentation via ImageDataGenerator
- Dropout(0.5) regularization
- ReduceLROnPlateau learning rate scheduler
- Train/validation/test split

**Entry Points**: `AllCNN/{CNN,VGG19,CNN50,CNN101}/Train.py`
**Framework**: TensorFlow/Keras and PyTorch

### 4. Symptom Diagnosis Mapping
**Location**: `SymptomDiagnosisMapping/`
**Purpose**: Comparative study of ML algorithms for symptom-to-diagnosis classification.

**Models**:
1. **ANN** (`ANN/ANN.py`): 6-layer sequential (256→128→64→32→16→8)
2. **Random Forest** (`RandomForest/Train.py`): 100 estimators
3. **LightGBM** (`LightGBM/TrainLight.py`): Multi-label classification with MultiOutputClassifier
4. **SVM** (`SVM/trainn.py`): Single-label SVC (C=1.0, gamma='scale')

**Data Source**: `Dataset/Symptoms/augmented_dataset.csv`
**Metrics**: Accuracy, Precision, Recall, F1-Score

### 5. Facial Recognition
**Location**: `FacialRecognition/`
**Purpose**: User registration and identification via face recognition.

**Components**:
- `UserRegistration.py`: Captures face, generates encoding, stores pickle file
- `UserIdentification.py`: Real-time face matching, opens user folder on match

**Storage**: Pickle-serialized encodings in `user_data/{user_id}/`
**Framework**: face_recognition library, OpenCV

### 6. Gender Classification (MATLAB)
**Location**: `GenderClassificationMatlab/`
**Purpose**: Binary gender classification using DWT features and multiple classifiers.

**Pipeline**:
- Feature extraction: 7-level Haar wavelet decomposition (35 features)
- Classifiers: Perceptron, Linear SVM, Classification Tree
- Training: ~70 samples from Male/Female folders

**Entry Points**:
- Training: `ANN_training.m`
- Testing: `ANN_testing.m`
- Feature extraction: `get_featureVector.m`

### 7. Spam Detection (MATLAB)
**Location**: `HamSpamMatlab/`
**Purpose**: Text classification for spam/ham using multiple algorithms.

**Models**: Naive Bayes, SVM, Decision Tree

**Pipeline**:
- Text preprocessing: lowercase, punctuation removal, tokenization
- Stop word removal, Porter stemmer
- Bag of Words and TF-IDF feature extraction
- Split: 60% train, 20% validation, 20% test

**Entry Points**: `spam_detection_*.m`

### 8. Image Watermarking (MATLAB)
**Location**: `WatermarkingMatlab/`
**Purpose**: Copyright protection via DCT and DWT watermarking methods.

**Methods**:
- **DCT**: Discrete Cosine Transform (alpha=1.5 scaling factor)
- **DWT**: 2-level Discrete Wavelet Transform (Haar basis)

**Entry Points**:
- Embedding: `DCTEmbedding.m`, `DWTEmbedding.m`
- Detection: `DCTDetection.m`, `DWTDetection.m`

### 9. Data Augmentation
**Location**: `DataAugmentation/`
**Purpose**: Utilities for augmenting training data.

**Components**:
- `DataAugmentation.py`: Symptom data augmentation (generates 100K samples)
- `CNN.py`: Image transformation pipeline with PyTorch transforms

## Technologies & Frameworks

### Python Stack
- **Deep Learning**: PyTorch, TensorFlow, Keras
- **ML Libraries**: scikit-learn, LightGBM
- **Computer Vision**: OpenCV (cv2), Pillow (PIL), torchvision
- **Data Processing**: pandas, numpy
- **GUI**: Kivy
- **Biometrics**: face_recognition
- **Utilities**: tqdm, joblib

### MATLAB Stack
- Image Processing Toolbox
- Wavelet Toolbox
- Statistics and Machine Learning Toolbox
- Text Analytics Toolbox

### Model Persistence
- PyTorch: `.pth` files via `torch.save()`
- Keras: `.keras` files via `model.save()`
- scikit-learn: `.pkl` files via `pickle` or `joblib`
- MATLAB: `.mat` files

## Key Conventions & Patterns

### Naming Conventions
- Training scripts: `Train.py`, `train.py`, `*training.m`
- Testing scripts: `Test.py`, `test.py`, `*testing.m`
- Model files: `*.pth`, `*.keras`, `*.pkl`, `*.mat`

### Standard Training Pipeline
```python
# 1. Data loading with error handling (multiple encodings)
# 2. Preprocessing/normalization (ImageNet stats for images)
# 3. Train/validation/test split (typically 70/30 or 80/20)
# 4. Model initialization (often transfer learning)
# 5. Loss function & optimizer setup
# 6. Training loop with epoch iteration
# 7. Validation after each epoch
# 8. Model checkpointing (save best model)
# 9. Metrics calculation (Accuracy, Precision, Recall, F1)
```

### Transfer Learning Pattern
```python
# Load pretrained weights (ResNet50/101 from ImageNet)
# Replace final layer for task-specific classes
# Add Dropout(0.5) for regularization
# Use ReduceLROnPlateau for learning rate scheduling
```

### Data Handling Best Practices
- **Encoding robustness**: Try UTF-8, fallback to ISO-8859-1
- **Missing data**: Use `fillna(0)` for NaN values
- **Normalization**: ImageNet stats `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
- **Augmentation**: Apply both image-level and data-level transformations

### File Path Handling
⚠️ **Note**: Many scripts use hardcoded file paths. When modifying code:
- Check for absolute paths that may need adjustment
- Verify dataset directories exist before running
- Use relative paths where possible

## Expected Dataset Structure

```
Dataset/
├── Symptoms/
│   ├── medical_conditions.csv      # Original symptom data
│   └── augmented_dataset.csv       # Generated by DataAugmentation.py
├── SkinDisease/
│   └── archive/
│       ├── train/                  # ImageFolder format
│       │   ├── class1/
│       │   ├── class2/
│       │   └── ...
│       └── test/
└── (Other project-specific datasets)
```

## Development Workflows

### Working with Python Projects

1. **Before Making Changes**:
   - Read existing code to understand structure
   - Check for dataset dependencies
   - Verify imports and library availability
   - Review any configuration or hardcoded paths

2. **Training New Models**:
   ```bash
   # Example: Train RottenDetector ResNet50
   cd RottenDetector/resnet50
   python RottenDetectorCVTrain.py
   ```

3. **Testing Models**:
   ```bash
   # Example: Test RottenDetector
   cd RottenDetector/resnet50
   python RottenDetectorCVTest.py
   ```

4. **Running GUI Applications**:
   ```bash
   # Example: Self-Service Nurse
   cd SelfServiceNurse
   python SelfServiceNurse.py
   ```

### Working with MATLAB Projects

1. **Training**:
   ```matlab
   % Open MATLAB in project directory
   cd('GenderClassificationMatlab')
   ANN_training
   ```

2. **Testing**:
   ```matlab
   ANN_testing
   ```

### Model Evaluation Pattern
All projects use consistent metrics:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

## Dependencies

### Python Dependencies (Implied)
**Note**: No `requirements.txt` found. Infer dependencies from imports:

```txt
torch>=1.9.0
torchvision>=0.10.0
tensorflow>=2.5.0
keras>=2.5.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
lightgbm>=3.2.0
pandas>=1.3.0
numpy>=1.21.0
Pillow>=8.3.0
kivy>=2.0.0
face-recognition>=1.3.0
tqdm>=4.62.0
joblib>=1.0.0
```

### MATLAB Dependencies
- MATLAB R2019b or later
- Required Toolboxes: Image Processing, Wavelet, Statistics and ML, Text Analytics

## Best Practices for AI Assistants

### Code Modification Guidelines

1. **Always Read First**: Never propose changes to code you haven't read
2. **Understand Context**: Each project is independent with its own patterns
3. **Preserve Structure**: Maintain existing conventions and file organization
4. **Avoid Over-Engineering**:
   - Don't add features beyond what's requested
   - Don't refactor surrounding code unnecessarily
   - Don't add docstrings to unchanged code
5. **Security Awareness**: Check for command injection, XSS, SQL injection vulnerabilities
6. **Test Changes**: Verify modifications don't break existing functionality

### Path and File Handling

- **Hardcoded Paths**: Many scripts have hardcoded paths - identify and adjust as needed
- **Dataset Directories**: Verify existence before running training scripts
- **Model Output**: Check that output directories exist or create them
- **Relative Paths**: Prefer relative paths for portability when modifying code

### Model Training Considerations

- **GPU Availability**: Code assumes CUDA availability for PyTorch projects
- **Memory Requirements**: ResNet101 and VGG19 require significant memory
- **Training Time**: ResNet models may take hours to train on full datasets
- **Data Augmentation**: Already implemented - don't add unless specifically requested

### Multi-Architecture Projects

When working on projects with multiple models (e.g., SkinDiseaseDetection):
- Each architecture is in its own subdirectory
- Training parameters differ between models
- Compare results across architectures systematically
- Document performance differences

### MATLAB-Python Interaction

- Python and MATLAB projects are independent
- No cross-language calling observed
- Each has its own dataset and model persistence
- Consider suggesting Python equivalents for new MATLAB features

## Special Features to Understand

### HSV-Based Bruise Detection (RottenDetector)
`SubSystem.py` implements sophisticated image analysis:
- BGR→HSV color space conversion
- HSV range masking for bruise detection
- Morphological operations (MORPH_OPEN) for noise reduction
- Area percentage calculation

### Multi-Label Classification (LightGBM)
Handles comma-separated labels:
- Uses `MultiLabelBinarizer` for encoding
- Wraps LightGBM with `MultiOutputClassifier`
- Supports multiple diagnoses per symptom set

### Wavelet Transform Features (Gender Classification)
Feature extraction process:
- 7-level Discrete Wavelet Transform (Haar basis)
- Extracts approximation subband (A7) as 35 features
- Consistent dimensionality across all samples

### IoT Integration (SelfServiceNurse)
Placeholder methods for hardware:
- Ultrasonic sensor integration
- Heart rate monitor
- Temperature gauge
- Real-time webcam via cv2.VideoCapture
- Kivy Clock scheduler for 30Hz updates

## Confidentiality & NDA

### Important Notes
- **Status**: NDA in effect - Confidential
- **Organizations**: Northumbria University & Lockheed Martin
- **Public Use**: Some components explicitly marked as not for public use
- **ess-Al Project**: Mentioned in README but not in codebase - automated essay generator using GPT-3.5

### When Working on This Codebase
- Maintain confidentiality of proprietary algorithms
- Don't share code snippets externally without authorization
- Be aware of ethical considerations (e.g., plagiarism detection bypass)
- Respect the NDA requirements

## Code Quality Observations

### Strengths
✓ Clear separation of train/test logic
✓ Consistent evaluation metrics
✓ Proper data splitting
✓ Transfer learning utilization
✓ Multi-algorithm comparison

### Areas for Improvement
⚠️ No configuration files (requirements.txt, .env)
⚠️ Hardcoded file paths throughout
⚠️ Limited error handling in some files
⚠️ Print statements instead of logging framework
⚠️ Sparse code comments/docstrings
⚠️ No unit tests or integration tests

### When Contributing
Consider adding:
- Configuration management
- Logging framework
- Error handling improvements
- Type hints for Python functions
- Documentation strings
- Unit tests for critical functions

## Common Issues & Solutions

### Issue: Import Errors
**Solution**: Check that all dependencies are installed. Infer from import statements if needed.

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size, use smaller model variant (ResNet50 vs ResNet101), or use CPU.

### Issue: File Not Found Errors
**Solution**: Verify dataset structure matches expected layout. Check hardcoded paths in code.

### Issue: Encoding Errors in CSV Loading
**Solution**: Code already handles UTF-8/ISO-8859-1. If issues persist, check file encoding with `file -bi filename.csv`.

### Issue: Kivy Application Won't Start
**Solution**: Ensure Kivy is properly installed. Check that `.kv` file is in the same directory as Python script.

### Issue: MATLAB Toolbox Missing
**Solution**: Verify required toolboxes are licensed and installed. Use `ver` in MATLAB to check.

## Quick Reference

### Project Entry Points Summary

| Project | Command | Description |
|---------|---------|-------------|
| SelfServiceNurse | `python SelfServiceNurse.py` | Launch medical triage GUI |
| RottenDetector Training | `python resnet50/RottenDetectorCVTrain.py` | Train ripeness detector |
| RottenDetector Testing | `python SubSystem.py` | Run inference with bruise detection |
| Skin Disease Training | `python AllCNN/CNN50/Train.py` | Train ResNet50 for skin diseases |
| Symptom Diagnosis | `python ANN/ANN.py` | Train ANN for symptom mapping |
| Face Registration | `python UserRegistration.py` | Register new user face |
| Face Identification | `python UserIdentification.py` | Identify user by face |
| Gender Classification | `matlab -r "ANN_training"` | Train gender classifier |
| Spam Detection | `matlab -r "spam_detection_fitcnb"` | Train Naive Bayes spam detector |

### Model File Locations

- PyTorch: `models/*.pth`
- Keras: `AllCNN/*/model.keras`
- scikit-learn: `*/model.pkl`
- MATLAB: `*.mat`
- Face encodings: `user_data/{user_id}/*.pkl`

### Dataset Locations

- Symptoms: `Dataset/Symptoms/`
- Skin Diseases: `Dataset/SkinDisease/archive/`
- Gender: `GenderClassificationMatlab/Male/` and `Female/`
- Ripeness: `VEG/` (ImageFolder structure)

---

## Final Notes for AI Assistants

This repository represents a comprehensive portfolio of AI/ML projects spanning healthcare, food waste reduction, biometrics, and image processing. When working with this codebase:

1. **Respect Independence**: Each project is self-contained
2. **Maintain Consistency**: Follow existing patterns within each project
3. **Verify Datasets**: Ensure data availability before running scripts
4. **Test Thoroughly**: Changes in one project shouldn't affect others
5. **Document Changes**: Maintain clear commit messages
6. **Consider Ethics**: Be mindful of confidentiality and ethical implications

For questions or clarifications, refer to the README.md or examine the specific project's source code directly.

---

*Last Updated*: 2025-11-24
*Repository Owner*: Lockheed Martin / Northumbria University (Confidential)
