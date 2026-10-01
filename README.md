# Speech Emotion Recognition
An LSTM-based deep learning project for classifying human speech into different emotional categories using
**MFCC (Mel-Frequency Cepstral Coefficients)** audio features.

## 📌 Project Overview
This project uses the **Toronto Emotional Speech Set (TESS)** dataset to recognize emotions from speech recordings.
The audio files are converted into numerical MFCC features, which are then processed using an **LSTM neural network** for multi-class emotion classification.
The model classifies speech into **7 emotions** and achieves **98.21% accuracy on unseen test data**.

## 🎯 Objectives
* Extract meaningful audio features from speech recordings using MFCC.
* Build an LSTM-based deep learning model for emotion classification.
* Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
* Test the model on individual audio samples and display prediction confidence.

## 📊 Dataset
The project uses the **Toronto Emotional Speech Set (TESS)** dataset.
* **Total audio samples:** 2,800
* **Number of emotions:** 7
* **Samples per emotion:** 400
* **Test samples:** 560
* **Training samples:** 2,240

### Emotion Classes
| Emotion           | Samples |
| ----------------- | ------: |
| Angry             |     400 |
| Disgust           |     400 |
| Fear              |     400 |
| Happy             |     400 |
| Neutral           |     400 |
| Pleasant Surprise |     400 |
| Sad               |     400 |

The dataset contains speech recordings from the OAF and YAF speaker groups.
> The audio dataset is not included in this repository. Please obtain the TESS dataset separately and update the dataset path in the notebook before running the project.

## 🔧 Methodology
### 1. Data Loading
The project scans the dataset directory for `.wav` audio files and extracts emotion labels from the filenames.
### 2. MFCC Feature Extraction
Each audio file is loaded using **Librosa** and converted into **40 MFCC features**.
The resulting input shape is:
```text
(2800, 40, 1)
```
MFCCs provide a compact representation of the characteristics of speech that can be used for audio classification.
### 3. Label Encoding
The seven emotion labels are converted into one-hot encoded vectors for multi-class classification.
### 4. Train-Test Split
The dataset is divided using an **80/20 stratified split**:
* Training: 2,240 samples
* Testing: 560 samples
Stratification maintains the class distribution across the training and testing sets.
### 5. LSTM Model
The neural network consists of:
```text
Input: 40 MFCC features
        ↓
LSTM (128 units)
        ↓
Dense (64 units, ReLU)
        ↓
Dropout (20%)
        ↓
Dense (32 units, ReLU)
        ↓
Dropout (20%)
        ↓
Dense (7 units, Softmax)
```
### 6. Model Training
The model was trained using:
* **Optimizer:** Adam
* **Loss:** Categorical Cross-Entropy
* **Epochs:** 50
* **Batch Size:** 64
* **Activation:** ReLU for hidden layers, Softmax for output

## 📈 Results
| Metric            |     Result |
| ----------------- | ---------: |
| Training Accuracy |     99.33% |
| Testing Accuracy  | **98.21%** |
| Accuracy Gap      |      1.12% |
| Macro F1-Score    |   **0.98** |

The small **1.12 percentage-point gap** between training and testing accuracy indicates that the model maintained strong performance on the unseen test set.

### Classification Performance
| Emotion           | Precision |   Recall | F1-Score |
| ----------------- | --------: | -------: | -------: |
| Angry             |      0.96 |     0.97 |     0.97 |
| Disgust           |      0.99 |     0.96 |     0.97 |
| Fear              |      1.00 |     0.99 |     0.99 |
| Happy             |      0.99 |     0.97 |     0.98 |
| Neutral           |      0.98 |     1.00 |     0.99 |
| Pleasant Surprise |      0.96 |     0.99 |     0.98 |
| Sad               |      1.00 |     0.99 |     0.99 |
| **Macro Average** |  **0.98** | **0.98** | **0.98** |

## 📊 Visualizations & Evaluation
The notebook includes:

* Emotion class distribution
* MFCC visualization
* Training vs. validation accuracy curve
* Training vs. validation loss curve
* Confusion matrix
* Classification report
* Individual audio prediction with confidence scores

The model can also take an individual `.wav` file and display the predicted emotion along with the probability distribution across all seven classes.

## 🛠️ Technologies Used
* Python
* TensorFlow / Keras
* Librosa
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## 📁 Repository Structure
```text
speech-emotion-recognition/
│
├── speech emotion .ipynb
└── README.md
```

## 🚀 How to Run
1. Download the TESS dataset.
2. Clone or download this repository.
3. Install the required Python libraries.
4. Update the dataset path in the notebook.
5. Run the notebook cells sequentially.

Example:

```bash
pip install librosa tensorflow pandas numpy scikit-learn matplotlib seaborn
```

## 💡 Key Takeaway
This project demonstrates an end-to-end **speech emotion recognition pipeline**, from raw audio preprocessing and MFCC feature extraction
to LSTM-based classification and model evaluation.
The final model achieved **98.21% test accuracy** and a **0.98 macro F1-score** across seven emotion classes.

## 👩‍💻 Author
**Aarya Mangsulikar**
MSc Data Science
