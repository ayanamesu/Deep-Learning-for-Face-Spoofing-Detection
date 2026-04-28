# Deep-Learning-for-Face-Spoofing-Detection

## Project Overview
This project implements a CNN-based deepfake detection system to classify whether a face video 
is real or spoofed using the FakeAVCeleb_v1.2 dataset. The system extracts frames from videos, 
preprocesses them using MobileNetV2 preprocessing, and trains a fine-tuned MobileNetV2 
classifier to distinguish real from fake faces.

------------------------------------------------------------------------------------------------------------------------------------------

## Data Set
- **Dataset:** FakeAVCeleb_v1.2
- **Total videos:** 21,544 (1,000 real, 20,544 fake)
- **Subjects:** Multiple races and genders                              <---------->
- **Modalities:** Video (.mp4)
- **Resolution:** Frames resized to 128x128 during preprocessing
- **Access:** Dataset is too large for GitHub. Download via Google Drive shared 
  by Sravani Gurram (accessible through Canvas conversation)

------------------------------------------------------------------------------------------------------------------------------------------

## How to set up dataset:
1. Download FakeAVCeleb_v1.2 from the shared Google Drive link on Canvas
2. Extract the folder
3. Place it in the root project directory so the path is:
   `FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/`

------------------------------------------------------------------------------------------------------------------------------------------

## Preprocessing Pipeline
1. Videos are split 80/20 into train and test sets with stratification
2. All real videos are forced into training to handle class imbalance (800 real, 3000 fake sampled)
3. Frames are extracted using random sampling (15 frames per video)
4. Face detection applied using Haar Cascade before resizing to 128x128
5. MobileNetV2 preprocessing applied (scales pixels to [-1, 1])
6. Training set balanced to 3:1 fake/real ratio using undersampling
7. Class weights set to {fake: 1.0, real: 3.0} to penalize minority class errors

------------------------------------------------------------------------------------------------------------------------------------------


## Model Architecture
- Base: MobileNetV2 pretrained on ImageNet (last 80 layers unfrozen)
- GlobalAveragePooling2D
- Dense(256, relu)
- BatchNormalization
- Dropout(0.4)
- Dense(1, sigmoid)

------------------------------------------------------------------------------------------------------------------------------------------

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 97.4% |
| EER | 2.6% |
| FAR | 2.6% |
| FRR | 2.7% |

------------------------------------------------------------------------------------------------------------------------------------------

## Requirements
Install required libraries. (This is according to my code to Readme set up so its pretty linear step)

1. pip install opencv-python numpy matplotlib scikit-learn tensorflow

------------------------------------------------------------------------------------------------------------------------------------------

## How to Run
1. Clone the repository
2. Set up the dataset as described above
3. Open `face_spoofing.ipynb` in Jupyter Notebook or JupyterLab
4. Select the Python kernel
5. Run all cells in order from top to bottom

------------------------------------------------------------------------------------------------------------------------------------------

## Project Structure
1. FakeAVCeleb_v1.2 -> dataset
2. meta_data.csv → inside of FakeAVCeleb_v1.2 folder Dataset metadata
3. deep_learning_based_face_spoofing_detection -> overleaf report
4. face_spoofing.ipynb → Main training and evaluation notebook
5. meta_data.csv → Dataset metadata
6. ReadME.md -> current file 
7. requirements.txt -> the run command 
