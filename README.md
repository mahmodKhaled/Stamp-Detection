# Stamp-Detection
Two models for detecting and localizing stamps in document images, one based on traditional image processing techniques, and another using a YOLO-based deep learning approach.

## Dataset Source

The dataset is a manually assembled open source dataset for the tasks of finding seals and signatures.

The dataset is from **Kaggle** at this link: [Signatures and Stamps](https://www.kaggle.com/datasets/igorkarayman/signatures-and-stamps/data).

## Labeled Dataset Source

To generate labeled data, I built a Streamlit application in this script **label_images_app.py** that uses a stamp detection algorithm based on traditional image processing techniques. This application automatically labeled 300 images from the original Kaggle dataset, creating YOLO-compatible annotations for training purposes.

The labeled dataset can be found here: [Labeled Stamps](https://kaggle.com/datasets/b4929ea694c75fca467320bd956de9e6474d1c208d5a48cdb8449f4a6c48f715).


## Installation

Follow these steps to set up the project on your local machine:

1. Clone the repository:

```bash
git clone https://github.com/your-username/stamp-detection.git
cd stamp-detection
```

2. Install Poetry
If you don't have Poetry installed, you can install it using pip:

```bash
pip install poetry
```

3. Install dependencies and set up the environment

```bash
poetry install
```

4. Activate the virtual environment

```bash
poetry shell
```

5. Run the main application
This application allows you to upload a document image and detect stamps using one of two models:

- An algorithm based on traditional image processing techniques.
- A trained YOLO model.


```bash
streamlit run app.py
```

6. Run the automated labeling tool
This Streamlit tool labels images using the image-processing-based stamp detector and saves them in YOLO format.

```bash
streamlit run label_images_app.py
```

## Alternative Installation

1. Export dependencies from Poetry to requirements.txt

```bash
poetry export -f requirements.txt --output requirements.txt
```

2. Create and activate a virtual environment using Python

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the main application

```bash
streamlit run app.py
```

5. Run the automated labeling tool

```bash
streamlit run label_images_app.py
```

## How the Stamp Detection Algorithm Works (Classic Image Processing Techniques based)

The stamp detection algorithm leverages a sequence of image processing techniques to identify and isolate stamp-like objects within scanned documents or images. The process is designed to be robust to varying lighting conditions, noise, and different stamp shapes (circular, elliptical, or rectangular). Here's a step-by-step breakdown of how it works:

### 1. **Color Calibration**
The pipeline starts by calibrating the image colors using the **calibrate_colors** method:

- The image is converted to the LAB color space to better isolate luminance.

- Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to the luminance channel to enhance contrast.

- The image is converted back to BGR and undergoes color channel stretching to improve overall color balance and clarity.

### 2. **Edge Detection**

Using the **Canny edge detector**, the algorithm extracts sharp boundaries from the grayscale version of the image. A dynamic thresholding mechanism (based on the median intensity) ensures adaptability to various image contrasts.

### 3. **Contour Detection**

Contours are extracted from the edge map using OpenCV's contour detection method. These contours are drawn onto a binary mask to localize candidate stamp regions.

### 4. **Color-Based Segmentation**

To isolate actual stamps (often colored), the algorithm converts the image to HSV color space and filters out non-colored (grayscale or black/white) areas. This helps in discarding irrelevant background content.

### 5. **Mask Application**

The color-segmented image is used to create a binary mask that is applied to the original image. The non-stamp areas are replaced with a white background, visually highlighting the regions of interest.

### 6. **Merging Connected Components**

Morphological operations (specifically closing with an elliptical kernel) are used to merge fragmented components of stamps, making it easier to detect complete stamp shapes.

### 7. **Valid Contour Selection**

The **_get_valid_contours** method filters out noise by selecting only those contours that:

- Exceed a minimum area threshold.
- Match the expected geometric features of stamps that might be (circular, elliptical, or rectangular).

### 8. **Bounding Box Drawing**

Finally, the algorithm draws bounding boxes around the valid contours. This visual overlay clearly marks detected stamps and labels them with a class name **Stamp**.

## YOLO Model Training Pipeline

The YOLO-based model was trained using a carefully curated subset of images from the original Kaggle dataset.

Here's how the training process was carried out:

- We started with a dataset of **1,000 raw document images** from [Signatures and Stamps](https://www.kaggle.com/datasets/igorkarayman/signatures-and-stamps/data).
- To generate labeled data, we used the `label_images_app.py` Streamlit application, which automatically detects stamps using an image-processing-based algorithm.
- This automated labeling tool was run across the full dataset. From those, **300 high-quality images** were selected where the detections were very accurate.
- The selected 300 images were converted to **YOLO format** annotations.
- The dataset was then split into **80% training** and **20% validation** sets.
- We trained a `YOLOv8n` (YOLOv8 nano) model on this curated dataset.

### Training Results

After training, we evaluated the model using various metrics and visualizations:

1. **Confusion Matrix** – Shows true positives, false positives, etc.
![Confusion Matrix](./runs/detect/stamp_yolov8n/confusion_matrix.png)
2. **Training & Validation Metrics** – Includes precision, recall, mAP, and loss curves.
![Training & Validation Metrics](./runs/detect/stamp_yolov8n/results.png)
3. **Sample Predictions** – Visual samples of batch predictions on the validation set.
![Sample Predictions](./runs/detect/stamp_yolov8n/val_batch0_pred.jpg)

All training artifacts including model weights, logs, and evaluation plots are saved in the [`runs`](./runs) directory.
