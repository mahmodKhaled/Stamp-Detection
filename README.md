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






