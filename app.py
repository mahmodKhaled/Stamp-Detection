import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
from src.detector import StampDetector
from src.utils import get_path_to

MODEL_PATH = get_path_to('runs', 'detect', 'stamp_yolov8n', 'weights', 'best.pt')

def detctor_pipeline(
    image: np.ndarray
) -> np.ndarray:
    detector = StampDetector()

    calibrated_image = detector.calibrate_colors(image)
    edges_image = detector.detect_edges(calibrated_image)
    contours_image = detector.detect_contours(edges_image, calibrated_image.shape)
    contours_masked_image = detector.apply_mask(calibrated_image, contours_image)
    segmented_image = detector.segment(contours_masked_image)
    segmented_masked_image = detector.apply_mask(calibrated_image, segmented_image)
    merged_image = detector.merge_connected_components(segmented_masked_image)
    merged_masked_image = detector.apply_mask(calibrated_image, merged_image)
    detected_image = detector.draw_bounding_boxes(image, merged_masked_image)
    
    return detected_image

def yolo_detector(
    image: np.ndarray,
    model_path: str
) -> Image.Image:
    model = YOLO(model_path)
    results = model(image)
    result_img = results[0].plot()
    return Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

def main():
    st.title("Stamp Detection")
    st.write("Upload an image to detect stamps")
    
    detection_method = st.selectbox(
        "Select Detection Method",
        ["YOLOv8", "Image Processing"]
    )
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if st.button("Detect Stamps"):
        if image is not None:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.write("Detecting stamps...")
            
            if detection_method == "YOLOv8":
                detected_image = yolo_detector(image, MODEL_PATH)
            else:
                detected_image = detctor_pipeline(image)
            
            st.image(detected_image, caption="Detected Stamps", use_container_width=True)

if __name__ == "__main__":
    main()