import streamlit as st
import cv2
import numpy as np
from src.detector import StampDetector

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

def main():
    st.title("Stamp Detection")
    st.write("Upload an image to detect stamps")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if st.button("Detect Stamps"):
        if image is not None:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.write("Detecting stamps...")
            detected_image = detctor_pipeline(image)
            st.image(detected_image, caption="Detected Stamps", use_container_width=True)

if __name__ == "__main__":
    main()