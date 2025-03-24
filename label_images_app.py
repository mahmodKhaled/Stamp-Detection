import os
import sys
import cv2
import numpy as np
from typing import List, Tuple
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib import patches
sys.path.append('../')
from src.detector import StampDetector
from src.utils import get_path_to, imread

def convert_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

def save_yolo_label(file_path, boxes, img_width, img_height, class_id=0):
    with open(file_path, 'w') as f:
        for (x_min, y_min, x_max, y_max) in boxes:
            x_c, y_c, w, h = convert_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height)
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

def detector_pipeline(
    detector: StampDetector,
    image: np.ndarray,
) -> np.ndarray:
    calibrated_image = detector.calibrate_colors(image)
    edges_image = detector.detect_edges(calibrated_image)
    contours_image = detector.detect_contours(edges_image, calibrated_image.shape)
    contours_masked_image = detector.apply_mask(calibrated_image, contours_image)
    segmented_image = detector.segment(contours_masked_image)
    segmented_masked_image = detector.apply_mask(calibrated_image, segmented_image)
    merged_image = detector.merge_connected_components(segmented_masked_image)
    merged_masked_image = detector.apply_mask(calibrated_image, merged_image)
    
    return merged_masked_image

def get_bounding_boxes(
    detector: StampDetector,
    detected_img: np.ndarray
) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(detected_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    final_contours = detector._get_valid_contours(mask)
    
    boxes = []
    for cnt in final_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))
    return boxes

def main():
    st.set_page_config(page_title="Stamp Image Labeling Tool", layout="wide")
    st.title("Stamp Image Labeling Tool")

    detector = StampDetector()
    
    # Setup paths
    dataset_path = get_path_to('input', 'datasets', 'igorkarayman', 'signatures-and-stamps', 'versions', '1', '1')
    labels_path = get_path_to('input', 'labeled_images')
    os.makedirs(labels_path, exist_ok=True)
    
    # Get list of images
    images = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Initialize counter in session state if not exists
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
        
    # Navigation and action buttons in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Previous") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()

    with col3:
        if st.button("Next") and st.session_state.current_index < len(images) - 1:
            st.session_state.current_index += 1
            st.rerun()
            
    # Display counter and current image name
    st.write(f"Image {st.session_state.current_index} of {len(images)-1}")
    current_image_name = images[st.session_state.current_index]
    st.write(f"Current image: {current_image_name}")
    
    # Load and process current image
    current_image = imread(os.path.join(dataset_path, current_image_name))
    detected_img = detector_pipeline(detector, current_image)
    boxes = get_bounding_boxes(detector, detected_img)
    
    # Display image with bounding boxes
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(current_image)
    for box in boxes:
        x, y, w, h = box
        # Ensure coordinates are within image bounds
        x = min(x, current_image.shape[1])
        y = min(y, current_image.shape[0]) 
        w = min(w - x, current_image.shape[1] - x)
        h = min(h - y, current_image.shape[0] - y)
        rect = patches.Rectangle((x, y), w, h, linewidth=0.5, edgecolor='green', facecolor='none')
        ax.text(x, y, 'Stamp', color='green', fontsize=2)
        ax.add_patch(rect)
    plt.axis('off')
    st.pyplot(fig, use_container_width=False)

    with col2:
        if st.button("Accept Boxes"):
            # Save the labels
            label_file = os.path.join(labels_path, os.path.splitext(current_image_name)[0] + '.txt')
            # save the label file
            save_yolo_label(label_file, boxes, current_image.shape[1], current_image.shape[0])
            # save the image with the boxes
            cv2.imwrite(
                os.path.join(labels_path, os.path.splitext(current_image_name)[0] + '.png'),
                detector.draw_bounding_boxes(current_image, detected_img)
            )
            st.success("Labels saved successfully!")

if __name__ == "__main__":
    main()