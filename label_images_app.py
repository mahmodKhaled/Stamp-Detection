import os
import cv2
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib import patches
from src.detector import StampDetector
from src.utils import get_path_to, imread, get_bounding_boxes, save_yolo_label, rename_images


def detector_pipeline(
    detector: StampDetector,
    image: np.ndarray,
) -> np.ndarray:
    """
    This function is the pipeline for the stamp detection.

    applies the following operations in the following order:
    1. Calibrate colors
    2. Detect edges
    3. Detect contours
    4. Apply mask
    5. Segment
    6. Apply mask
    7. Merge connected components
    8. Apply mask

    Parameters
    ----------
    detector : StampDetector
        The detector object.

    image : np.ndarray
        The image to detect stamps in.
    
    Returns
    -------
    merged_masked_image : np.ndarray
        The image with the detected stamps.
    """
    calibrated_image = detector.calibrate_colors(image)
    edges_image = detector.detect_edges(calibrated_image)
    contours_image = detector.detect_contours(edges_image, calibrated_image.shape)
    contours_masked_image = detector.apply_mask(calibrated_image, contours_image)
    segmented_image = detector.segment(contours_masked_image)
    segmented_masked_image = detector.apply_mask(calibrated_image, segmented_image)
    merged_image = detector.merge_connected_components(segmented_masked_image)
    merged_masked_image = detector.apply_mask(calibrated_image, merged_image)
    
    return merged_masked_image

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
    images = rename_images(images, dataset_path)

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