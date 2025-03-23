import cv2
import numpy as np
from typing import List, Tuple

class StampDetector:
    def __init__(
        self,
    ) -> None:
        pass
    
    def detect_edges(
        self,
        image: np.ndarray,
        sigma: float = 0.33
    ) -> np.ndarray:
        # Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find edges using canny edge detector
        v = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(gray, lower, upper)

        return edged
    
    def detect_contours(
        self,
        image: np.ndarray,
        image_shape: tuple
    ) -> np.ndarray:
        # Find contours from the edged image
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank mask with same height and width as the original image
        mask = np.zeros(image_shape[:2], dtype=np.uint8)  # grayscale mask

        # Draw contours on the mask - filled
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

        # Convert single-channel mask to 3-channel for apply_mask
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return mask_3channel

    def segment(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define HSV thresholds to isolate non-B/W (i.e., colored regions)
        lower_color = np.array([0, 55, 55])
        upper_color = np.array([180, 255, 255])
        color_mask = cv2.inRange(hsv, lower_color, upper_color)

        # Apply mask to extract the colored regions
        colored_only = cv2.bitwise_and(image, image, mask=color_mask)
        return colored_only

    def apply_mask(
        self,
        original_img: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        # Convert segmented image to grayscale to create a binary mask
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Create binary mask where non-black pixels are 255 (white)
        _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Create inverse mask for background
        inverse_mask = cv2.bitwise_not(binary_mask)

        # Extract foreground from original image
        foreground = cv2.bitwise_and(original_img, original_img, mask=binary_mask)

        # Create white background
        background = np.full(original_img.shape, 255, dtype=np.uint8)
        background = cv2.bitwise_and(background, background, mask=inverse_mask)

        # Combine foreground and background
        masked_output = cv2.add(foreground, background) 
        return masked_output
    
    def merge_connected_components(
        self,
        segmented_img: np.ndarray,
        kernel_size: int = 25
    ) -> np.ndarray:
        # Convert segmented image to grayscale
        gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

        # Invert and threshold: segmented objects become white on black
        _, binary_mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

        # Create elliptical structuring element (more natural for rounded stamps)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Apply morphological closing
        merged_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Convert to color image
        merged_mask_colored = cv2.cvtColor(merged_mask, cv2.COLOR_GRAY2BGR) 

        return merged_mask_colored

    def _get_valid_contours(
        self,
        mask: np.ndarray,
        min_area: int = 1000
    ) -> List[np.ndarray]:
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # identify contours that are follow the same shape as circle, ellipse or rectangle
        final_contours = []
        for cnt in contours:
            # calculate the area of the contour
            area = cv2.contourArea(cnt)
            if area > min_area:
                final_contours.append(cnt)
        
        return final_contours

    def draw_bounding_boxes(
        self,
        original_img: np.ndarray,
        segmented_img: np.ndarray,
        class_name: str = "Stamp",
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        # Convert segmented image to grayscale
        gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

        # Create binary mask: anything not white is assumed to be part of the object
        _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

        final_contours = self._get_valid_contours(mask)

        # Copy original image to draw on
        output_img = original_img.copy()

        for cnt in final_contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw bounding box
            cv2.rectangle(output_img, (x, y), (x + w, y + h), color, thickness)

            # Label the box
            label_pos = (x, y - 10 if y - 10 > 10 else y + 10)
            cv2.putText(output_img, class_name, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return output_img
