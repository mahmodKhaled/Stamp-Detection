import cv2
import numpy as np
from typing import List, Tuple

class StampDetector:
    def __init__(
        self,
    ) -> None:
        pass

    def calibrate_colors(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        This function calibrates the colors of the image to enhance the quality of the image.

        Parameters
        ----------
        image : np.ndarray
            The image to calibrate.
        
        Returns
        -------
        balanced : np.ndarray
            The calibrated image.
        """
        # Step 1: Convert to LAB color space for better light adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Step 2: Apply CLAHE to L-channel (contrast limited adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)

        # Step 3: Merge and convert back to BGR
        lab_eq = cv2.merge((l_eq, a, b))
        corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # Step 4: Apply color balance to stretch BGR channels
        balanced = self._stretch_color_channels(corrected)

        return balanced

    def _stretch_color_channels(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        This function stretches the color channels of the image to enhance the quality of the image.

        Parameters
        ----------
        image : np.ndarray
            The image to stretch.
        
        Returns
        -------
        out : np.ndarray
            The stretched image.
        """
        out = np.zeros_like(image)
        for i in range(3):  # For B, G, R
            channel = image[:, :, i]
            min_val = np.percentile(channel, 1)
            max_val = np.percentile(channel, 99)

            # Avoid division by zero
            if max_val - min_val > 0:
                channel_stretched = np.clip((channel - min_val) * 255.0 / (max_val - min_val), 0, 255)
            else:
                channel_stretched = channel
            out[:, :, i] = channel_stretched.astype(np.uint8)
        return out

    def detect_edges(
        self,
        image: np.ndarray,
        sigma: float = 0.33
    ) -> np.ndarray:
        """
        This function detects the edges of the image using the canny edge detector.

        Parameters
        ----------
        image : np.ndarray
            The image to detect edges in.
        
        sigma : float
            The sigma value for the canny edge detector.
        
        Returns
        -------
        edged : np.ndarray
            The edges of the image.
        """
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
        """
        This function detects the contours of the image by taking the edged image.

        Parameters
        ----------
        image : np.ndarray
            The image to detect contours in.
        
        image_shape : tuple
            The shape of the image.
        
        Returns
        -------
        mask_3channel : np.ndarray
            The mask of the image.
        """
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
        """
        This function segments the image by isolating the colored regions.

        Parameters
        ----------
        image : np.ndarray
            The image to segment.
        
        Returns
        -------
        colored_only : np.ndarray
            The segmented image.
        """
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
        """
        This function acts as a filter to the image by applying a mask to it.

        Parameters
        ----------
        original_img : np.ndarray
            The image to apply the mask to.
        
        mask : np.ndarray
            The mask to apply to the image.
        
        Returns
        -------
        masked_output : np.ndarray
            The masked image.
        """
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
        """
        This function merges the connected components of the image by using a morphological closing,
        to close the area inside the edges of the stamps.

        Parameters
        ----------
        segmented_img : np.ndarray
            The segmented image.
        
        kernel_size : int
            The size of the kernel to use for the morphological closing.
        
        Returns
        -------
        merged_mask_colored : np.ndarray
            The merged mask of the image.
        """
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
        """
        This function identifies the contours of the image that are valid stamps which are following:
        - circular
        - elliptical
        - rectangular

        Parameters
        ----------
        mask : np.ndarray
            The mask of the image.

        min_area : int
            The minimum area of the contour to be considered a valid stamp.
        
        Returns
        -------
        final_contours : List[np.ndarray]
            The contours of the image that are valid stamps.
        """
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
        """
        This function draws the bounding boxes of the detected stamps in the image.

        Parameters
        ----------
        original_img : np.ndarray
            The original image.
        
        segmented_img : np.ndarray
            The segmented image.
        
        class_name : str
            The name of the class.
        
        color : Tuple[int, int, int]
            The color of the bounding box.
        
        thickness : int
            The thickness of the bounding box.
        
        Returns
        -------
        output_img : np.ndarray
            The image with the bounding boxes drawn on it of the detected stamps.
        """
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
