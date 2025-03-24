import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from src.detector import StampDetector

def get_path_to(
    *args,
    repo_name: str = 'Stamp-Detection',
) -> str:
    """
    This function returns the path to any file or directory.

    Parameters
    ----------
    repo_name : str
        Name of the repository.
    *args : str
        Variable number of path components (folders/files) to join.
    
    Returns
    -------
    path : str
        Full path to the requested location.
    """
    base_dir = os.path.dirname(os.getcwd())
    if repo_name not in base_dir:
        base_dir = os.path.join(base_dir, repo_name)
        
    path = os.path.join(base_dir, *args)
    
    # Create parent directory if it doesn't exist
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    return path

def show_image(
    img: np.ndarray,
    showAxis: bool = False,
    size: Tuple[int, int] = (20, 10)
) -> None:
    plt.figure(figsize=size)
    if not showAxis:
        plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img[:,:,::-1])
    else:
        plt.imshow(img, cmap='gray')

def get_index_of_image_from_name(
    name: str,
    image_names: List[str]
) -> int:
    return image_names.index(name)

def imread(
    path: str
) -> np.ndarray:
    img = plt.imread(path)

    # If image is float convert to uint8
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    # Handle alpha channel if present
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    # Convert RGB (matplotlib default) to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def convert_to_yolo_format(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

def save_yolo_label(
    file_path: str,
    boxes: List[Tuple[int, int, int, int]],
    img_width: int,
    img_height: int,
    class_id: int = 0
) -> None:
    with open(file_path, 'w') as f:
        for (x_min, y_min, x_max, y_max) in boxes:
            x_c, y_c, w, h = convert_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height)
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

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
