import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from src.detector import StampDetector
import pandas as pd
import shutil


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
    """
    This function shows an image.

    Parameters
    ----------
    img : np.ndarray
        Image to show.
    
    showAxis : bool
        Whether to show the axis.
        
    size : Tuple[int, int]
        Size of the figure.
    
    Returns
    -------
    None
    """
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
    """
    This function returns the index of an image from a list of image names.

    Parameters
    ----------
    name : str
        Name of the image.
    
    image_names : List[str]
        List of image names.
    
    Returns
    -------
    index : int
        Index of the image.
    """
    return image_names.index(name)

def imread(
    path: str
) -> np.ndarray:
    """
    This function reads an image.

    Parameters
    ----------
    path : str
        Path to the image.
    
    Returns
    -------
    img : np.ndarray
        Image.
    """
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
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    img_width: float,
    img_height: float
) -> Tuple[float, float, float, float]:
    """
    This function converts the bounding boxes to the YOLO format.

    Parameters
    ----------
    x_min : float
        x_min of the bounding box.

    y_min : float
        y_min of the bounding box.

    x_max : float
        x_max of the bounding box.

    y_max : float
        y_max of the bounding box.
    
    img_width : float
        Width of the image.

    img_height : float
        Height of the image.
    
    Returns
    -------
    x_center : float
        x_center of the bounding box.

    y_center : float
        y_center of the bounding box.
    
    width : float
        Width of the bounding box.

    height : float
        Height of the bounding box.
    """
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

def save_yolo_label(
    file_path: str,
    boxes: List[Tuple[float, float, float, float]],
    img_width: float,
    img_height: float,
    class_id: int = 0
) -> None:
    """
    This function saves the YOLO labels.

    Parameters
    ----------
    file_path : str
        Path to the file.
    
    boxes : List[Tuple[float, float, float, float]]
        List of bounding boxes.

    img_width : float
        Width of the image.

    img_height : float
        Height of the image.

    class_id : int
        Class ID of the bounding box.
    
    Returns
    -------
    None
    """
    with open(file_path, 'w') as f:
        for (x_min, y_min, x_max, y_max) in boxes:
            x_c, y_c, w, h = convert_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height)
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

def get_bounding_boxes(
    detector: StampDetector,
    detected_img: np.ndarray
) -> List[Tuple[int, int, int, int]]:   
    """
    This function gets the bounding boxes from the detected image.

    Parameters
    ----------
    detector : StampDetector
        Detector object.
    
    detected_img : np.ndarray
        Detected image.
    
    Returns
    -------
    boxes : List[Tuple[int, int, int, int]]
        List of bounding boxes.
    """
    gray = cv2.cvtColor(detected_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    final_contours = detector._get_valid_contours(mask)
    
    boxes = []
    for cnt in final_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))
    return boxes

def rename_images(
    images: List[str],
    dataset_path: str
) -> List[str]:
    """
    This function renames the images.

    Parameters
    ----------
    images : List[str]
        List of image names.
    
    dataset_path : str
        Path to the dataset.
    
    Returns
    -------
    new_images : List[str]
        List of new image names.
    """
    if all(os.path.splitext(image)[0].isdigit() for image in images):
        return images
    
    new_images = []

    for i, image in enumerate(images):
        image_ext = os.path.splitext(image)[1]
        os.rename(os.path.join(dataset_path, image), os.path.join(dataset_path, f"{i}{image_ext}"))
        new_images.append(f"{i}{image_ext}")

    return new_images

def create_yolo_dataset(
    df: pd.DataFrame,
    dataset_path: str,
    yolo_dataset_path: str,
    split_type: str
) -> None:
    """
    This function creates the YOLO dataset to train the model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    
    yolo_dataset_path : str
        Path to the YOLO dataset.
    
    split_type : str
        Split type (train, val).
    
    Returns
    -------
    None
    """
    for _, row in df.iterrows():
        shutil.copy(
            os.path.join(dataset_path, row['images']), 
            os.path.join(yolo_dataset_path, split_type, 'images', row['images'])
        )
        shutil.copy(
            os.path.join(dataset_path, row['labels']),
            os.path.join(yolo_dataset_path, split_type, 'labels', row['labels'])
        )
