"""
Data readers for multispectral images and related files.
"""

import numpy as np
import torch
from sklearn import preprocessing


def read_raw(filename):
    """
    Read raw multispectral image data.
    
    Args:
        filename (str): Path to the raw file
        
    Returns:
        tuple: (image_data, image_height, image_width, num_channels)
    """
    (num_channels, image_height, image_width) = np.fromfile(filename, count=3, dtype=np.uint32)
    image_data = np.fromfile(filename, count=num_channels * image_height * image_width, offset=3 * 4, dtype=np.int32)
    
    print("* Read dataset:", filename)
    print("  Num channels (B):", num_channels, "Image height (H):", image_height, "Image width (V):", image_width)
    print("  Read:", len(image_data))

    # normalize data to [-1, 1]
    image_data = image_data.astype(np.float64)
    preprocessing.minmax_scale(image_data, feature_range=(-1, 1), copy=False)

    image_data = image_data.reshape(image_width, image_height, num_channels)
    image_data = torch.FloatTensor(image_data)
    return image_data, image_height, image_width, num_channels


def read_seg(filename):
    """
    Read segmentation data.
    
    Args:
        filename (str): Path to the segmentation file
        
    Returns:
        tuple: (segmentation_data, image_height, image_width)
    """
    (image_height, image_width) = np.fromfile(filename, count=2, dtype=np.uint32)
    segmentation_data = np.fromfile(filename, count=image_height * image_width, offset=2 * 4, dtype=np.uint32)
    
    print("* Read segmentation:", filename)
    print("  Image width (H):", image_height, "Image height (V):", image_width)
    print("  Read:", len(segmentation_data))
    
    return segmentation_data, image_height, image_width


def read_pgm(filename):
    """
    Read PGM (Portable Gray Map) ground truth file.
    
    Args:
        filename (str): Path to the PGM file
        
    Returns:
        tuple: (pixel_values, image_height, image_width)
    """
    try:
        pgmf = open(filename, "rb")
    except IOError:
        print("Cannot open", filename)
        raise
    else:
        assert pgmf.readline().decode() == "P5\n"
        line = pgmf.readline().decode()
        while line[0] == "#":
            line = pgmf.readline().decode()
        (image_height, image_width) = line.split()
        image_height = int(image_height)
        image_width = int(image_width)
        num_classes = int(pgmf.readline().decode())
        assert num_classes <= 255
        
        pixel_values = []
        for i in range(image_height * image_width):
            pixel_values.append(ord(pgmf.read(1)))
            
        print("* Read GT:", filename)
        print("  Image height (H):", image_height, "Image width (V):", image_width, "depth:", num_classes)
        print("  Read:", len(pixel_values))
        
        return pixel_values, image_height, image_width


def read_seg_centers(filename):
    """
    Read segment centers data.
    
    Args:
        filename (str): Path to the segment centers file
        
    Returns:
        tuple: (segment_centers, image_height, image_width, nseg)
    """
    (image_height, image_width, nseg) = np.fromfile(filename, count=3, dtype=np.uint32)
    segment_centers = np.fromfile(filename, count=image_height * image_width, offset=3 * 4, dtype=np.uint32)
    
    print("* Read centers:", filename)
    print("  Image width (H):", image_height, "Image height (V):", image_width, "Number of segments (nseg):", nseg)
    print("  Read:", len(segment_centers))
    
    return segment_centers, image_height, image_width, nseg


def load_multispectral_dataset(input_dir, filename, read_raw_data=True):
    """
    Load a complete multispectral dataset (raw data, ground truth, and segment centers).
    
    Args:
        input_dir (str): Directory containing the dataset files
        filename (str): Base filename without extension
        
    Returns:
        dict: Dictionary containing all loaded data with keys:
            - 'data': Multispectral image data
            - 'truth': Ground truth labels
            - 'centers': Segment centers
            - 'dimensions': Image dimensions
            - 'metadata': Additional metadata
    """
    import os
    
    # Construct file paths
    raw_file = os.path.join(input_dir, f"{filename}.raw")
    gt_file = os.path.join(input_dir, f"{filename}.pgm")
    centers_file = os.path.join(input_dir, f"seg_{filename}_wp_centers.raw")
    seg_file = os.path.join(input_dir, f"seg_{filename}_wp.raw")
    
    # Read all data
    if read_raw_data:
        data, image_height, image_width, num_channels = read_raw(raw_file)
    truth, gt_height, gt_width = read_pgm(gt_file)
    if not read_raw_data:
        data = None
        image_height, image_width = gt_height, gt_width
        num_channels = 5  # only is used when evaluating so, we can hardcode 5 channels
    centers, centers_height, centers_width, nseg = read_seg_centers(centers_file)
    segmentation_data, segmentation_height, segmentation_width = read_seg(seg_file)
    
    # Validate dimensions
    assert image_height == gt_height == segmentation_height and image_width == gt_width == segmentation_width, \
        f"Data and GT dimensions do not match: {image_height}x{image_width} vs {gt_height}x{gt_width}"
    
    # Transpose data to band-vector format for convolutions
    if data is not None:
        data = np.transpose(data, (2, 0, 1))
    
    return {
        'data': data,
        'truth': truth,
        'centers': centers,
        'segmentation_data': segmentation_data,
        'dimensions': {
            'height': image_height,
            'width': image_width,
            'channels': num_channels
        },
        'metadata': {
            'nseg': nseg,
            'filename': filename
        }
    }