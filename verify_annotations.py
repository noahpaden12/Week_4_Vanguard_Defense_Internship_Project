import os
import json
import sys
from pathlib import Path

def verify_coco_annotations(annotations_path):
    """
    Verify that the COCO annotations file exists and check its basic structure.
    """
    # Check if file exists
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found at {annotations_path}")
        return False

    # Load and check annotations file
    try:
        with open(annotations_path, 'r') as f:
            coco = json.load(f)
            
        # Check if the required fields exist
        if 'images' not in coco:
            print(f"Error: 'images' field not found in {annotations_path}")
            return False
            
        if 'annotations' not in coco:
            print(f"Error: 'annotations' field not found in {annotations_path}")
            return False
            
        if 'categories' not in coco:
            print(f"Error: 'categories' field not found in {annotations_path}")
            return False
            
        # Check if there are any images and annotations
        num_images = len(coco['images'])
        num_annotations = len(coco['annotations'])
        
        if num_images == 0:
            print(f"Warning: No images found in {annotations_path}")
            
        if num_annotations == 0:
            print(f"Warning: No annotations found in {annotations_path}")
        
        print(f"Found {num_images} images and {num_annotations} annotations in the COCO file.")
        
        # Check the bbox format in the annotations
        bbox_issues = 0
        for i, ann in enumerate(coco['annotations'][:5]):  # Check the first 5 annotations
            if 'bbox' not in ann:
                print(f"Warning: Annotation {i} is missing 'bbox' field")
                bbox_issues += 1
                continue
                
            # Check the format of the bbox
            bbox = ann['bbox']
            print(f"Annotation {i}: bbox = {bbox}")
            
            if isinstance(bbox, list) and len(bbox) < 4:
                print(f"Warning: Annotation {i} has incomplete bbox: {bbox}")
                bbox_issues += 1
        
        if bbox_issues > 0:
            print(f"\nFound {bbox_issues} annotations with bbox format issues.")
            print("The standard COCO format requires bbox to be a list with 4 values: [x, y, width, height]")
            return False
        
        return True
        
    except json.JSONDecodeError:
        print(f"Error: {annotations_path} is not a valid JSON file")
        return False
    except Exception as e:
        print(f"Error: Failed to process {annotations_path}: {str(e)}")
        return False

def check_yolo_format_directories():
    """
    Check if YOLO format directories exist and contain files.
    """
    splits = ['train', 'val', 'test']
    all_exist = True
    
    for split in splits:
        labels_dir = Path(f'yolo_format/{split}/labels')
        images_dir = Path(f'yolo_format/{split}/images')
        
        if not labels_dir.exists():
            print(f"YOLO format directory not found: {labels_dir}")
            all_exist = False
        else:
            label_count = len(list(labels_dir.glob('*.txt')))
            print(f"Found {label_count} label files in {labels_dir}")
            
        if not images_dir.exists():
            print(f"YOLO format directory not found: {images_dir}")
            all_exist = False
        else:
            image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
            print(f"Found {image_count} image files in {images_dir}")
            
    return all_exist

def main():
    # Path to the annotations file
    annotations_path = 'annotations/yolov8_annotations.json'
    
    # Verify COCO annotations
    if verify_coco_annotations(annotations_path):
        # Check YOLO format directories
        if check_yolo_format_directories():
            print("\nAll checks passed. Your YOLO format directories appear to be set up correctly.")
        else:
            print("\nYOLO format directories are missing or empty.")
            print("To convert COCO JSON to YOLO format, add the following to your model_trainer.py:")
            print_conversion_function()
    else:
        print("\nTo convert COCO JSON to YOLO format, add the following to your model_trainer.py:")
        print_conversion_function()

def print_conversion_function():
    """Print the convert_coco_to_yolo function."""
    print("""
import json
from pathlib import Path
import os
import shutil

def convert_coco_to_yolo():
    # Load annotations
    with open('annotations/yolov8_annotations.json', 'r') as f:
        coco = json.load(f)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f'yolo_format/{split}/labels', exist_ok=True)
        os.makedirs(f'yolo_format/{split}/images', exist_ok=True)
    
    # Create image id to filename mapping
    img_id_to_name = {img['id']: img['file_name'] for img in coco['images']}
    img_id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}
    
    # Process annotations
    for ann in coco['annotations']:
        if 'bbox' not in ann or len(ann['bbox']) < 4:
            continue
            
        img_id = ann['image_id']
        img_name = img_id_to_name[img_id]
        img_w, img_h = img_id_to_size[img_id]
        
        # Convert bbox [x, y, width, height] to YOLO format [x_center/img_w, y_center/img_h, width/img_w, height/img_h]
        x, y, w, h = ann['bbox']
        x_center = (x + w/2) / img_w
        y_center = (y + h/2) / img_h
        w = w / img_w
        h = h / img_h
        
        category_id = ann['category_id']
        
        # Split data: 80% train, 10% val, 10% test based on image_id
        if img_id % 10 < 8:
            split = 'train'
        elif img_id % 10 == 8:
            split = 'val'
        else:
            split = 'test'
        
        # Write to label file
        with open(f'yolo_format/{split}/labels/{os.path.splitext(img_name)[0]}.txt', 'a') as f:
            f.write(f"{category_id} {x_center} {y_center} {w} {h}\\n")
        
        # Copy image file if it doesn't exist
        src_img = f'xview_sample/{img_name}'
        dst_img = f'yolo_format/{split}/images/{img_name}'
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            shutil.copy(src_img, dst_img)

# Call this function before training
convert_coco_to_yolo()
""")

if __name__ == "__main__":
    main()
