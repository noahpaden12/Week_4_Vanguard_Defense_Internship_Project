import json
from pathlib import Path
import os
import shutil
import yaml
import random
from ultralytics import YOLO
import csv
import matplotlib.pyplot as plt

def split_data(images_dir, labels_dir, train_ratio=0.8):
    """Split data into training and validation sets."""
    # Create output directories if they don't exist
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(images_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split, "labels"), exist_ok=True)
    
    # Check if images are already in yolo_format directory structure
    yolo_dir = Path("yolo_format")
    if yolo_dir.exists():
        train_count = len(list((yolo_dir / "train" / "images").glob("*.jpg")))
        val_count = len(list((yolo_dir / "val" / "images").glob("*.jpg")))
        test_count = len(list((yolo_dir / "test" / "images").glob("*.jpg")))
        
        if train_count > 0 or val_count > 0 or test_count > 0:
            print(f"Found existing data in yolo_format: {train_count} training images, {val_count} validation images, and {test_count} test images.")
            return train_count, val_count
    
    # Count images using correct path handling
    images_path = Path(images_dir)
    train_count = len(list((images_path / "train" / "images").glob("*.jpg")))
    val_count = len(list((images_path / "val" / "images").glob("*.jpg")))
    test_count = len(list((images_path / "test" / "images").glob("*.jpg")))
    
    total = train_count + val_count + test_count
    
    print(f"Found {train_count} training images, {val_count} validation images, and {test_count} test images in {images_dir}.")
    
    # Check if we need to copy images from source to split directories
    source_images = []
    
    # First try the flat xview_sample directory
    source_images_dir = Path("xview_sample")
    if source_images_dir.exists():
        source_images = list(source_images_dir.glob("*.jpg"))
        if source_images:
            print(f"Found {len(source_images)} images in {source_images_dir}")
    
    # If no images found in xview_sample, check if annotations exist directly
    if not source_images:
        script_dir = Path(__file__).parent.absolute()
        annotations_path = script_dir / 'annotations' / 'manual_annotations.json'
        
        if annotations_path.exists():
            try:
                with open(annotations_path, 'r') as f:
                    coco = json.load(f)
                
                image_filenames = [img['file_name'] for img in coco['images']]
                print(f"Found {len(image_filenames)} image references in annotations.")
                
                # Look for these images in the current directory or other common places
                for img_name in image_filenames[:5]:  # Just check a few as example
                    possible_paths = [
                        Path(img_name),
                        Path("images") / img_name,
                        Path("data") / img_name
                    ]
                    for path in possible_paths:
                        if path.exists():
                            print(f"Example image found at: {path}")
                            break
                    else:
                        print(f"Example image not found: {img_name}")
            except Exception as e:
                print(f"Error reading annotations: {e}")
    
    if total == 0 and source_images:
        print("No images found in split directories. Copying from source...")
        # Shuffle for random split
        random.shuffle(source_images)
        
        # Split based on ratio
        split_idx = int(len(source_images) * train_ratio)
        train_images = source_images[:split_idx]
        val_images = source_images[split_idx:]
        
        # Copy to train directory
        for img_path in train_images:
            dst_path = images_path / "train" / "images" / img_path.name
            shutil.copy(img_path, dst_path)
            # Copy corresponding label if it exists
            label_path = Path(labels_dir) / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label = Path(labels_dir) / "train" / "labels" / f"{img_path.stem}.txt"
                shutil.copy(label_path, dst_label)
        
        # Copy to val directory
        for img_path in val_images:
            dst_path = images_path / "val" / "images" / img_path.name
            shutil.copy(img_path, dst_path)
            # Copy corresponding label if it exists
            label_path = Path(labels_dir) / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label = Path(labels_dir) / "val" / "labels" / f"{img_path.stem}.txt"
                shutil.copy(label_path, dst_label)
                
        # Count again after copying
        train_count = len(list((images_path / "train" / "images").glob("*.jpg")))
        val_count = len(list((images_path / "val" / "images").glob("*.jpg")))
        
        print(f"After copying: {train_count} training images and {val_count} validation images.")
    elif total == 0:
        print("No source images found to copy. Make sure your images are in the correct location.")
        print("Checking for files in the yolo_format directory instead...")
        
        # Check if files were already processed directly to yolo_format by the conversion function
        yolo_train_count = len(list(Path("yolo_format/train/images").glob("*.jpg")))
        yolo_val_count = len(list(Path("yolo_format/val/images").glob("*.jpg")))
        
        if yolo_train_count > 0 or yolo_val_count > 0:
            print(f"Found {yolo_train_count} training and {yolo_val_count} validation images in yolo_format directory.")
            train_count = yolo_train_count
            val_count = yolo_val_count
    
    return train_count, val_count

def convert_coco_to_yolo():
    """Convert COCO format annotations to YOLO format."""
    print("Converting COCO annotations to YOLO format...")
    
    # Get the directory containing the script
    script_dir = Path(__file__).parent.absolute()
    
    # Define annotation paths
    yolo_annotations_path = script_dir / 'annotations' / 'yolov8_annotations.json'
    manual_annotations_path = script_dir / 'annotations' / 'manual_annotations.json'
    
    # Check which annotation file exists
    if yolo_annotations_path.exists():
        annotations_path = yolo_annotations_path
    elif manual_annotations_path.exists():
        annotations_path = manual_annotations_path
        print(f"Using manual annotations file at {manual_annotations_path}")
    else:
        print(f"Error: Annotations file not found at {yolo_annotations_path} or {manual_annotations_path}")
        print("Please make sure the annotations file exists before running conversion.")
        return False
    
    try:
        # Load annotations
        with open(annotations_path, 'r') as f:
            coco = json.load(f)
        
        # Create directories
        for split in ['train', 'val', 'test']:
            os.makedirs(script_dir / 'yolo_format' / split / 'labels', exist_ok=True)
            os.makedirs(script_dir / 'yolo_format' / split / 'images', exist_ok=True)
        
        # Create image id to filename mapping
        img_id_to_name = {img['id']: img['file_name'] for img in coco['images']}
        img_id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}
        
        print(f"Found {len(img_id_to_name)} images referenced in annotations.")
        
        # Create category mapping to ensure zero-indexed classes
        category_id_map = {}
        if 'categories' in coco:
            categories = sorted(coco['categories'], key=lambda x: x.get('id', 0) if isinstance(x, dict) else 0)
            for i, cat in enumerate(categories):
                if isinstance(cat, dict) and 'id' in cat:
                    category_id_map[cat['id']] = i
        
        # Process annotations
        annotation_count = 0
        images_copied = 0
        images_already_exist = 0
        
        # Look for source images in multiple possible locations
        possible_source_dirs = [
            script_dir / 'xview_sample',
            script_dir / 'images',
            script_dir / 'data',
            script_dir,
            Path('xview_sample'),
            Path('images'),
            Path('data')
        ]
        
        # Check which directories exist and report
        existing_dirs = [d for d in possible_source_dirs if d.exists() and d.is_dir()]
        print(f"Searching for images in these directories: {', '.join(str(d) for d in existing_dirs)}")
        
        # Create a lookup of all image files across all source directories
        image_lookup = {}
        for source_dir in existing_dirs:
            for img_file in source_dir.glob('*.jpg'):
                image_lookup[img_file.name] = img_file
        
        print(f"Found {len(image_lookup)} image files in source directories.")
        
        for ann in coco['annotations']:
            if 'bbox' not in ann or not isinstance(ann['bbox'], list) or len(ann['bbox']) < 4:
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
            
            # Use the mapped category_id (zero-indexed)
            if 'category_id' in ann:
                original_category_id = ann['category_id']
                # Map to zero-indexed ID or use as-is if mapping doesn't exist
                category_id = category_id_map.get(original_category_id, original_category_id)
            else:
                category_id = 0  # Default to first class if no category_id
            
            # Split data: 80% train, 10% val, 10% test based on image_id
            if img_id % 10 < 8:
                split = 'train'
            elif img_id % 10 == 8:
                split = 'val'
            else:
                split = 'test'
            
            # Write to label file
            with open(script_dir / f'yolo_format/{split}/labels/{os.path.splitext(img_name)[0]}.txt', 'a') as f:
                f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")
            
            # Copy image file if it doesn't exist
            dst_img = script_dir / f'yolo_format/{split}/images/{img_name}'
            if os.path.exists(dst_img):
                images_already_exist += 1
            elif not os.path.exists(dst_img):
                # Try to find the image in our lookup
                if img_name in image_lookup:
                    src_img = image_lookup[img_name]
                    shutil.copy(src_img, dst_img)
                    images_copied += 1
            
            annotation_count += 1
            
            # Print progress every 100 annotations
            if annotation_count % 100 == 0:
                print(f"Processed {annotation_count} annotations, copied {images_copied} images, {images_already_exist} already exist...")
        
        # Count how many images were successfully copied to each split
        train_count = len(list((script_dir / 'yolo_format/train/images').glob('*.jpg')))
        val_count = len(list((script_dir / 'yolo_format/val/images').glob('*.jpg')))
        test_count = len(list((script_dir / 'yolo_format/test/images').glob('*.jpg')))
        
        print(f"Conversion complete. Processed {annotation_count} annotations.")
        print(f"Images in yolo_format: {train_count} training, {val_count} validation, {test_count} test images.")
        print(f"Images copied: {images_copied}, Images already existed: {images_already_exist}")
        
        if train_count + val_count + test_count == 0:
            print("\nWARNING: No images were copied to yolo_format directory!")
            print("Please check these potential issues:")
            print("1. Verify image files exist in one of the searched directories")
            print("2. Ensure the image filenames in the annotation file match the actual files")
            print("3. Check file permissions to ensure the script can read the source images and write to the destination")
            
            # Print the first few image filenames from annotations for debugging
            print("\nFirst 5 image filenames from annotations:")
            for i, name in enumerate(list(img_id_to_name.values())[:5]):
                print(f"  {i+1}. {name}")
                
            # Try to find these images in the source directories
            print("\nAttempting to locate these images:")
            for name in list(img_id_to_name.values())[:5]:
                found = False
                for source_dir in existing_dirs:
                    possible_path = source_dir / name
                    if possible_path.exists():
                        print(f"  Found '{name}' at: {possible_path}")
                        found = True
                        break
                if not found:
                    print(f"  Could not find '{name}' in any of the source directories.")
        
        return train_count + val_count + test_count > 0
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_dataset_yaml_from_categories(json_file, dataset_root, images_folder="yolo_format"):
    """Create dataset.yaml file for YOLOv8 training."""
    # Get the directory containing the script
    script_dir = Path(__file__).parent.absolute()
    
    # Try to load categories from the annotation file
    try:
        if not Path(json_file).is_absolute():
            json_file = script_dir / json_file
            
        with open(json_file) as f:
            data = json.load(f)
            
        # Fix for the categories in the JSON
        if 'categories' in data:
            # Sort categories by ID to ensure proper ordering
            categories = sorted(data["categories"], key=lambda x: x.get("id", 0) if isinstance(x, dict) else 0)
            
            # Create a properly indexed dictionary starting from 0
            names = {}
            for i, cat in enumerate(categories):
                # Handle both formats of category data
                if isinstance(cat, dict):
                    if 'name' in cat:
                        names[i] = cat['name']
                    elif 'supercategory' in cat:
                        names[i] = cat['supercategory']
                else:
                    # Fallback name if category is not a dict
                    names[i] = f"class_{i}"
        else:
            # Default class names if no categories in JSON
            names = {
                0: 'aircraft',
                1: 'building',
                2: 'vehicle',
                3: 'boat',
                4: 'tank',
                5: 'missile',
                6: 'rocket',
                7: 'artillery',
                8: 'helicopter'
            }
    except Exception as e:
        print(f"Error loading categories from {json_file}: {e}")
        # Default class names as fallback
        names = {
            0: 'aircraft',
            1: 'building',
            2: 'vehicle',
            3: 'boat',
            4: 'tank',
            5: 'missile',
            6: 'rocket',
            7: 'artillery',
            8: 'helicopter'
        }
    
    # Create dataset.yaml with absolute paths
    absolute_path = Path(dataset_root).absolute()
    dataset = {
        "path": str(absolute_path),
        "train": "yolo_format/train/images",  # Use direct path to images folder
        "val": "yolo_format/val/images",      # Use direct path to images folder
        "test": "yolo_format/test/images",    # Use direct path to images folder
        "names": names
    }
    
    yaml_path = Path("dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset, f, sort_keys=False)
    
    print(f"Dataset YAML saved to {yaml_path}")
    return yaml_path

def train_yolov8_with_split(json_annotations, images_folder, labels_folder, epochs=5, batch=16, imgsz=640, lr0=0.01, weight_decay=0.0005, run_name="run"):
    """Train YOLOv8 model with the specified parameters."""
    # Verify the image directory structure
    for split in ["train", "val", "test"]:
        split_images_dir = Path(images_folder) / split / "images"
        if not split_images_dir.exists() or len(list(split_images_dir.glob("*.jpg"))) == 0:
            print(f"Warning: No images found in {split_images_dir}")
    
    # Calculate splits
    train_count, val_count = split_data(images_folder, labels_folder)
    print(f"Split {train_count} images for training and {val_count} for validation.")
    
    # Get absolute path of project root
    dataset_root = Path(__file__).parent.absolute()
    
    # Create dataset.yaml file
    dataset_yaml = create_dataset_yaml_from_categories(json_annotations, dataset_root, images_folder)
    
    # Train model
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name=run_name,
        save=True,
        save_period=1,
        device="cpu",
        cache=True,
        lr0=lr0,
        weight_decay=weight_decay,
    )
    
    return results

# Main execution
if __name__ == "__main__":
    # Get the directory containing the script
    script_dir = Path(__file__).parent.absolute()
    
    # Convert COCO to YOLO format
    conversion_success = convert_coco_to_yolo()
    if not conversion_success:
        print("Warning: Annotation conversion failed or was incomplete.")
        print("Attempting to continue with any existing YOLO format data...")
        
        # If conversion failed, check if we need to manually copy images
        print("\nChecking for images that might need to be manually copied...")
        
        # Look for source images and annotations
        annotations_path = script_dir / 'annotations' / 'manual_annotations.json'
        if annotations_path.exists():
            try:
                with open(annotations_path, 'r') as f:
                    coco = json.load(f)
                
                image_filenames = [img['file_name'] for img in coco['images']]
                print(f"Found {len(image_filenames)} image references in annotations.")
                
                # Check for any images in common directories
                possible_dirs = [
                    script_dir / 'xview_sample',
                    script_dir / 'images',
                    script_dir / 'data',
                    script_dir,
                    Path('xview_sample'),
                    Path('images'),
                    Path('data')
                ]
                
                for dir_path in possible_dirs:
                    if dir_path.exists() and dir_path.is_dir():
                        img_files = list(dir_path.glob('*.jpg'))
                        if img_files:
                            print(f"Found {len(img_files)} image files in {dir_path}")
                            
                            # Create a manual mapping from annotation to actual files
                            # This helps when filenames don't match exactly
                            actual_files = {f.name: f for f in img_files}
                            
                            # Check first few images as examples
                            print("Examples of images found:")
                            for i, f in enumerate(list(actual_files.keys())[:5]):
                                print(f"  {i+1}. {f}")
                                
                            # Ask user if they'd like to copy these images
                            print("\nWould you like to manually copy images to yolo_format directory? (y/n)")
                            print("(Script will continue with existing data if any)")
            
            except Exception as e:
                print(f"Error checking for images: {e}")
    
    json_annotations = "annotations/manual_annotations.json"
    # Adjust to use yolo_format as the primary image source
    images_folder = "yolo_format"
    labels_folder = "yolo_format"
    
    # Sweep over different weight_decay values
    weight_decay_values = [0.0001, 0.0005, 0.001]
    all_metrics = {}
    
    for wd in weight_decay_values:
        run_name = f"weight_decay_{str(wd).replace('.', '_')}"
        print(f"\n=== Training with weight_decay: {wd} ===")
        metrics = train_yolov8_with_split(
            json_annotations=json_annotations,
            images_folder=images_folder,
            labels_folder=labels_folder,
            epochs=5,
            batch=16,
            imgsz=640,
            lr0=0.01,
            weight_decay=wd,
            run_name=run_name
        )
        all_metrics[wd] = metrics
    
    # Plot validation mAP@0.5 per epoch for each weight_decay
    plt.figure(figsize=(8,6))
    for wd, metrics in all_metrics.items():
        epochs = list(range(len(metrics)))
        val_map = [m.get("metrics/mAP_0.5", 0) for m in metrics]
        plt.plot(epochs, val_map, label=f"Weight Decay={wd}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Validation mAP@0.5")
    plt.title("Weight Decay Sweep: Validation mAP@0.5 per Epoch")
    plt.legend()
    plt.savefig("weight_decay_sweep_results.png")
    plt.show()
    print("Plot saved as weight_decay_sweep_results.png")