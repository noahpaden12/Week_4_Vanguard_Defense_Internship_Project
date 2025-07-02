import json
from pathlib import Path
import shutil
import yaml
from ultralytics import YOLO

def split_data(images_dir, labels_dir, train_ratio=0.8):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    images = sorted(images_dir.glob("*.jpg")) 
    total = len(images)
    train_count = int(total * train_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:]

    for folder in ["train", "val"]:
        (images_dir / folder).mkdir(parents=True, exist_ok=True)
        (labels_dir / folder).mkdir(parents=True, exist_ok=True)

    for img_path in train_images:
        shutil.copy(img_path, images_dir / "train" / img_path.name)
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, labels_dir / "train" / label_path.name)

    for img_path in val_images:
        shutil.copy(img_path, images_dir / "val" / img_path.name)
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, labels_dir / "val" / label_path.name)

    return train_count, total - train_count

def create_dataset_yaml_from_categories(json_file, dataset_root, images_folder="dataset_v1", labels_folder="labels"):
    with open(json_file) as f:
        data = json.load(f)
    categories = sorted(data["categories"], key=lambda x: x["id"])
    names = [cat["name"] for cat in categories]

    dataset = {
        "path": str(dataset_root),
        "train": f"{images_folder}/train",
        "val": f"{images_folder}/val",
        "names": names,
        "nc": len(names),
    }

    yaml_path = Path("dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset, f, sort_keys=False)

    print(f"Dataset YAML saved to {yaml_path}")
    return yaml_path

def train_yolov8_with_split(json_annotations, images_folder, labels_folder, epochs=5, batch=16, imgsz=640, lr0=0.01):
    train_count, val_count = split_data(images_folder, labels_folder)

    print(f"Split {train_count} images for training and {val_count} for validation.")

    dataset_root = Path(images_folder).parent  
    dataset_yaml = create_dataset_yaml_from_categories(json_annotations, dataset_root)

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name="manual_yolov8n_training",
        save=True,
        save_period=1,
        device="cpu",
        cache=True,
        lr0=lr0,
    )

    print(f"Training complete. Best model saved at: runs/train/manual_yolov8n_training/weights/best.pt")

if __name__ == "__main__":
    json_annotations = "annotations/manual_annotations.json"
    images_folder = "xview_sample"  
    labels_folder = "yolo_format"

    train_yolov8_with_split(json_annotations, images_folder, labels_folder, epochs=5, batch=16, imgsz=640, lr0=0.01)
