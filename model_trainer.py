import json
from pathlib import Path
import yaml
from ultralytics import YOLO

def create_dataset_yaml_from_categories(json_file, dataset_root, images_folder="dataset_v1"):
    with open(json_file) as f:
        data = json.load(f)

    categories = sorted(data["categories"], key=lambda x: x["id"])
    names = [cat["name"] for cat in categories]

    dataset = {
        "path": str(dataset_root),
        "train": f"{images_folder}/test/images",
        "val": f"{images_folder}/test/images",
        "names": names,
        "nc": len(names),
    }

    yaml_path = dataset_root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset, f, sort_keys=False)

    print(f"✅ dataset.yaml saved to: {yaml_path}")
    return yaml_path

def train_yolov8(json_annotations, dataset_folder="dataset_v1", epochs=3, batch=16, imgsz=640, lr0=0.01):
    dataset_root = Path(dataset_folder).resolve().parent
    yaml_path = create_dataset_yaml_from_categories(
        json_file=json_annotations,
        dataset_root=dataset_root,
        images_folder=Path(dataset_folder).name
    )

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name="manual_yolov8n_training",
        save=True,
        save_period=1,
        device="cpu",  # "cuda" if you have GPU
        cache=True,
        lr0=lr0,
    )

    print("✅ Training complete.")
    print("📦 Best model saved at: runs/train/manual_yolov8n_training/weights/best.pt")

if __name__ == "__main__":
    json_annotations = "dataset_v1/test/annotations.json"  # label annotations in test folder
    dataset_folder = "dataset_v1"

    train_yolov8(
        json_annotations=json_annotations,
        dataset_folder=dataset_folder,
        epochs=3,
        batch=16,
        imgsz=640,
        lr0=0.01
    )
