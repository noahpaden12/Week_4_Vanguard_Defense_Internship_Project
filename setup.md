Setup (all in the terminal for windows, use VS Code if possible thats where I'm running it):
Then after you input these commands and set up your environment, download the "xview_sample", "yolo_format", and "annotations" folders and put them into a folder that also contains the "model_trainer.py".

## Important Note for YOLOv8 Training
The annotations in the "annotations/yolov8_annotations.json" file are in COCO format, but YOLOv8 requires annotations in YOLO format (txt files). The updated model_trainer.py includes a function to convert COCO format to YOLO format automatically before training.

Make sure:
1. The dataset.yaml file is configured correctly with paths to train, val, and test sets
2. The conversion from COCO to YOLO format runs successfully
3. After conversion, check that the yolo_format/train/labels, yolo_format/val/labels, and yolo_format/test/labels directories contain .txt files with annotations

## Clean Restart Steps
If you need to completely reset and start from scratch:
1. Delete the `runs` folder (this contains previous training outputs and logs)
2. Make sure the yolo_format folder is clean or let the conversion function recreate it
3. Run model_trainer.py which will first convert annotations and then begin training

If you still get the "no labels found in detect set" error, run the verify_annotations.py script to diagnose the issue.
