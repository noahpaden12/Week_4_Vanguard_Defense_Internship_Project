Setup (all in the terminal for windows, use VS Code if possible thats where I'm running it):
  -"python -m venv yolov8env" 
  -"yolov8env\Scripts\activate"
  -"pip install --upgrade pip setuptools wheel"
  -"pip install numpy==1.24.4"
  -"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
  -"pip install ultralytics"


Then after you input these commands and set up your environment, download the "xview_sample", "yolo_format", and "annotations" folders and put them into a folder that also contains the "model_trainer.py".
(all output folders that the "model_trainer.py" is involved with will be automatically created when you run it for the first time)


Trouble Shooting Commands:
  -"pip install --upgrade --force-reinstall numpy"
  -"pip install --upgrade --force-reinstall ultralytics"
  -"pip install --upgrade --force-reinstall torch torchvision"
  -"python -m venv yolov8-env"
  -"pip install numpy ultralytics torch torchvision"
