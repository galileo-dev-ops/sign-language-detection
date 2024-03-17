import tkinter as tk
import threading
import os
import sys

# Import your scripts
sys.path.append(os.path.abspath('path_to_your_scripts'))
import collect_imgs
import create_dataset
import train_classifier
import inference_classifier


def collect_images():
    threading.Thread(target=collect_imgs.main).start()


def create_dataset():
    threading.Thread(target=create_dataset.main).start()


def train_model():
    threading.Thread(target=train_classifier.main).start()


def inference():
    threading.Thread(target=inference_classifier.main).start()


# Create the main window
root = tk.Tk()
root.title("Hand Gesture Recognition")

# Create buttons
collect_button = tk.Button(root, text="Collect Images", command=collect_images)
create_dataset_button = tk.Button(root, text="Create Dataset", command=create_dataset)
train_model_button = tk.Button(root, text="Train Model", command=train_model)
inference_button = tk.Button(root, text="Inference", command=inference)

# Place the buttons
collect_button.pack(fill=tk.X)
create_dataset_button.pack(fill=tk.X)
train_model_button.pack(fill=tk.X)
inference_button.pack(fill=tk.X)

# Start the event loop
root.mainloop()
