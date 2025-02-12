import os
import numpy as np

labels_dir = "data/dataset/train/labels"
all_labels = []

for label_file in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, label_file)
    annotation = np.load(label_path, allow_pickle=True)
    all_labels.extend(annotation.flatten())

# Compter les occurrences
unique, counts = np.unique(all_labels, return_counts=True)
print("Distribution des labels :", dict(zip(unique, counts)))
