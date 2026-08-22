import sys, os, torch, numpy as np
sys.path.insert(0, r"D:\brain-tumor-dx\brain-tumor-dx\src")
from brain_tumor_dx.config import settings
from brain_tumor_dx.models.classifier import TumorClassifier
from brain_tumor_dx.data.io import load_image_2d
from brain_tumor_dx.data.preprocessing import preprocess_for_classifier

model = TumorClassifier(num_classes=4)
model.load_state_dict(torch.load(r"D:\brain-tumor-dx\brain-tumor-dx\checkpoints\classifier.pt", map_location="cpu"))
model.eval()

classes = settings.tumor_classes
test_dir = r"D:\archive\Testing"
correct = 0
total = 0
per_class = {c: {"correct": 0, "total": 0} for c in classes}

folder_to_class = {"glioma": "glioma", "meningioma": "meningioma", "notumor": "no_tumor", "pituitary": "pituitary"}

for folder, true_label in folder_to_class.items():
    folder_path = os.path.join(test_dir, folder)
    images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")][:20]
    for fname in images:
        path = os.path.join(folder_path, fname)
        img = load_image_2d(path)
        arr = preprocess_for_classifier(img, 224)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=-1).squeeze()
        pred = classes[probs.argmax().item()]
        total += 1
        per_class[true_label]["total"] += 1
        if pred == true_label:
            correct += 1
            per_class[true_label]["correct"] += 1

print("=== Classification Results (20 images per class) ===\n")
for cls in classes:
    c = per_class[cls]
    acc = c["correct"] / c["total"] if c["total"] > 0 else 0
    print("  {}: {}/{} ({:.0%})".format(cls, c["correct"], c["total"], acc))

print("\nOverall: {}/{} ({:.0%})".format(correct, total, correct / total))
