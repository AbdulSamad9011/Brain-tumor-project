# Evaluation Results Summary

| Date | Task | Metric | Value | Model | Dataset | Notes |
|------|------|--------|-------|-------|---------|-------|
| 2026-08-28 | Classification | accuracy | 0.9889 | ResNet50 | 5429 imgs | arch=ResNet50 epochs=N/A lr=N/A batch=32 data=5429 |
| 2026-08-28 | Classification | precision | 0.9889 | ResNet50 | 5429 imgs | arch=ResNet50 epochs=N/A lr=N/A batch=32 data=5429 |
| 2026-08-28 | Classification | recall | 0.9890 | ResNet50 | 5429 imgs | arch=ResNet50 epochs=N/A lr=N/A batch=32 data=5429 |
| 2026-08-28 | Classification | f1 | 0.9890 | ResNet50 | 5429 imgs | arch=ResNet50 epochs=N/A lr=N/A batch=32 data=5429 |
| 2026-08-28 | Classification | kappa | 0.9852 | ResNet50 | 5429 imgs | arch=ResNet50 epochs=N/A lr=N/A batch=32 data=5429 |
| 2026-08-28 | Classification | mcc | 0.9852 | ResNet50 | 5429 imgs | arch=ResNet50 epochs=N/A lr=N/A batch=32 data=5429 |
| 2026-08-28 | Classification (glioma) | precision | 0.9949 | ResNet50 | 5429 imgs | arch=ResNet50 epochs=N/A lr=N/A batch=32 data=5429 |
| 2026-08-28 | Classification (meningioma) | precision | 0.9904 | ResNet50 | 5429 imgs | arch=ResNet50 epochs=N/A lr=N/A batch=32 data=5429 |
| 2026-08-28 | Classification (pituitary) | precision | 0.9862 | ResNet50 | 5429 imgs | arch=ResNet50 epochs=N/A lr=N/A batch=32 data=5429 |
| 2026-08-28 | Classification (no_tumor) | precision | 0.9842 | ResNet50 | 5429 imgs | arch=ResNet50 epochs=N/A lr=N/A batch=32 data=5429 |

## Training Details

### Classifier
- **Architecture**: ResNet50 (pretrained ImageNet, fine-tuned)
- **Dataset**: Kaggle Brain Tumor MRI (5,429 images after deduplication)
  - glioma: 1,400 | meningioma: 1,386 | pituitary: 1,362 | no_tumor: 1,281
- **Split**: 85% train (4,615) / 15% val (814), seed=42
- **Training**: AdamW, lr=1e-4, weight_decay=1e-4, CosineAnnealingLR, label_smoothing=0.1
- **Hardware**: Tesla T4 (16GB VRAM) on Google Colab
- **Best epoch**: 9/20, early stopping at epoch 19

### Segmenter
- **Architecture**: MONAI 3D-UNet (encoder-decoder with skip connections)
- **Dataset**: BraTS2020 HDF5 → 369 NIfTI volumes
- **Loss**: Dice + BCE (pos_weight=10.0)
- **Augmentation**: Random flips along 3 axes
- **Split**: 80% train (295) / 20% val (74), seed=42

## Known Limitations
- Classifier trained on Kaggle dataset (not full medical-grade dataset)
- Segmenter Dice ~0.24 (class imbalance, small dataset, needs more training)
- Both models are decision-support only — require radiologist confirmation
