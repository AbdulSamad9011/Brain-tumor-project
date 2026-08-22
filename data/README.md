# Data

This folder is intentionally empty in version control (`.gitignore` excludes
everything except `.gitkeep`) — datasets are large and license-restricted,
so they're never committed.

Run `python scripts/download_data.py` to scaffold the expected directory
layout, then manually download:

- **Classification**: Kaggle "Brain Tumor MRI Dataset"
- **Segmentation**: BraTS (Brain Tumor Segmentation Challenge)

See that script's docstring for the exact folder structure expected by
`src/brain_tumor_dx/data/datasets.py`.
