# Models Folder

This folder is intentionally empty in the repository.

Trained model files (`.h5`) and fitted scalers/column-lists (`.pkl`) are
generated when you run the training notebooks or scripts locally, and are
not committed to version control (they're large binary files and can
always be regenerated from the notebooks + raw data).

## How this folder gets populated

Run any ONE of the following for each disease:

**Option A - via Jupyter notebooks** (recommended first time, lets you see
the classical ML vs ANN comparison and activation function experiments):
```
notebooks/04_heart_models.ipynb
notebooks/05_diabetes_models.ipynb
notebooks/06_kidney_models.ipynb
```

**Option B - via clean scripts** (faster, no experimentation, just trains
and saves the final chosen model):
```bash
cd src
python train_heart_model.py
python train_diabetes_model.py
python train_kidney_model.py
```

## Expected files after running everything

```
models/
├── heart_model.h5
├── heart_scaler.pkl
├── diabetes_model.h5
├── diabetes_scaler.pkl
├── kidney_model.h5
├── kidney_scaler.pkl
└── kidney_columns.pkl
```

The Streamlit app (`app/streamlit_app.py`) expects all of these files to
exist before it can make predictions - run the training first, then launch
the app.
