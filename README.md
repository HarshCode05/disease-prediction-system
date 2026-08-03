# AI-Powered Multi-Disease Prediction System

An end-to-end machine learning system that predicts risk for **Heart Disease**, **Diabetes**, and **Chronic Kidney Disease** using classical ML and deep learning models, deployed as an interactive Streamlit app with an AI chatbot layer for follow-up health guidance.

## Live Demo
[Add your deployed Streamlit link here once deployed]

## Features
- Three independent prediction pipelines (Heart, Diabetes, Kidney), each trained on its own dataset
- Classical ML models: Logistic Regression, Random Forest, XGBoost
- Deep Learning: custom Artificial Neural Network (Keras/TensorFlow) with a controlled **activation function comparison** (ReLU vs tanh vs sigmoid vs ELU)
- AI chatbot (Google Gemini API) that gives general, non-diagnostic follow-up guidance after a prediction
- Full exploratory data analysis for each dataset, including handling real-world data issues (missing values, hidden zero-as-missing encoding, mixed categorical/numeric types)

## Tech Stack
| Category | Tools |
|---|---|
| Data Analysis | pandas, numpy, matplotlib, seaborn |
| Classical ML | scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| LLM / Chatbot | Google Gemini API (via OpenAI-compatible SDK) |
| Deployment | Streamlit |

## Project Structure
```
disease-prediction-system/
├── data/                  # datasets (not committed - see Setup below)
├── notebooks/             # EDA + model training/experimentation notebooks
├── src/                   # reusable preprocessing, training scripts, chatbot module
├── models/                # trained model files (generated locally, not committed)
├── app/                   # Streamlit application
└── requirements.txt
```

## Setup

### 1. Clone and install dependencies
```bash
git clone <your-repo-url>
cd disease-prediction-system
pip install -r requirements.txt
```

### 2. Download datasets
Place the following files in the `data/` folder:
- `heart_disease.csv` — [UCI Heart Disease dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- `diabetes.csv` — [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- `kidney_disease.csv` — [Chronic Kidney Disease dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease)

### 3. Train the models
Run the notebooks in order (01 → 07), or run the equivalent scripts in `src/`:
```bash
cd src
python train_heart_model.py
python train_diabetes_model.py
python train_kidney_model.py
```
This generates the `.h5` model files and scalers inside `models/`.

### 4. Set up the chatbot API key
Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com) (no credit card required), then set it as an environment variable:
```bash
export GEMINI_API_KEY="your-key-here"      # Mac/Linux
$env:GEMINI_API_KEY="your-key-here"        # Windows PowerShell
```

### 5. Run the app
```bash
cd app
streamlit run streamlit_app.py
```

## Model Performance Summary

| Disease | Best Model | F1 Score | ROC-AUC |
|---|---|---|---|
| Heart Disease | Random Forest | 0.869 | 0.917 |
| Diabetes | XGBoost | 0.641 | 0.827 |
| Chronic Kidney Disease | Random Forest | 1.000 | 1.000 |

**Note on Kidney Disease results**: the near-perfect scores reflect this being a small (~400 row), highly separable dataset rather than a claim that the model is flawless — this was validated by comparing multiple model types that all converged on similarly high scores, and is discussed further in `notebooks/07_final_comparison.ipynb`.

## Disclaimer
This project is for educational and portfolio purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

## Author
Harsh
