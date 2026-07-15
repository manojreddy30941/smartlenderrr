# Smart Lender

AI-powered loan approval prediction system built from the provided guide. The project trains a classifier on the Kaggle loan approval dataset, saves the best model, and serves predictions through a Flask web app.

## Project Structure

```text
smartlenderrr/
├── app.py
├── train.py
├── data/
│   └── loan_approval_dataset.csv
├── models/
│   ├── loan_model.joblib
│   └── metadata.joblib
├── templates/
│   └── index.html
├── static/
│   ├── script.js
│   └── style.css
├── notebook.ipynb
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

XGBoost is listed in `requirements.txt`, but `train.py` also works without it by comparing Logistic Regression and Random Forest.

## Train

```bash
python train.py
```

The training pipeline cleans whitespace in the CSV, encodes categorical fields, compares available models, and writes:

- `models/loan_model.joblib`
- `models/metadata.joblib`

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

## API

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"dependents\":0,\"education\":0,\"self_employed\":0,\"income\":8000000,\"loan_amount\":1000000,\"loan_term\":5,\"cibil_score\":780,\"residential_assets\":3000000,\"commercial_assets\":2000000,\"luxury_assets\":1500000,\"bank_assets\":500000}"
```
