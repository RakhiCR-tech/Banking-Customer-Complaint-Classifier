# BankAssist — Banking Customer Complaint Classifier

A beginner-friendly machine-learning portfolio project that classifies short banking complaints into support categories and suggests a responsible department.

## Features
- Text classification with a Multinomial Naive Bayes algorithm implemented using Python's standard library.
- Six complaint categories: UPI and Payments, Cards, Loans, ATM and Cash, Internet Banking, and Account Services.
- Flask web interface with predicted category, illustrative confidence score, and department suggestion.
- Small transparent CSV dataset included.
- No NumPy, Pandas, or Scikit-learn dependency.

## Technology
- Python 3.10+
- Flask
- Python standard library (`csv`, `pickle`, `math`, `re`, `collections`)
- HTML and CSS

## Project structure
```text
banking-complaint-classifier/
├── dataset/complaints.csv
├── templates/index.html
├── static/style.css
├── naive_bayes.py
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run locally

### 1. Create and activate a virtual environment (Windows PowerShell)
```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, use this only for the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
python -m pip install -r requirements.txt
```

### 3. Train the model
```powershell
python train_model.py
```
This creates `model.pkl` and `metrics.json` locally. They are intentionally excluded from Git; retrain after cloning.

### 4. Start the web app
```powershell
python app.py
```
Open http://127.0.0.1:5000 in your browser.

## How it works
1. The dataset contains example complaints with human-assigned categories.
2. The training script shuffles the examples with a fixed seed and uses an 80/20 train/test split.
3. The custom Naive Bayes classifier counts words per category and applies Laplace smoothing.
4. For a submitted complaint, it calculates log probabilities and normalizes the category scores.
5. Flask displays the predicted category and suggested support department.

## Dataset and evaluation limitations
The included dataset is a small, hand-written demonstration dataset, not a representative bank dataset. The reported test accuracy is only a demonstration of the code pipeline and must not be presented as real-world banking performance. Confidence scores are model scores, not calibrated probabilities. Improve the dataset with diverse, ethically sourced, anonymized and correctly labelled examples before drawing conclusions.

## Privacy and responsible use
Do not enter real customer personal information, account numbers, card numbers, passwords, OTPs or transaction identifiers. This project is for learning and portfolio demonstration only; it must not be used to make real banking decisions or replace human complaint review.

## Future improvements
- Add a larger, approved and anonymized dataset.
- Add confusion matrix and per-category evaluation.
- Add complaint history with secure storage.
- Add human review and feedback before any operational use.
- Add authentication and deployment hardening before public production use.

## Author
Rakhi C R
