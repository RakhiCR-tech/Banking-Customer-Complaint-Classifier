# 🏦 Banking Customer Complaint Classification System

An intelligent banking customer support application that automatically classifies customer complaints into relevant banking categories using Machine Learning and recommends the appropriate support department.

The application uses a custom **Multinomial Naive Bayes classifier implemented in pure Python** and a Flask-based web interface to make complaint classification simple and accessible.

---

## 📌 Project Overview

Banks receive customer complaints through different channels, including online banking, mobile applications, ATM services, debit and credit cards, loans, and account-related services.

Manually reviewing and categorizing every complaint can take time and may result in delays in directing complaints to the appropriate department.

This project demonstrates how Machine Learning can help automate the initial complaint classification process.

Customers can enter their complaints through a web interface. The trained model analyzes the complaint text, predicts the most relevant category, displays category scores, and suggests the appropriate support department.

### 🎯 Project Objectives

- Automate the initial classification of banking customer complaints.
- Identify the most relevant complaint category from customer-provided text.
- Recommend the appropriate banking support department.
- Demonstrate a practical application of Natural Language Processing (NLP).
- Build an interactive web application using Python and Flask.
- Implement a Machine Learning algorithm without relying on external machine-learning libraries.

---

## ✨ Key Features

1. **Automatic Complaint Classification**  
   Predicts the category of a banking complaint based on its text.

2. **Six Banking Complaint Categories**  
   Supports UPI and Payments, Cards, Loans, ATM and Cash, Internet Banking, and Account Services.

3. **Model Confidence Display**  
   Displays the model's probability score for its predicted category.

4. **Category-wise Scores**  
   Shows the scores associated with the other complaint categories.

5. **Department Recommendation**  
   Suggests the relevant support department based on the predicted category.

6. **Interactive Web Dashboard**  
   Provides a simple interface for entering complaints and viewing classification results.

7. **Custom Machine Learning Implementation**  
   Implements Multinomial Naive Bayes using Python without NumPy or Scikit-learn.

8. **Model Training and Evaluation**  
   Includes a separate training script that prepares the dataset, trains the classifier, and evaluates its performance.

---

## 🧠 Complaint Categories

The system supports six major banking complaint categories.

| No. | Complaint Category | Suggested Department |
|---|---|---|
| 1 | UPI and Payments | Digital Payments Support |
| 2 | Cards | Card Services Support |
| 3 | Loans | Loan Services Support |
| 4 | ATM and Cash | ATM Operations Support |
| 5 | Internet Banking | Internet Banking Support |
| 6 | Account Services | Account Services Support |

### Example Customer Complaints

| Customer Complaint | Expected Category |
|---|---|
| Money was deducted from my account, but the UPI transaction failed. | UPI and Payments |
| My debit card is blocked and I cannot use it. | Cards |
| My loan EMI was deducted twice. | Loans |
| The ATM deducted money, but cash was not dispensed. | ATM and Cash |
| I cannot log in to my internet banking account. | Internet Banking |
| I want to update my registered mobile number. | Account Services |

---

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Multinomial Naive Bayes | Complaint classification algorithm |
| Natural Language Processing | Text preprocessing and tokenization |
| Flask | Backend web framework |
| HTML5 | Web page structure |
| CSS3 | Dashboard styling |
| CSV | Dataset storage |
| Pickle | Trained model serialization |
| JSON | Training evaluation metrics |
| Git and GitHub | Version control and project hosting |

### Python Libraries

- Flask
- Python Standard Library modules

The custom classification algorithm is implemented without NumPy or Scikit-learn.

---

## 🤖 Machine Learning Algorithm

### Multinomial Naive Bayes

This project uses the Multinomial Naive Bayes algorithm for text classification.

Naive Bayes is a probabilistic machine-learning algorithm based on Bayes' theorem. It estimates the probability of a complaint belonging to each available category based on the words appearing in the complaint.

### How the Model Works

**Step 1: Dataset Preparation**

The complaint dataset is loaded from a CSV file. Each record contains a complaint description and its corresponding category.

**Step 2: Text Preprocessing**

The complaint text is converted into tokens. Basic preprocessing helps the model analyze the words in the customer complaint.

**Step 3: Training**

The classifier learns the word frequencies associated with each complaint category.

**Step 4: Probability Calculation**

The model calculates category probabilities using Multinomial Naive Bayes with Laplace smoothing.

**Step 5: Prediction**

The category with the highest calculated probability is selected as the predicted complaint category.

**Step 6: Department Recommendation**

The predicted category is mapped to its corresponding banking support department.

### Laplace Smoothing

Laplace smoothing is used to reduce the problem of zero probabilities when a word does not appear in a particular category during training.

---

## 📂 Project Structure

```text
Banking-Customer-Complaint-Classifier/
│
├── dataset/
│   └── complaints.csv
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── naive_bayes.py
│
├── train_model.py
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

### File Description

| File / Folder | Description |
|---|---|
| `dataset/complaints.csv` | Contains sample banking customer complaints and their categories. |
| `templates/index.html` | Defines the web application's HTML interface. |
| `static/style.css` | Contains the dashboard styling and visual design. |
| `naive_bayes.py` | Implements the custom Multinomial Naive Bayes classifier. |
| `train_model.py` | Loads the dataset, trains the model, evaluates it, and saves model artifacts. |
| `app.py` | Runs the Flask application and handles customer complaint predictions. |
| `requirements.txt` | Lists the Python package dependencies. |
| `.gitignore` | Excludes virtual environments, generated model files, and unnecessary files from Git. |

---

## 📊 Dataset Information

The project uses a manually prepared sample dataset containing **60 banking customer complaint records**.

The dataset covers six complaint categories.

### Dataset Split

| Dataset | Number of Records |
|---|---:|
| Total records | 60 |
| Training records | 48 |
| Testing records | 12 |

The training script uses an 80:20 training and testing split.

The dataset is intended for educational and demonstration purposes.

---

## 📈 Model Evaluation

The model is evaluated using a held-out test dataset.

### Initial Evaluation Result

| Evaluation Metric | Result |
|---|---:|
| Total dataset records | 60 |
| Training records | 48 |
| Testing records | 12 |
| Test Accuracy | 41.67% |

**Important:** The reported accuracy is the actual initial result obtained from the demonstration dataset. It should not be interpreted as evidence of production-level performance.

The dataset is relatively small, and the model requires additional training examples and more comprehensive evaluation before practical use.

---

## 🖥️ Application Workflow

The application follows this workflow:

1. The user opens the Banking Complaint Classification dashboard.
2. The user enters a banking customer complaint.
3. The complaint text is submitted to the Flask backend.
4. The text is processed by the custom Naive Bayes classifier.
5. The model calculates category probabilities.
6. The application displays the predicted category and category scores.
7. The application recommends the corresponding support department.

---

## 🚀 Installation and Setup

Follow these instructions to run the project locally.

### Prerequisites

Make sure the following software is installed:

- Python 3.10 or later
- Visual Studio Code (recommended)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/RakhiCR-tech/Banking-Customer-Complaint-Classifier.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Banking-Customer-Complaint-Classifier
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 6: Train the Machine Learning Model

```bash
python train_model.py
```

This command:

- Loads the complaint dataset.
- Prepares the training and testing data.
- Trains the Naive Bayes classifier.
- Evaluates the model.
- Generates the trained model and evaluation metrics.

The training process generates:

```text
model.pkl
metrics.json
```

These generated files are excluded from Git using `.gitignore`. Run the training script after cloning the repository to generate them locally.

### Step 7: Start the Flask Application

```bash
python app.py
```

### Step 8: Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000
```

The Banking Complaint Classification dashboard will be available locally.

---

## 🔍 Example Prediction

**Sample Customer Complaint:**

```text
I transferred money through UPI, but the amount
was deducted from my account and the receiver
did not receive the payment.
```

**Expected Output:**

| Output | Result |
|---|---|
| Predicted Category | UPI and Payments |
| Suggested Department | Digital Payments Support |
| Model Confidence | Calculated by the trained model |

The displayed confidence is a model-generated probability score, not a guarantee that the prediction is correct.

---

## 🔐 Limitations and Responsible Use

This application is an educational Machine Learning demonstration.

- The dataset contains only 60 manually prepared sample records.
- The initial test accuracy is 41.67%.
- The model may misclassify complaints containing unfamiliar words or ambiguous descriptions.
- The displayed probability scores are not calibrated confidence estimates.
- The application is not integrated with any real banking system.
- The application does not connect to customer bank accounts or payment networks.
- It should not be used to make real banking, financial, or customer-service decisions.
- Do not enter real customer information, account numbers, card details, passwords, OTPs, or other sensitive banking data.

The model should be improved and independently validated before any real-world deployment.

---

## 🔮 Future Enhancements

The following improvements can be considered in future versions:

- Expand the dataset with more diverse banking complaints.
- Improve text preprocessing and model performance.
- Add precision, recall, F1-score, and confusion matrix evaluation.
- Introduce additional banking complaint categories.
- Add complaint history and classification reports.
- Develop an administrator dashboard.
- Store complaint records using a database.
- Add secure user authentication.
- Explore more advanced NLP and Machine Learning models.
- Deploy the application to a cloud platform.

---

## 👩‍💻 Author

**Rakhi C R**

GitHub: [@RakhiCR-tech](https://github.com/RakhiCR-tech)

---

## 📜 License

This project is intended for educational and portfolio demonstration purposes.

---

## ⭐ Acknowledgement

This project was developed to demonstrate the practical application of Python, Natural Language Processing, Machine Learning, and Flask in the banking customer support domain.

If you find this project useful, consider giving the repository a star!

**Repository:** [Banking Customer Complaint Classifier](https://github.com/RakhiCR-tech/Banking-Customer-Complaint-Classifier)
