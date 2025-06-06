# AI-Powered-Financial-Profile-Analyzer
Which helps to find the financial health status

This is a Streamlit-based interactive web application that analyzes an individual's financial profile using a PyTorch deep learning model. The tool provides a comprehensive financial health report with visual insights and personalized recommendations.

---

## 🚀 Features

- Predicts financial risk level (Low, Medium, High)
- Calculates financial metrics: savings rate, expense ratio, disposable income
- Displays health grade based on smart scoring logic
- Offers personalized financial advice based on risk
- Visualizes income allocation and ratios with matplotlib
- Fully interactive Streamlit user interface

---

## 🧠 Model Overview

- Built using PyTorch
- Multiclass classification (3 financial risk categories)
- Trained on synthetic financial data
- Uses engineered financial ratios like:
  - Savings Rate
  - Expense Ratio
  - Debt-to-Income Ratio
  - Disposable Income

---

## 🛠 Tech Stack

- **Frontend:** Streamlit
- **Backend:** PyTorch (Neural Network)
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML Utilities:** Scikit-learn

---

## 📦 Installation

### 1. Clone the repo:
```bash
git clone https://github.com/your-username/financial-health-analyzer.git
cd financial-health-analyzer
2. Create a virtual environment (optional but recommended):
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
3. Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
4. Run the app:
bash
Copy
Edit
streamlit run financial_analyzer.py
🧪 Example Inputs
Monthly Income: $5000

Savings: $10,000

Fixed Expenses: $2000

Variable Expenses: $1500

Age: 30

Dependents: 1

Output: Risk profile, financial metrics, pie/bar charts, and suggestions.

📈 Future Enhancements
Add authentication and save user history

Integrate real-time financial APIs

Add investment & goal planning modules

Enable model retraining from user data

📄 License
This project is open-source under the MIT License.

🙋‍♂️ Author
Tallam Venkata Hanuman
B.Tech CSE | KL University
