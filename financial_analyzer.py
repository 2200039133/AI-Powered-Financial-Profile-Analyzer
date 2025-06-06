import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import streamlit as st

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Neural Network Model
class FinancialHealthNN(nn.Module):
    def __init__(self, input_size):
        super(FinancialHealthNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 output classes (Low, Medium, High risk)
        )
        
    def forward(self, x):
        return self.layers(x)

# Preprocessing Function
def preprocess_data(df):
    df['savings_rate'] = df['savings'] / df['income']
    df['expense_ratio'] = (df['fixed_expenses'] + df['variable_expenses']) / df['income']
    df['debt_to_income'] = df['debt'] / df['income']
    df['disposable_income'] = df['income'] - (df['fixed_expenses'] + df['variable_expenses'])

    features = df[['income', 'age', 'dependents', 'savings', 
                   'fixed_expenses', 'variable_expenses', 
                   'savings_rate', 'expense_ratio', 'disposable_income']]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, scaler

# Model Training
def train_model(X_train, y_train, epochs=100, lr=0.001):
    input_size = X_train.shape[1]
    model = FinancialHealthNN(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    return model

# Generate Synthetic Data
def generate_training_data(num_samples=1000):
    data = {
        'income': np.random.normal(50000, 15000, num_samples),
        'age': np.random.randint(20, 65, num_samples),
        'dependents': np.random.randint(0, 4, num_samples),
        'savings': np.random.normal(20000, 10000, num_samples),
        'fixed_expenses': np.random.normal(15000, 5000, num_samples),
        'variable_expenses': np.random.normal(10000, 3000, num_samples),
        'debt': np.random.normal(10000, 8000, num_samples)
    }

    df = pd.DataFrame(data)

    # ✅ Add savings_rate before using it in conditions
    df['savings_rate'] = df['savings'] / df['income']

    conditions = [
        (df['savings_rate'] > 0.3) & (df['dependents'] <= 1),
        (df['savings_rate'] > 0.15) & (df['savings_rate'] <= 0.3),
        (df['savings_rate'] <= 0.15)
    ]
    choices = [2, 1, 0]  # High, Medium, Low risk
    df['risk_profile'] = np.select(conditions, choices)

    return df

# Streamlit App
def main():
    st.title("AI-Powered Financial Profile Analyzer")
    st.subheader("Complete Financial Health Assessment")

    with st.form("financial_profile"):
        col1, col2 = st.columns(2)

        with col1:
            income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
            age = st.slider("Age", 18, 80, 30)
            dependents = st.slider("Number of Dependents", 0, 10, 1)

        with col2:
            savings = st.number_input("Current Savings ($)", min_value=0, value=10000)
            fixed_exp = st.number_input("Fixed Expenses ($)", min_value=0, value=2000)
            var_exp = st.number_input("Variable Expenses ($)", min_value=0, value=1500)

        submitted = st.form_submit_button("Analyze Financial Health")

    if submitted:
        input_data = pd.DataFrame([[income, age, dependents, savings, fixed_exp, var_exp]],
                                  columns=['income', 'age', 'dependents', 'savings', 
                                           'fixed_expenses', 'variable_expenses'])
        input_data['debt'] = 0  # Placeholder if debt not entered

        train_df = generate_training_data()
        X, scaler = preprocess_data(train_df)
        y = train_df['risk_profile'].values

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)

        user_features, _ = preprocess_data(input_data)
        user_tensor = torch.FloatTensor(user_features)

        with torch.no_grad():
            outputs = model(user_tensor)
            _, predicted = torch.max(outputs.data, 1)
            risk_level = predicted.item()

        st.subheader("Financial Health Analysis")
        total_exp = fixed_exp + var_exp
        savings_rate = savings / income if income > 0 else 0
        disposable_income = income - total_exp

        risk_map = {0: "Low", 1: "Medium", 2: "High"}
        risk_label = risk_map[risk_level]

        health_score = (savings_rate * 0.4 +
                        (disposable_income / income) * 0.3 +
                        (1 - (total_exp / income)) * 0.3)

        if health_score > 0.7:
            health_grade = "A (Excellent)"
        elif health_score > 0.5:
            health_grade = "B (Good)"
        elif health_score > 0.3:
            health_grade = "C (Needs Improvement)"
        else:
            health_grade = "D (At Risk)"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Profile", risk_label)
            st.metric("Financial Health Grade", health_grade)

        with col2:
            st.metric("Savings Rate", f"{savings_rate:.1%}")
            st.metric("Disposable Income", f"${disposable_income:,.2f}")

        with col3:
            st.metric("Expense Ratio", f"{(total_exp/income):.1%}")
            st.metric("Monthly Surplus/Deficit", 
                      f"${disposable_income - savings:,.2f}", delta_color="inverse")

        st.subheader("Spending Breakdown")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        expenses = {
            'Fixed': fixed_exp,
            'Variable': var_exp,
            'Savings': savings,
            'Remaining': max(0, disposable_income - savings)
        }
        ax[0].pie(expenses.values(), labels=expenses.keys(), autopct='%1.1f%%')
        ax[0].set_title("Income Allocation")

        metrics = {
            'Savings Rate': savings_rate,
            'Expense Ratio': total_exp/income,
            'Disposable Income': disposable_income/income
        }
        ax[1].bar(metrics.keys(), metrics.values())
        ax[1].set_title("Financial Ratios")
        ax[1].set_ylim(0, 1)

        st.pyplot(fig)

        st.subheader("Personalized Recommendations")
        if risk_level == 0:
            st.success("""
            **Excellent financial health!** Recommendations:
            - Consider investing 20-30% of your disposable income
            - Explore tax-advantaged retirement accounts
            - Review insurance coverage for optimal protection
            """)
        elif risk_level == 1:
            st.warning("""
            **Good financial health with room for improvement:** Recommendations:
            - Aim to increase savings rate by 5%
            - Review variable expenses for potential reductions
            - Build 3-6 month emergency fund
            """)
        else:
            st.error("""
            **Financial health needs attention:** Recommendations:
            - Create a strict budget to reduce expenses
            - Prioritize building a small emergency fund
            - Consider debt consolidation if applicable
            - Seek professional financial advice
            """)

if __name__ == "__main__":
    main()
