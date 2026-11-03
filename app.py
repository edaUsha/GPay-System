from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
from ledger import init_mock_databases, run_rule_based_checks, execute_transfer_db
from model import extract_features, predict_fraud_risk, train_model_on_real_data, MODEL_PATH

app = Flask(__name__)
app.secret_key = "secure_fintech_token_key"

USER_DB_PATH = "data/users.csv"

init_mock_databases()
if not os.path.exists(MODEL_PATH):
    train_model_on_real_data()

@app.route('/')
def root():
    if 'user_id' in session:
        return redirect(url_for('home_menu'))
    return render_template('index.html', screen='desktop_6')

@app.route('/register-screen')
def register_screen():
    """Renders the account creation onboarding panel view"""
    return render_template('index.html', screen='register')

@app.route('/register', methods=['POST'])
def handle_registration():
    """Validates uniqueness and commits new identity data to users.csv ledger database"""
    reg_user = request.form.get('reg_username', '').strip().upper()
    reg_pass = request.form.get('reg_password', '').strip()
    reg_curr = request.form.get('reg_currency', 'INR').strip().upper()
    
    if not reg_user or not reg_pass:
        return render_template('index.html', screen='desktop_6', msg_error="Fields cannot be blank.")

    df = pd.read_csv(USER_DB_PATH)
    
    # Security Rule Check: Prevent overlapping matching Username duplicates
    if reg_user in df['user_id'].values:
        return render_template('index.html', screen='desktop_6', msg_error="User ID already exists. Choose another.")
        
    # Append the newly engineered wallet row with custom baseline parameters (1000 starting wallet allocation)
    new_user_row = {
        "user_id": reg_user,
        "balance": 1000.00,
        "currency": reg_curr,
        "password": reg_pass
    }
    
    df = pd.concat([df, pd.DataFrame([new_user_row])], ignore_index=True)
    df.to_csv(USER_DB_PATH, index=False)
    
    # Redirect back to Desktop 6 with verification message status variables
    return render_template('index.html', screen='desktop_6', msg_success="Profile Created! You can now log in.")

@app.route('/login', methods=['POST'])
def login_auth():
    username = request.form.get('username', '').strip().upper()
    password = request.form.get('password', '').strip()
    
    if os.path.exists(USER_DB_PATH):
        df = pd.read_csv(USER_DB_PATH)
        user_record = df[df['user_id'] == username]
        
        if not user_record.empty and str(user_record.iloc[0]['password']) == password:
            session['user_id'] = username
            return redirect(url_for('home_menu'))
            
    return render_template('index.html', screen='desktop_6', msg_error="Invalid Credentials. Try Again.")

@app.route('/home')
def home_menu():
    if 'user_id' not in session: return redirect(url_for('root'))
    return render_template('index.html', screen='desktop_1')

@app.route('/pay-screen')
def pay_screen():
    if 'user_id' not in session: return redirect(url_for('root'))
    return render_template('index.html', screen='desktop_2')

@app.route('/check-balance')
def check_balance():
    if 'user_id' not in session: return redirect(url_for('root'))
    df = pd.read_csv(USER_DB_PATH)
    user_record = df[df['user_id'] == session['user_id']]
    bal = user_record.iloc[0]['balance']
    curr = user_record.iloc[0]['currency']
    return render_template('index.html', screen='desktop_7', balance_val=f"{bal:,.2f}", currency_val=curr)

@app.route('/process-payment', methods=['POST'])
def process_payment():
    if 'user_id' not in session: return redirect(url_for('root'))
    sender_id = session['user_id']
    receiver_id = request.form.get('receiver_id', '').strip().upper()
    try:
        amount = float(request.form.get('amount', 0))
    except ValueError:
        return render_template('index.html', screen='desktop_5')

    rule_status, rule_msg = run_rule_based_checks(sender_id, receiver_id, amount)
    if rule_status == "REJECT":
        if rule_msg == "NOT ENOUGH BALANCE":
            return render_template('index.html', screen='desktop_5')
        elif rule_msg == "INVALID RECIPIENT":
            return render_template('index.html', screen='desktop_4')
        return render_template('index.html', screen='desktop_5')
        
    live_features = extract_features(sender_id, receiver_id, amount)
    is_fraud = predict_fraud_risk(live_features)
    if is_fraud == 1:
        return render_template('index.html', screen='ml_block')
        
    execute_transfer_db(sender_id, receiver_id, amount)
    return render_template('index.html', screen='desktop_3', receiver=receiver_id)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('root'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
