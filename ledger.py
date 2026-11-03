import os
import uuid
import pandas as pd

USER_DB_PATH = "data\\users.csv"
TRANSACTION_DB_PATH = "data\\transactions.csv"

def init_mock_databases():
    """Initializes the CSV databases with correct headers if they don't exist or are empty."""
    os.makedirs("data", exist_ok=True)
    
    # 1. Initialize Users Database safely
    should_create_users = (
        not os.path.exists(USER_DB_PATH) or 
        os.path.getsize(USER_DB_PATH) == 0
    )
    if should_create_users:
        users_df = pd.DataFrame([
            {"user_id": "USER_A", "balance": 5000.00, "currency": "INR"},
            {"user_id": "USER_B", "balance": 150.00, "currency": "INR"},
            {"user_id": "USER_C", "balance": 1200.00, "currency": "INR"}
        ])
        users_df.to_csv(USER_DB_PATH, index=False)
        print("[SYSTEM LOG]: Initialized data/users.csv with headers.")
        
    # 2. Initialize Transactions Database safely
    should_create_txns = (
        not os.path.exists(TRANSACTION_DB_PATH) or 
        os.path.getsize(TRANSACTION_DB_PATH) == 0
    )
    if should_create_txns:
        cols = ["timestamp", "transaction_id", "sender", "receiver", 
                "sent_amount", "sent_currency", "received_amount", "received_currency", "exchange_rate"]
        pd.DataFrame(columns=cols).to_csv(TRANSACTION_DB_PATH, index=False)
        print("[SYSTEM LOG]: Initialized data/transactions.csv with headers.")

def run_rule_based_checks(sender_id, receiver_id, amount):
    """
    Executes Rule Based Checking 1 (Balance) & 2 (Receiver Existence)
    Returns: (status: str, error_message: str)
    """
    users_df = pd.read_csv(USER_DB_PATH)
    
    # Check 1: Does Sender exist and have enough balance?
    sender_row = users_df[users_df['user_id'] == sender_id]
    if sender_row.empty:
        return "REJECT", "SENDER_NOT_FOUND"
        
    sender_balance = sender_row.iloc[0]['balance']
    if sender_balance < amount:
        return "REJECT", "NOT ENOUGH BALANCE"
        
    # Check 2: Does the Receiver exist?
    receiver_row = users_df[users_df['user_id'] == receiver_id]
    if receiver_row.empty:
        return "REJECT", "INVALID RECIPIENT"
        
    # Self-transfer safeguard
    if sender_id == receiver_id:
        return "REJECT", "INVALID TRANSACTION"
        
    return "PASS", "SUCCESS"

def execute_transfer_db(sender_id, receiver_id, amount):
    """
    Executes the green 'EXECUTE TRANSFER' step from your flowchart.
    Deducts balance, adds balance, and appends to transactions.csv.
    """
    users_df = pd.read_csv(USER_DB_PATH)
    
    # 1. Adjust balances in memory
    users_df.loc[users_df['user_id'] == sender_id, 'balance'] -= amount
    users_df.loc[users_df['user_id'] == receiver_id, 'balance'] += amount
    users_df.to_csv(USER_DB_PATH, index=False) # Commit balance changes
    
    # 2. Append new row to transactions ledger
    txn_record = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "transaction_id": str(uuid.uuid4())[:8],
        "sender": sender_id,
        "receiver": receiver_id,
        "sent_amount": amount,
        "sent_currency": "INR",
        "received_amount": amount,
        "received_currency": "INR",
        "exchange_rate": 1.0
    }
    
    txns_df = pd.read_csv(TRANSACTION_DB_PATH)
    txns_df = pd.concat([txns_df, pd.DataFrame([txn_record])], ignore_index=True)
    txns_df.to_csv(TRANSACTION_DB_PATH, index=False)
