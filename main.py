import time
import os
import pandas as pd
from ledger import init_mock_databases, run_rule_based_checks, execute_transfer_db, USER_DB_PATH
from model import extract_features, predict_fraud_risk, train_model_on_real_data, MODEL_PATH

def print_ui_header(title):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 50)
    print(f" {title.center(48)} ")
    print("=" * 50)

def display_balances():
    """Helper to show current account states from users.csv"""
    if os.path.exists(USER_DB_PATH):
        df = pd.read_csv(USER_DB_PATH)
        print("\n--- Current System Balances ---")
        for _, row in df.iterrows():
            print(f"User: {row['user_id']} | Balance: {row['currency']} {row['balance']:.2f}")
        print("-" * 31 + "\n")

def main():
    # Step 0: Initialize data files and train the ML model if not already present
    init_mock_databases()
    if not os.path.exists(MODEL_PATH):
        train_model_on_real_data()
        
    while True:
        display_balances()
        print("--- P2P Payment System Menu ---")
        print("1. Send a Payment (User A View)")
        print("2. Force Re-train ML Model")
        print("3. Exit System")
        choice = input("Select an option (1-3): ").strip()
        
        if choice == "3":
            print("\nShutting down payment infrastructure. Goodbye!")
            break
            
        elif choice == "2":
            train_model_on_real_data()
            input("\nPress Enter to return to main menu...")
            continue
            
        elif choice == "1":
            print_ui_header("USER A: INITIATE TRANSACTION")
            sender_id = input("Enter Sender ID (e.g., USER_A): ").strip().upper()
            receiver_id = input("Enter Receiver ID (e.g., USER_B): ").strip().upper()
            
            try:
                amount = float(input("Enter Transfer Amount: ").strip())
            except ValueError:
                print("\n❌ Error: Amount must be a valid number.")
                input("\nPress Enter to restart...")
                continue

            # UI State change: Simulating the "LOADING..." box from your diagram
            print("\n[SYSTEM LOG]: Processing transaction...")
            print("LOADING...")
            time.sleep(1.2) # Artificial processing delay for realism
            
            # ==========================================
            # STEP 1: RULE BASED CHECKING (1 & 2)
            # ==========================================
            rule_status, rule_msg = run_rule_based_checks(sender_id, receiver_id, amount)
            
            if rule_status == "REJECT":
                print_ui_header("USER A: TRANSACTION REJECTED")
                if rule_msg == "NOT ENOUGH BALANCE":
                    print("❌ REJECTION ERROR: NOT ENOUGH BALANCE")
                elif rule_msg == "INVALID RECIPIENT":
                    print("❌ REJECTION ERROR: INVALID RECIPIENT")
                else:
                    print(f"❌ REJECTION ERROR: {rule_msg}")
                input("\nPress Enter to return to main menu...")
                continue
                
            # ==========================================
            # STEP 2: LIVE FEATURE ENGINEERING
            # ==========================================
            # Automatically calculate data points before asking the ML model
            live_features = extract_features(sender_id, receiver_id, amount)
            
            # ==========================================
            # STEP 3: ML BASED CHECKING
            # ==========================================
            is_fraud = predict_fraud_risk(live_features)
            
            if is_fraud == 1:
                print_ui_header("USER A: TRANSACTION BLOCKED")
                print("🚨 SECURITY ALERT: FRAUD DETECTED")
                print("This transaction exhibits suspicious anomalies and has been halted.")
                print(f"Engineered Risk Vectors: {live_features}")
                input("\nPress Enter to return to main menu...")
                continue
                
            # ==========================================
            # STEP 4: EXECUTE TRANSFER & STATE CHANGE
            # ==========================================
            execute_transfer_db(sender_id, receiver_id, amount)
            
            # Success Screens as split in your flowchart
            print_ui_header("TRANSACTION COMPLETED SUCCESSFULLY")
            print(f"✅ USER A VIEW: AMOUNT SENT! ({amount} INR to {receiver_id})")
            print(f"🎉 USER B VIEW: AMOUNT RECEIVED! (+{amount} INR)")
            
            input("\nPress Enter to view updated ledger balances...")

if __name__ == "__main__":
    main()
