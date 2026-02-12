import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_transaction_data(n=50):
    """Generate 50 synthetic credit card transactions with fraud labels"""
    
    transaction_ids = [f'TXN{1000+i}' for i in range(n)]
    start_date = datetime(2024, 3, 1)
    
    # Customer IDs (simulate 20 unique customers)
    customer_ids = [f'C{2000+i}' for i in range(20)]
    
    records = []
    
    # Base probabilities for features
    for i in range(n):
        # Time: simulate more transactions during business hours
        hour = np.random.choice(range(24), p=[0.01,0.01,0.01,0.01,0.01,0.02,0.03,0.04,0.06,0.07,0.08,0.08,
                                               0.08,0.07,0.06,0.06,0.05,0.05,0.04,0.03,0.02,0.02,0.01,0.01])
        date = start_date + timedelta(days=np.random.randint(0, 15), hours=hour, minutes=np.random.randint(0,60))
        
        customer_id = np.random.choice(customer_ids)
        
        # Transaction amount (log-normal distribution)
        amount = np.random.lognormal(mean=4.5, sigma=1.0)  # ~$90 avg
        amount = round(amount, 2)
        
        # Location: simulate that customers usually transact near their home location
        # We'll assign each customer a home location (latitude, longitude) and sometimes they travel
        home_lat = np.random.uniform(40.0, 45.0)
        home_lon = np.random.uniform(-75.0, -70.0)
        
        if np.random.random() < 0.1:  # 10% travel
            location_lat = home_lat + np.random.uniform(-5, 5)
            location_lon = home_lon + np.random.uniform(-5, 5)
        else:
            location_lat = home_lat + np.random.normal(0, 0.5)
            location_lon = home_lon + np.random.normal(0, 0.5)
        
        # Distance from home (km) - approximate
        distance_from_home = np.sqrt((location_lat - home_lat)**2 + (location_lon - home_lon)**2) * 111  # approx km per degree
        
        # Merchant category
        merchant_cats = ['Grocery', 'Restaurant', 'Electronics', 'Clothing', 'Travel', 'Gas', 'Online']
        merchant_category = np.random.choice(merchant_cats)
        
        # Card present? Online transactions have higher fraud risk
        card_present = np.random.choice([0,1], p=[0.3,0.7])
        
        # Previous transaction time for this customer (if any)
        # Simplified: we'll just simulate time since last transaction (hours)
        time_since_last_txn = np.random.exponential(scale=48)  # avg 2 days
        
        # Fraud label: base probability 5%
        fraud = 0
        
        # Generate fraud cases with characteristic patterns
        fraud_prob = 0.05
        # Increase fraud probability for certain patterns
        if amount > 300:
            fraud_prob += 0.1
        if merchant_category == 'Electronics' and amount > 200:
            fraud_prob += 0.15
        if not card_present:
            fraud_prob += 0.1
        if distance_from_home > 100:
            fraud_prob += 0.1
        if time_since_last_txn < 1:  # very quick repeat transaction
            fraud_prob += 0.1
        
        # Normalize fraud probability
        fraud_prob = min(fraud_prob, 0.8)
        
        if np.random.random() < fraud_prob:
            fraud = 1
        
        records.append({
            'transaction_id': transaction_ids[i],
            'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
            'customer_id': customer_id,
            'amount': amount,
            'merchant_category': merchant_category,
            'card_present': card_present,
            'distance_from_home_km': round(distance_from_home, 1),
            'time_since_last_txn_hours': round(time_since_last_txn, 1),
            'hour_of_day': hour,
            'day_of_week': date.weekday(),
            'fraud': fraud
        })
    
    df = pd.DataFrame(records)
    return df

def main():
    import os
    os.makedirs('data', exist_ok=True)
    
    df = generate_transaction_data(50)
    df.to_csv('data/transactions.csv', index=False)
    print(f"Created transactions.csv with {len(df)} rows")
    print(f"Fraud rate: {df['fraud'].mean():.2%}")
    
    print("\nSample of generated data:")
    print(df.head())
    
    print("\nDataset creation complete.")

if __name__ == '__main__':
    main()
