import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import sys
from config import MONGODB_URI, MONGODB_DB, AQI_DATA_FILE, DATA_DIR

def train_model():
    
    if not MONGODB_URI:
        print("MONGODB_URI not set in .env file")
        return False
    
    print("Training AQI Prediction Model")
    
    print(f"\nConnecting to MongoDB...")
    try:
        client = MongoClient(MONGODB_URI, connectTimeoutMS=5000, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("Successfully connected to MongoDB")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        print("Please check your MONGODB_URI in .env file")
        return False
    
    try:
        db = client[MONGODB_DB]
        collection = db["aqi_data"]
        
        print(f"Database: {MONGODB_DB}")
        print(f"Collection: aqi_data")
        
        print(f"\nFetching data from MongoDB...")
        records = list(collection.find())
        
        if len(records) == 0:
            print("No data found in MongoDB")
            print("Please run backfill.py first")
            return False
        
        print(f"Retrieved {len(records)} records from MongoDB")
        
        df = pd.DataFrame(records)
        
        print(f"\nData shape: {df.shape}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(AQI_DATA_FILE, index=False)
        print(f"Updated CSV file: {AQI_DATA_FILE}")
        
        print(f"\nPreparing features...")
        features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        target = 'aqi'
        
        df_clean = df[features + [target]].dropna()
        
        print(f"Clean data shape: {df_clean.shape}")
        
        X = df_clean[features]
        y = df_clean[target]
        
        print(f"\nFeatures: {features}")
        print(f"Target: {target}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled, y)
        
        print(f"Model trained successfully")
        print(f"Model score: {model.score(X_scaled, y):.4f}")
        
        model_path = DATA_DIR / "model.pkl"
        scaler_path = DATA_DIR / "scaler.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        
        print(f"\nFeature importance:")
        for feature, importance in zip(features, model.feature_importances_):
            print(f"  - {feature}: {importance:.4f}")
        
        print(f"\nTraining completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        client.close()
        print("MongoDB connection closed")


if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
