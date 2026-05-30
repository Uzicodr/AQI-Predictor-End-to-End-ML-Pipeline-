import pandas as pd
from pymongo import MongoClient
from config import MONGODB_URI, MONGODB_DB, AQI_DATA_FILE
from feature_pipeline import FEATURE_FILE
from features import build_feature_frame
import sys


def upload_to_mongodb():
    
    if not MONGODB_URI:
        print("✗ MONGODB_URI not set in .env file")
        print("  Please add your MongoDB connection string to .env")
        return False
    
    print("MongoDB Backfill - AQI Data Upload")
    
    print(f"\nReading CSV file: {AQI_DATA_FILE}")
    try:
        df = pd.read_csv(AQI_DATA_FILE)
        print(f"✓ Successfully read {len(df)} records from CSV")
    except FileNotFoundError:
        print(f"✗ CSV file not found: {AQI_DATA_FILE}")
        print("  Please run fetch_aqi_data.py first to generate the CSV")
        return False
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return False
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"\nConnecting to MongoDB...")
    try:
        client = MongoClient(MONGODB_URI, connectTimeoutMS=5000, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✓ Successfully connected to MongoDB")
    except Exception as e:
        print(f"✗ Failed to connect to MongoDB: {e}")
        print("  Please check your MONGODB_URI in .env file")
        return False
    
    try:
        db = client[MONGODB_DB]
        collection = db["aqi_data"]
        feature_collection = db["feature_store"]
        
        print(f"\nDatabase: {MONGODB_DB}")
        print(f"Collection: aqi_data")
        
        print(f"\nClearing existing data...")
        result = collection.delete_many({})
        print(f"✓ Deleted {result.deleted_count} existing records")
        
        records = df.to_dict('records')
        
        print(f"\nUploading {len(records)} records to MongoDB...")
        result = collection.insert_many(records)
        print(f"✓ Successfully inserted {len(result.inserted_ids)} records")
        
        print(f"\nData Summary:")
        print(f"  - Total records: {collection.count_documents({})}")
        print(f"  - Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  - AQI values: {df['aqi'].min():.0f} - {df['aqi'].max():.0f}")
        print(f"  - PM2.5 mean: {df['pm2_5'].mean():.2f} μg/m³")
        print(f"  - PM10 mean: {df['pm10'].mean():.2f} μg/m³")
        
        print(f"\nCreating index on timestamp...")
        collection.create_index("timestamp")
        print("✓ Index created successfully")
        
        print("\nComputing engineered features for feature_store...")
        feature_df = build_feature_frame(df, include_future_target=True)
        feature_collection.delete_many({})
        feature_records = feature_df.to_dict("records")
        if feature_records:
            feature_collection.insert_many(feature_records)
        feature_collection.create_index("datetime")
        feature_collection.create_index("target_aqi_72h")
        FEATURE_FILE.parent.mkdir(exist_ok=True)
        feature_df.to_csv(FEATURE_FILE, index=False)
        print(f"Stored {len(feature_records)} engineered rows in MongoDB feature_store")
        print(f"Feature CSV written to {FEATURE_FILE}")

        print(f"\nBackfill completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during upload: {e}")
        return False
    
    finally:
        client.close()
        print("✓ MongoDB connection closed")


if __name__ == "__main__":
    success = upload_to_mongodb()
    sys.exit(0 if success else 1)
