import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from config import (
    OPENWEATHER_API_KEY,
    OPENWEATHER_BASE_URL,
    LATITUDE,
    LONGITUDE,
    AQI_DATA_FILE,
    DATA_DIR,
    START_TIMESTAMP,
    END_TIMESTAMP
)

def fetch_openweather_history():
    start_timestamp = START_TIMESTAMP
    end_timestamp = END_TIMESTAMP
    
    params = {
        'lat': LATITUDE,
        'lon': LONGITUDE,
        'start': start_timestamp,
        'end': end_timestamp,
        'appid': OPENWEATHER_API_KEY
    }
    
    print(f"Fetching data from OpenWeather API...")
    print(f"URL: {OPENWEATHER_BASE_URL}")
    print(f"Parameters: lat={LATITUDE}, lon={LONGITUDE}, start={start_timestamp}, end={end_timestamp}")
    
    response = requests.get(OPENWEATHER_BASE_URL, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    data = response.json()
    print(f"Successfully fetched {len(data.get('list', []))} records")
    
    return data

def process_to_csv(data): 
    if not data or 'list' not in data:
        print("No data to process")
        return False
    
    records = []
    
    for entry in data['list']:
        timestamp = entry.get('dt')
        main_pollutants = entry.get('main', {})
        components = entry.get('components', {})
        
        record = {
            'datetime': datetime.fromtimestamp(timestamp),
            'timestamp': timestamp,
            'aqi': main_pollutants.get('aqi'),
            'co': components.get('co'),
            'no': components.get('no'),
            'no2': components.get('no2'),
            'o3': components.get('o3'),
            'so2': components.get('so2'),
            'pm2_5': components.get('pm2_5'),
            'pm10': components.get('pm10'),
            'nh3': components.get('nh3'),
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df = df.sort_values('datetime').reset_index(drop=True)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(AQI_DATA_FILE, index=False)
    print(f"\n✓ Successfully saved {len(df)} records to {AQI_DATA_FILE}")
    
    print(f"\nData Summary:")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Total records: {len(df)}")
    print(f"\nFirst few records:")
    print(df.head())
    print(f"\nData statistics:")
    print(df[['aqi', 'co', 'no2', 'pm2_5', 'pm10']].describe())
    
    return True

if __name__ == "__main__":
    print("OpenWeather AQI Data Fetcher")
    
    data = fetch_openweather_history()
    
    if data:
        success = process_to_csv(data)
        
        if success:
            print("\n✓ Process completed successfully!")
        else:
            print("\n✗ Failed to process data")
    else:
        print("\n✗ Failed to fetch data from API")
