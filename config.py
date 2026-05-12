import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

LATITUDE = 24.8607
LONGITUDE = 67.0011
START_TIMESTAMP = 1738368000
END_TIMESTAMP = 1746057600
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = "aqi_predictor"

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
AQI_DATA_FILE = DATA_DIR / "aqi_data.csv"

def validate_openweather_config() -> bool:
    if not OPENWEATHER_API_KEY:
        print("OPENWEATHER_API_KEY not set in environment")
        return False
    return True

def validate_mongodb_config() -> bool:
    if not MONGODB_URI:
        print("MONGODB_URI not set in environment")
        print("  Set it to your MongoDB connection string")
        print("  Get it from https://www.mongodb.com/cloud/atlas")
        return False
    return True


def validate_all_config() -> bool:
    return validate_openweather_config() and validate_mongodb_config()

def print_config():
    print("AQI PREDICTOR CONFIGURATION")
    
    print("\n OpenWeather API Configuration:")
    print(f"  - API Key: {'✓ Set' if OPENWEATHER_API_KEY else '✗ Not set'}")
    print(f"  - Location: ({LATITUDE}, {LONGITUDE})")
    print(f"  - Base URL: {OPENWEATHER_BASE_URL}")
    
    print("\n MongoDB Configuration:")
    print(f"  - Connection: {'Set' if MONGODB_URI else '✗ Not set'}")
    print(f"  - Database: {MONGODB_DB}")
    
    print("\n Data Configuration:")
    print(f"  - Project Root: {PROJECT_ROOT}")
    print(f"  - Data Directory: {DATA_DIR}")
    print(f"  - AQI Data File: {AQI_DATA_FILE}")


if __name__ == "__main__":
    print_config()
    
    print("Configuration Status:")
    if validate_all_config():
        print("✓ All configurations are valid!")
    else:
        print("✗ Some configurations are missing. Please set them up in .env file.")
