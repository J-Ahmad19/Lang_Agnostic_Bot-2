import csv
import os
from datetime import datetime
from config import LOG_FILE

def initialize_logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "user_query", "detected_language", "bot_response"])

def log_interaction(query: str, language: str, response: str):
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, query, language, response])
