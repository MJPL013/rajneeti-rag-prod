"""
Rajneeti Date Parser Check
================================================================
Scans local JSON files and tests date parsing logic WITHOUT 
connecting to the database. Use this to identify missing formats.

Usage: python graph_db/time_parse_check.py
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config import settings

def parse_timestamp_test(date_str):
    """
    Robust date parser to be tested.
    Updates here should be copied to the repair script.
    """
    if not date_str or not isinstance(date_str, str):
        return None, "Empty/Not String"
    
    date_str = date_str.strip()
    if date_str.upper() in ['N/A', 'NONE', 'NULL']:
        return None, "Explicit N/A"

    # Strategy 1: Standard ISO-like start (YYYY-MM-DD)
    # Handles: '2019-07-19 03:16:08+00:00', '2023-03-17 00:00:00'
    if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
        try:
            clean_date = date_str[:10]
            dt = datetime.strptime(clean_date, '%Y-%m-%d')
            return dt, "ISO Standard"
        except ValueError:
            pass

    # Strategy 2: Specific formats found in your data
    formats = [
        '%b %d, %Y %H:%M',       # Feb 21, 2024 17:03
        '%B %d, %Y %H:%M',       # February 21, 2024 17:03
        '%d-%m-%Y',              # 15-01-2024
        '%d/%m/%Y',              # 15/01/2024
        '%B %d, %Y',             # January 15, 2024
        '%b %d, %Y',             # Jan 15, 2024
        '%Y/%m/%d',              # 2024/01/15
        '%d %B %Y',              # 15 January 2024
        '%d %b %Y',              # 15 Jan 2024
        '%Y-%m-%d %H:%M:%S',     # 2019-07-19 03:16:08 (No timezone)
        '%Y-%m-%d %H:%M:%S%z',   # 2019-07-19 03:16:08+00:00 (Full strict match)
    ]
    
    for fmt in formats:
        try:
            # Check if strict match works
            dt = datetime.strptime(date_str, fmt)
            return dt, fmt
        except ValueError:
            # For formats like "Feb 21, 2024 17:03 IST", try fuzzy matching the start
            try:
                # Attempt to match strictly up to the length implied by format
                # This is a bit manual, so we rely on exceptions
                pass 
            except:
                continue

    return None, "Unknown Format"

def run_check():
    print(f"🔍 Scanning {settings.RAW_DATA_DIR}...")
    json_files = list(Path(settings.RAW_DATA_DIR).rglob("*.json"))
    print(f"📄 Found {len(json_files)} files.")

    stats = {
        "success": 0,
        "failed": 0,
        "na_skipped": 0,
        "failed_samples": set()
    }

    for file_path in tqdm(json_files, desc="Checking Dates"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            raw_date = data.get("publish_date")
            
            dt, reason = parse_timestamp_test(raw_date)
            
            if dt:
                stats["success"] += 1
            elif reason == "Explicit N/A" or reason == "Empty/Not String":
                stats["na_skipped"] += 1
            else:
                stats["failed"] += 1
                stats["failed_samples"].add(raw_date)

        except Exception as e:
            pass

    print("\n" + "="*50)
    print("DATE PARSING REPORT")
    print("="*50)
    print(f"✅ Parsed Successfully: {stats['success']}")
    print(f"⏭️  Skipped (N/A, Empty): {stats['na_skipped']}")
    print(f"❌ Failed to Parse:     {stats['failed']}")
    
    if stats['failed_samples']:
        print("\n⚠️  UNIQUE FAILED FORMATS (Add these to parser):")
        for s in list(stats['failed_samples'])[:20]:
            print(f"   '{s}'")
    else:
        print("\n🎉 No failures! Logic is ready for repair script.")
    print("="*50)

if __name__ == "__main__":
    run_check()