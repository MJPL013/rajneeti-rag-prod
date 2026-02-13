import json
from pathlib import Path
from collections import Counter
import pandas as pd
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.ingestion.loader import DataLoader

def profile_raw_data():
    print(f"Scanning data in: {settings.RAW_DATA_DIR}")
    
    # We'll use your existing loader to ensure we're reading the same way the app does
    loader = DataLoader(settings.RAW_DATA_DIR)
    
    sources_per_politician = {}
    unique_politician_names = Counter()
    total_articles = 0
    
    # We walk the directory to get article-level info directly
    json_files = list(settings.RAW_DATA_DIR.rglob("*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract basic article metadata
            pol = data.get("politician", "Unknown")
            src = data.get("source", "Unknown")
            
            unique_politician_names[pol] += 1
            
            if pol not in sources_per_politician:
                sources_per_politician[pol] = Counter()
            
            sources_per_politician[pol][src] += 1
            total_articles += 1
            
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    # --- Print Results ---
    print("\n" + "="*50)
    print("RAJNEETI DATA PROFILE")
    print("="*50)
    print(f"Total JSON Articles Found: {total_articles}")
    
    print("\n1. Detected Politician Names (in JSON 'politician' field):")
    for name, count in unique_politician_names.items():
        print(f"   - {name}: {count} articles")

    print("\n2. Media Sources per Politician:")
    for pol, sources in sources_per_politician.items():
        print(f"\n--- {pol} ---")
        for src, count in sources.most_common():
            print(f"   - {src}: {count}")

    print("\n3. Overall Unique Sources Found:")
    all_sources = set()
    for sources in sources_per_politician.values():
        all_sources.update(sources.keys())
    for s in sorted(all_sources):
        print(f"   - {s}")
    print("="*50)

if __name__ == "__main__":
    profile_raw_data()