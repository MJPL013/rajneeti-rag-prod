"""
Rajneeti Graph Repair - DB Updater
================================================================
Applies the verified date parsing logic to Neo4j.
Updates Articles and re-links TimeFrames.

UPDATES:
- Added URL-based date extraction as fallback.
- Added explicit logging for N/A values.
- Implemented 'Unknown Date' TimeFrame for truly dateless articles.

Usage: python graph_db/time_repair_script.py
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from neo4j import GraphDatabase
from src.core.config import settings

class GraphDateRepair:
    def __init__(self, batch_size=200):
        self.batch_size = batch_size
        self.stats = {"updated": 0, "errors": 0, "skipped": 0, "url_fallback": 0, "unknown_assigned": 0}
        self.skipped_samples = []
        self._init_neo4j()

    def _init_neo4j(self):
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI, 
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            print("✅ Connected to Neo4j Aura")
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            sys.exit(1)

    def extract_date_from_url(self, url):
        """
        Tries to find YYYY/MM/DD or YYYY-MM-DD pattern in URL.
        """
        if not url or not isinstance(url, str):
            return None
        
        # Regex 1: /2023/05/21/ or /2023-05-21/
        match = re.search(r'/(\d{4})[-/](\d{2})[-/](\d{2})', url)
        if match:
            y, m, d = match.groups()
            try:
                dt = datetime(int(y), int(m), int(d))
                return self._fmt_response(dt, dt.strftime('%Y-%m-%d'))
            except ValueError:
                pass
        
        # Regex 2: /2023/05/ (Year/Month only) -> Default to 1st of month
        match = re.search(r'/(\d{4})[-/](\d{2})/', url)
        if match:
            y, m = match.groups()
            try:
                dt = datetime(int(y), int(m), 1)
                return self._fmt_response(dt, dt.strftime('%Y-%m-%d'))
            except ValueError:
                pass

        return None

    def parse_timestamp(self, date_str, url=None):
        """
        Logic: 
        1. Try parsing date_str
        2. If invalid/empty/N/A, try extracting from URL
        3. If both fail, assign a placeholder 'Unknown' date (2000-01-01)
        """
        # 1. Try standard parsing
        parsed = self._parse_date_string(date_str)
        if parsed:
            return parsed

        # 2. Try URL fallback
        if url:
            parsed = self.extract_date_from_url(url)
            if parsed:
                self.stats["url_fallback"] += 1
                return parsed
        
        # 3. Last Resort: Placeholder for "N/A" cases so they aren't lost
        self.stats["unknown_assigned"] += 1
        dt = datetime(2000, 1, 1) # Placeholder date
        return {
            'year': 2000,
            'month': 1,
            'quarter': 'Unknown',
            'year_month': 'Unknown',
            'date_str': '2000-01-01'
        }

    def _parse_date_string(self, date_str):
        if not date_str or not isinstance(date_str, str):
            return None
        
        date_str = date_str.strip()
        if not date_str or date_str.upper() in ['N/A', 'NONE', 'NULL', '']:
            return None

        # Strategy 1: ISO Start
        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
            try:
                clean_date = date_str[:10]
                dt = datetime.strptime(clean_date, '%Y-%m-%d')
                return self._fmt_response(dt, clean_date)
            except ValueError:
                pass

        # Strategy 2: Formats list
        formats = [
            '%b %d, %Y %H:%M',       # Feb 21, 2024 17:03
            '%B %d, %Y %H:%M',       # February 21, 2024 17:03
            '%d-%m-%Y', '%d/%m/%Y', '%B %d, %Y', 
            '%b %d, %Y', '%Y/%m/%d', '%d %B %Y', '%d %b %Y',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return self._fmt_response(dt, dt.strftime('%Y-%m-%d'))
            except ValueError:
                continue
        
        return None

    def _fmt_response(self, dt, date_str):
        return {
            'year': dt.year,
            'month': dt.month,
            'quarter': f'Q{(dt.month-1)//3 + 1}',
            'year_month': dt.strftime('%Y-%m'),
            'date_str': date_str
        }

    def update_batch(self, tx, updates):
        query = """
        UNWIND $updates AS row
        WITH row
        WHERE row.date_str IS NOT NULL
        
        MATCH (a:Article {article_id: row.article_id})
        
        // Remove old rel
        OPTIONAL MATCH (a)-[r1:PUBLISHED_IN]->(:TimeFrame)
        DELETE r1
        
        // Merge new TimeFrame
        MERGE (tf:TimeFrame {year_month: row.year_month})
        SET tf.year = row.year, 
            tf.month = row.month, 
            tf.quarter = row.quarter
            
        // Link Article
        SET a.publish_date = date(row.date_str)
        MERGE (a)-[:PUBLISHED_IN]->(tf)
        
        // Link Statements
        WITH a, tf
        MATCH (a)-[:CONTAINS_STATEMENT]->(s:Statement)
        OPTIONAL MATCH (s)-[r2:PUBLISHED_IN]->(:TimeFrame)
        DELETE r2
        MERGE (s)-[:PUBLISHED_IN]->(tf)
        """
        tx.run(query, updates=updates)

    def run_repair(self):
        print(f"🔍 Scanning files...")
        json_files = list(Path(settings.RAW_DATA_DIR).rglob("*.json"))
        
        batch = []
        for file_path in tqdm(json_files, desc="Repairing DB"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                aid = data.get("article_id")
                raw_date = data.get("publish_date")
                url = data.get("url")
                
                if not aid: continue

                date_info = self.parse_timestamp(raw_date, url)
                
                if date_info:
                    batch.append({
                        "article_id": aid,
                        **date_info
                    })
                else:
                    self.stats["skipped"] += 1
                    if len(self.skipped_samples) < 10:
                        self.skipped_samples.append({"id": aid, "date": raw_date, "url": url})

                if len(batch) >= self.batch_size:
                    with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                        session.execute_write(self.update_batch, batch)
                    self.stats["updated"] += len(batch)
                    batch = []

            except Exception:
                self.stats["errors"] += 1

        if batch:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                session.execute_write(self.update_batch, batch)
            self.stats["updated"] += len(batch)

        self.rechain_timeframes()
        self.driver.close()
        
        print(f"\n✅ Updated: {self.stats['updated']}")
        print(f"🔗 Recovered via URL: {self.stats['url_fallback']}")
        print(f"👻 Unknown Assigned: {self.stats['unknown_assigned']}")
        print(f"⏭️ Skipped: {self.stats['skipped']}")

    def rechain_timeframes(self):
        print("🔗 Linking TimeFrames (NEXT_PERIOD)...")
        query = """
        MATCH (:TimeFrame)-[r:NEXT_PERIOD]->(:TimeFrame) DELETE r
        WITH 1 as x
        MATCH (tf:TimeFrame) 
        WITH tf ORDER BY tf.year_month ASC
        WITH collect(tf) as frames
        FOREACH (i in range(0, size(frames)-2) |
            FOREACH (f1 in [frames[i]] |
                FOREACH (f2 in [frames[i+1]] |
                    MERGE (f1)-[:NEXT_PERIOD]->(f2))))
        """
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run(query)

if __name__ == "__main__":
    GraphDateRepair().run_repair()