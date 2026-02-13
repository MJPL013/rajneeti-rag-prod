"""
Rajneeti RAG - Graph Ingestion V3
=================================
Hybrid version:
- Temporal chain support (NEXT_PERIOD)
- Strong normalization
- Validation layer
- Relationship tracking
- Date parse monitoring
- Lean metadata (NO credibility_score / language)
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Set
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.core.config import settings
from src.core.logger import logger


# ============================================================
# SOURCE METADATA (LEAN)
# ============================================================

SOURCE_METADATA = {
    "theprint.in": {"leaning": "center-right", "region": "National"},
    "ndtv": {"leaning": "center-left", "region": "National"},
    "scroll.in": {"leaning": "left", "region": "National"},
    "thewire.in": {"leaning": "left", "region": "National"},
    "indianexpress": {"leaning": "center", "region": "National"},
    "thehindu": {"leaning": "left", "region": "National"},
    "deccanherald": {"leaning": "center", "region": "South India"},
    "Unknown": {"leaning": "unknown", "region": "Unknown"},
}


# ============================================================
# DATE PARSER
# ============================================================

def robust_date_parser(date_str: str) -> Optional[Dict[str, Any]]:
    if not date_str or not isinstance(date_str, str):
        return None

    formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
        "%B %d, %Y", "%b %d, %Y",
        "%Y/%m/%d", "%d %B %Y", "%d %b %Y"
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return {
                "year": dt.year,
                "month": dt.month,
                "quarter": f"Q{(dt.month - 1)//3 + 1}",
                "year_month": dt.strftime("%Y-%m"),
                "date_str": dt.strftime("%Y-%m-%d"),
            }
        except ValueError:
            continue

    return None


# ============================================================
# VALIDATION
# ============================================================

def validate_article(data: Dict[str, Any]):
    if not data:
        return False, "Empty data"

    if not data.get("article_id"):
        return False, "Missing article_id"

    if not data.get("politician"):
        return False, "Missing politician"

    if data.get("is_relevant") is False:
        return False, "Marked irrelevant"

    return True, None


# ============================================================
# MAIN CLASS
# ============================================================

class RajneetiGraphIngestorV3:

    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self.checkpoint_file = Path("ingestion_checkpoint_v3.json")
        self.processed_articles: Set[str] = set()

        self.stats = {
            "articles_processed": 0,
            "articles_skipped": 0,
            "statements_created": 0,
            "timeframes_created": 0,
            "date_parse_failures": 0,
            "errors": 0,
            "relationships_created": {
                "NEXT_PERIOD": 0
            }
        }

        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        )
        self.embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            trust_remote_code=True
        )

        self._load_checkpoint()

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def _load_checkpoint(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
                self.processed_articles = set(data.get("processed_articles", []))

    def _save_checkpoint(self):
        with open(self.checkpoint_file, "w") as f:
            json.dump({
                "processed_articles": list(self.processed_articles),
                "stats": self.stats
            }, f, indent=2)

    # ========================================================
    # SCHEMA
    # ========================================================

    def create_schema(self):
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run("""
                CREATE CONSTRAINT article_id_unique IF NOT EXISTS
                FOR (a:Article) REQUIRE a.article_id IS UNIQUE
            """)

            session.run("""
                CREATE INDEX politician_name_idx IF NOT EXISTS
                FOR (p:Politician) ON (p.name)
            """)

            session.run("""
                CREATE INDEX timeframe_year_month_idx IF NOT EXISTS
                FOR (tf:TimeFrame) ON (tf.year_month)
            """)

            session.run("""
                CREATE INDEX source_leaning_idx IF NOT EXISTS
                FOR (s:Source) ON (s.political_leaning)
            """)

            dim = len(self.embedding_model.encode("test"))
            session.run(f"""
                CREATE VECTOR INDEX statement_embeddings IF NOT EXISTS
                FOR (s:Statement) ON (s.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """)

        logger.info("Schema initialized.")

    # ========================================================
    # INGEST
    # ========================================================

    def ingest_article(self, tx, article):

        valid, reason = validate_article(article)
        if not valid:
            self.stats["articles_skipped"] += 1
            logger.warning(f"Skipping article: {reason}")
            return

        aid = article["article_id"]

        if aid in self.processed_articles:
            return

        date_info = robust_date_parser(article.get("publish_date", ""))
        if not date_info:
            self.stats["date_parse_failures"] += 1
            date_info = {
                "year": 2024,
                "month": 1,
                "quarter": "Q1",
                "year_month": "2024-01",
                "date_str": "2024-01-01"
            }

        source_key = article.get("source", "Unknown")
        meta = SOURCE_METADATA.get(source_key, SOURCE_METADATA["Unknown"])

        tx.run("""
            MERGE (s:Source {name: $source})
            SET s.political_leaning = $lean,
                s.region = $region

            MERGE (p:Politician {name: $politician})

            MERGE (tf:TimeFrame {year_month: $ym})
            SET tf.year = $year,
                tf.month = $month,
                tf.quarter = $quarter

            MERGE (a:Article {article_id: $aid})
            SET a.title = $title,
                a.url = $url,
                a.publish_date = date($date)

            MERGE (a)-[:PUBLISHED_BY]->(s)
            MERGE (a)-[:FOCUSES_ON]->(p)
            MERGE (a)-[:PUBLISHED_IN]->(tf)
        """,
        source=source_key,
        lean=meta["leaning"],
        region=meta["region"],
        politician=article["politician"],
        ym=date_info["year_month"],
        year=date_info["year"],
        month=date_info["month"],
        quarter=date_info["quarter"],
        aid=aid,
        title=article.get("title", ""),
        url=article.get("url", ""),
        date=date_info["date_str"]
        )

        # Statements
        for idx, stmt in enumerate(article.get("statements", [])):
            text = stmt.get("statement")
            if not text:
                continue

            embedding = self.embedding_model.encode(text).tolist()
            sid = f"{aid}_s{idx}"

            tx.run("""
                MERGE (s:Statement {statement_id: $sid})
                SET s.text = $text,
                    s.embedding = $embedding
                WITH s
                MATCH (a:Article {article_id: $aid})
                MERGE (a)-[:CONTAINS_STATEMENT]->(s)
            """,
            sid=sid,
            text=text,
            embedding=embedding,
            aid=aid)


            self.stats["statements_created"] += 1

        self.processed_articles.add(aid)
        self.stats["articles_processed"] += 1

    # ========================================================
    # TEMPORAL CHAIN
    # ========================================================

    def create_temporal_chain(self):
        query = """
        MATCH (tf:TimeFrame)
        WITH tf ORDER BY tf.year_month
        WITH collect(tf) AS frames
        FOREACH (i IN range(0, size(frames)-2) |
            FOREACH (f1 IN [frames[i]] |
                FOREACH (f2 IN [frames[i+1]] |
                    MERGE (f1)-[:NEXT_PERIOD]->(f2)
                )
            )
        )
        """
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run(query)

        self.stats["relationships_created"]["NEXT_PERIOD"] += 1
        logger.info("Temporal chain created.")

    # ========================================================
    # RUN
    # ========================================================

    def run(self):
        self.create_schema()

        json_files = list(Path(settings.RAW_DATA_DIR).rglob("*.json"))
        logger.info(f"Found {len(json_files)} files.")

        batch = []

        for file_path in tqdm(json_files):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                batch.append(data)

            if len(batch) >= self.batch_size:
                with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                    for article in batch:
                        session.execute_write(self.ingest_article, article)
                batch = []
                self._save_checkpoint()

        if batch:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                for article in batch:
                    session.execute_write(self.ingest_article, article)

        self.create_temporal_chain()
        self._save_checkpoint()

        logger.info(f"Ingestion complete. Stats: {self.stats}")

    def close(self):
        self.driver.close()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    ingestor = RajneetiGraphIngestorV3(batch_size=100)
    try:
        ingestor.run()
    finally:
        ingestor.close()
