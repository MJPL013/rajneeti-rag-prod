import json
from pathlib import Path
from typing import Generator, List
from src.core.logger import logger
from src.core.exceptions import DataIngestionError
from src.models.schema import NewsArticle, Statement, ArticleEntities

class DataLoader:
    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = raw_data_dir

    def load_data(self) -> Generator[Statement, None, None]:
        """
        Walks through the raw data directory and yields validated Statement objects.
        """
        logger.info(f"Starting data loading from {self.raw_data_dir}")
        
        json_files = list(self.raw_data_dir.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files.")

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Basic validation: check if it's a relevant article
                if not data.get("is_relevant", True) and data.get("is_relevant") is not None:
                     # Some files might explicitly say is_relevant: False
                     if data["is_relevant"] is False:
                         continue

                # Parse Article Level Data
                # Note: We use a loose parsing here because we want to enrich statements
                article_id = data.get("article_id")
                politician = data.get("politician")
                if not article_id or not politician:
                    logger.warning(f"Skipping {file_path}: Missing article_id or politician")
                    continue

                # Extract Entities
                raw_entities = data.get("article_entities", {})
                entities = ArticleEntities(
                    persons=raw_entities.get("persons", []),
                    organizations=raw_entities.get("organizations", []),
                    locations=raw_entities.get("locations", []),
                    policies_schemes=raw_entities.get("policies_schemes", [])
                )

                # Process Statements
                statements_list = data.get("statements", [])
                for idx, stmt_data in enumerate(statements_list):
                    try:
                        statement_obj = Statement(
                            statement=stmt_data.get("statement", ""),
                            summary=stmt_data.get("summary", ""),
                            weight=stmt_data.get("weight", 1.0),
                            theme=stmt_data.get("theme", []),
                            classification=stmt_data.get("classification", "Unknown"),
                            temporal_focus=stmt_data.get("temporal_focus", "Unknown"),
                            content_type=stmt_data.get("content_type", "Unknown"),
                            perspective=stmt_data.get("perspective", "Unknown"),
                            sentiment=stmt_data.get("sentiment", "Unknown"),
                            
                            # Enriched Metadata
                            article_id=article_id,
                            statement_index=idx,
                            politician=politician,
                            source=data.get("source", "Unknown"),
                            publish_date=str(data.get("publish_date", "")),
                            url=data.get("url", ""),
                            title=data.get("title", ""),
                            is_relevant=True, # Assumed true if we are processing it
                            article_entities=entities
                        )
                        yield statement_obj
                    except Exception as e:
                        logger.error(f"Error parsing statement in file {file_path}: {e}")
                        continue

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON file: {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
