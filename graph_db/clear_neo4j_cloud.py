"""
Clear Neo4j Cloud Database
===========================
Deletes all nodes and relationships from Neo4j Aura instance.
Use with caution - this is irreversible!

Usage:
    python clear_neo4j_cloud.py
"""

import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from neo4j import GraphDatabase
from src.core.config import settings
from src.core.logger import logger


class Neo4jCleaner:
    def __init__(self):
        """Initialize connection to Neo4j Aura."""
        if not settings.NEO4J_URI:
            raise ValueError("NEO4J_URI not configured in .env file")
        if not settings.NEO4J_PASSWORD:
            raise ValueError("NEO4J_PASSWORD not configured in .env file")
            
        logger.info(f"Connecting to Neo4j at {settings.NEO4J_URI}")
        
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def get_counts(self):
        """Get current node and relationship counts."""
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            # Count nodes
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]
            
            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]
            
            return node_count, rel_count

    def delete_all(self):
        """Delete all nodes and relationships."""
        logger.info("Starting deletion process...")
        
        # Get initial counts
        node_count, rel_count = self.get_counts()
        logger.info(f"Current state: {node_count} nodes, {rel_count} relationships")
        
        if node_count == 0 and rel_count == 0:
            logger.info("✅ Database is already empty")
            return
        
        # Ask for confirmation
        print(f"\n⚠️  WARNING: This will delete:")
        print(f"   - {node_count} nodes")
        print(f"   - {rel_count} relationships")
        print(f"\nDatabase: {settings.NEO4J_URI}")
        print(f"This action is IRREVERSIBLE!\n")
        
        confirmation = input("Type 'DELETE' to confirm: ").strip()
        
        if confirmation != "DELETE":
            logger.info("Deletion cancelled by user")
            return
        
        # Delete in batches to avoid memory issues
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            batch_size = 10000
            deleted = 0
            
            while True:
                result = session.run(f"""
                    MATCH (n)
                    WITH n LIMIT {batch_size}
                    DETACH DELETE n
                    RETURN count(n) as deleted
                """)
                batch_deleted = result.single()["deleted"]
                deleted += batch_deleted
                
                if batch_deleted > 0:
                    logger.info(f"Deleted {deleted} nodes so far...")
                else:
                    break
            
            logger.info(f"✅ Deleted {deleted} nodes total")
        
        # Verify deletion
        node_count, rel_count = self.get_counts()
        logger.info(f"Final state: {node_count} nodes, {rel_count} relationships")
        
        if node_count == 0 and rel_count == 0:
            logger.info("✅ Database cleared successfully!")
        else:
            logger.warning(f"⚠️  Some data remains: {node_count} nodes, {rel_count} relationships")

    def drop_indexes(self):
        """Drop all indexes and constraints."""
        logger.info("Dropping indexes and constraints...")
        
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            # Get all indexes
            indexes_result = session.run("SHOW INDEXES")
            indexes = [record["name"] for record in indexes_result]
            
            for index_name in indexes:
                try:
                    session.run(f"DROP INDEX {index_name} IF EXISTS")
                    logger.info(f"Dropped index: {index_name}")
                except Exception as e:
                    logger.warning(f"Could not drop index {index_name}: {e}")
            
            # Get all constraints
            constraints_result = session.run("SHOW CONSTRAINTS")
            constraints = [record["name"] for record in constraints_result]
            
            for constraint_name in constraints:
                try:
                    session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                    logger.info(f"Dropped constraint: {constraint_name}")
                except Exception as e:
                    logger.warning(f"Could not drop constraint {constraint_name}: {e}")
        
        logger.info("✅ Indexes and constraints dropped")

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Connection closed")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Neo4j Cloud Database Cleaner")
    print("=" * 60)
    
    try:
        cleaner = Neo4jCleaner()
        
        # Delete all data
        cleaner.delete_all()
        
        # Drop indexes
        drop_indexes = input("\nDrop all indexes and constraints? (y/n): ").strip().lower()
        if drop_indexes == 'y':
            cleaner.drop_indexes()
        
        cleaner.close()
        
        print("\n" + "=" * 60)
        print("✅ Cleanup complete!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()