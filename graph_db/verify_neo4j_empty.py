"""
Verify Neo4j Database Empty
============================
Checks if Neo4j Aura database is empty and ready for ingestion.
Reports node counts, relationship counts, indexes, and constraints.

Usage:
    python verify_neo4j_empty.py
"""

import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from neo4j import GraphDatabase
from src.core.config import settings
from src.core.logger import logger
from tabulate import tabulate


class Neo4jVerifier:
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

    def get_node_counts(self):
        """Get counts of nodes by label."""
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """)
            return [(record["label"], record["count"]) for record in result]

    def get_relationship_counts(self):
        """Get counts of relationships by type."""
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """)
            return [(record["type"], record["count"]) for record in result]

    def get_total_counts(self):
        """Get total node and relationship counts."""
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            # Total nodes
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]
            
            # Total relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]
            
            return node_count, rel_count

    def get_indexes(self):
        """Get all indexes."""
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run("SHOW INDEXES")
            return [(record["name"], record["type"], record["state"]) for record in result]

    def get_constraints(self):
        """Get all constraints."""
        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run("SHOW CONSTRAINTS")
            return [(record["name"], record["type"]) for record in result]

    def verify(self):
        """Run complete verification and report results."""
        print("\n" + "=" * 80)
        print("NEO4J DATABASE VERIFICATION REPORT")
        print("=" * 80)
        print(f"\nDatabase: {settings.NEO4J_URI}")
        print(f"Database Name: {settings.NEO4J_DATABASE}")
        print(f"Username: {settings.NEO4J_USERNAME}")
        
        # Total counts
        print("\n" + "-" * 80)
        print("TOTAL COUNTS")
        print("-" * 80)
        node_count, rel_count = self.get_total_counts()
        print(f"Total Nodes: {node_count}")
        print(f"Total Relationships: {rel_count}")
        
        # Node counts by label
        print("\n" + "-" * 80)
        print("NODES BY LABEL")
        print("-" * 80)
        node_counts = self.get_node_counts()
        if node_counts:
            print(tabulate(node_counts, headers=["Label", "Count"], tablefmt="grid"))
        else:
            print("No nodes found")
        
        # Relationship counts by type
        print("\n" + "-" * 80)
        print("RELATIONSHIPS BY TYPE")
        print("-" * 80)
        rel_counts = self.get_relationship_counts()
        if rel_counts:
            print(tabulate(rel_counts, headers=["Type", "Count"], tablefmt="grid"))
        else:
            print("No relationships found")
        
        # Indexes
        print("\n" + "-" * 80)
        print("INDEXES")
        print("-" * 80)
        indexes = self.get_indexes()
        if indexes:
            print(tabulate(indexes, headers=["Name", "Type", "State"], tablefmt="grid"))
        else:
            print("No indexes found")
        
        # Constraints
        print("\n" + "-" * 80)
        print("CONSTRAINTS")
        print("-" * 80)
        constraints = self.get_constraints()
        if constraints:
            print(tabulate(constraints, headers=["Name", "Type"], tablefmt="grid"))
        else:
            print("No constraints found")
        
        # Final verdict
        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        
        if node_count == 0 and rel_count == 0:
            print("DATABASE IS EMPTY - READY FOR INGESTION")
            is_empty = True
        else:
            print(f"DATABASE IS NOT EMPTY")
            print(f"   Found {node_count} nodes and {rel_count} relationships")
            print(f"\n   Run 'python clear_neo4j_cloud.py' to clear the database")
            is_empty = False
        
        print("=" * 80 + "\n")
        
        return is_empty

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Connection closed")


def main():
    """Main execution function."""
    try:
        verifier = Neo4jVerifier()
        is_empty = verifier.verify()
        verifier.close()
        
        # Exit with appropriate code
        sys.exit(0 if is_empty else 1)
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()