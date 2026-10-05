"""Public query interface for the static clinical knowledge base.

The implementation currently resides in src.knowledge and is re-exported here.
"""

from src.knowledge import ClinicalKnowledgeBase, KnowledgeRetrieval, render_knowledge

__all__ = ["ClinicalKnowledgeBase", "KnowledgeRetrieval", "render_knowledge"]
