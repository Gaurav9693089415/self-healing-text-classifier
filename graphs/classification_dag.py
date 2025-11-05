"""
classification_dag.py
---------------------
Builds the LangGraph workflow for the Self-Healing Classification System.

Pipeline:
1. InferenceNode         → Base model prediction
2. ConfidenceCheckNode   → If confidence < threshold → fallback
3. FallbackNode          → User correction or backup model
4. FinalizeNode          → Print/log final decision
"""

from langgraph.graph import StateGraph, END
from utils.config import CONFIDENCE_THRESHOLD

from nodes.inference import InferenceNode
from nodes.confidence import ConfidenceCheckNode
from nodes.fallback import FallbackNode
from nodes.finalize import FinalizeNode


def build_classification_graph(model_path, use_backup=False, backup_classifier=None):
    """Build and compile the LangGraph DAG."""

    graph = StateGraph(dict)

    # Node initialization
    inference = InferenceNode(model_path)
    confidence = ConfidenceCheckNode(threshold=CONFIDENCE_THRESHOLD)
    fallback = FallbackNode(use_backup=use_backup, backup_classifier=backup_classifier, threshold=CONFIDENCE_THRESHOLD)
    finalize = FinalizeNode()

    # Register nodes
    graph.add_node("Inference", inference)
    graph.add_node("ConfidenceCheck", confidence)
    graph.add_node("Fallback", fallback)
    graph.add_node("Finalize", finalize)

    graph.set_entry_point("Inference")
    graph.add_edge("Inference", "ConfidenceCheck")

    # Routing logic (NO extra prints here)
    def route(state):
        return "Fallback" if state["confidence"] < CONFIDENCE_THRESHOLD else "Finalize"

    graph.add_conditional_edges("ConfidenceCheck", route, {"Fallback": "Fallback", "Finalize": "Finalize"})

    graph.add_edge("Fallback", "Finalize")
    graph.add_edge("Finalize", END)

    return graph.compile()
