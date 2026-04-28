"""
M2 Action Router — Central Dispatcher for retrieval_input from m3 Pipeline.

The m3 Memory Pipeline classifies every user message into 1 of 8 labels
and builds a structured retrieval_input object. This router dispatches
to the correct handler based on the 'action' field.

Supported actions:
    - catalog_search        (INITIAL_REQUEST, REFINEMENT)
    - item_attribute_lookup (ATTRIBUTE_QUESTION)
    - item_compare          (COMPARISON)
    - explanation_generate  (EXPLANATION_WHY)
    - item_detail_lookup    (SELECTION_REFERENCE)
    - None                  (FEEDBACK, CHITCHAT)
"""

from m2_multimodal_rag.m2_handlers import (
    handle_catalog_search,
    handle_attribute_lookup,
    handle_item_compare,
    handle_explanation_generate,
    handle_item_detail_lookup,
    handle_no_retrieval,
)


class M2ActionRouter:
    """
    Master orchestrator for M2 Multimodal RAG.
    Receives structured retrieval_input from the m3 adaptive trigger pipeline
    and routes to the appropriate action handler.
    """

    # Valid action types from the retrieval_input reference spec
    VALID_ACTIONS = {
        "catalog_search",
        "item_attribute_lookup",
        "item_compare",
        "explanation_generate",
        "item_detail_lookup",
    }

    def __init__(self):
        print("M2 Router: Action Router initialized. Ready to receive retrieval_input from m3 pipeline.")

    def process_retrieval_input(self, retrieval_input: dict | None, memory_context: dict | None = None) -> dict:
        """
        Main entry point — called by the FastAPI layer after m3 pipeline produces
        the retrieval_input and memory_context.

        Args:
            retrieval_input: The structured retrieval_input dict from m3, or None
                             for FEEDBACK/CHITCHAT labels.
            memory_context:  The memory_context dict from m3 containing user prefs,
                             dialogue state, feedback data, etc.

        Returns:
            dict with keys:
                - action (str): The action that was executed
                - success (bool): Whether the action completed successfully
                - response_text (str): Natural language response for the user
                - items (list[dict]): Recommended/fetched items (if applicable)
                - error (str|None): Error message if something failed
        """
        if memory_context is None:
            memory_context = {}

        # ---------------------------------------------------------------
        # CASE 1: retrieval_input is None → FEEDBACK or CHITCHAT
        # ---------------------------------------------------------------
        if retrieval_input is None:
            print("\n--- M2 Router: No retrieval needed (FEEDBACK/CHITCHAT) ---")
            return handle_no_retrieval(memory_context)

        # ---------------------------------------------------------------
        # CASE 2: retrieval_input has an action → route to handler
        # ---------------------------------------------------------------
        action = retrieval_input.get("action")
        user_message = retrieval_input.get("user_message", "")

        if action not in self.VALID_ACTIONS:
            print(f"M2 Router: [ERROR] Unknown action '{action}'")
            return {
                "action": action,
                "success": False,
                "response_text": "I'm sorry, I couldn't understand that request.",
                "items": [],
                "error": f"Unknown action type: {action}",
            }

        print(f"\n--- M2 Router: Dispatching action='{action}' ---")
        print(f"    User message: \"{user_message}\"")

        try:
            if action == "catalog_search":
                return handle_catalog_search(retrieval_input)

            elif action == "item_attribute_lookup":
                return handle_attribute_lookup(retrieval_input)

            elif action == "item_compare":
                return handle_item_compare(retrieval_input)

            elif action == "explanation_generate":
                return handle_explanation_generate(retrieval_input)

            elif action == "item_detail_lookup":
                return handle_item_detail_lookup(retrieval_input)

        except Exception as e:
            print(f"M2 Router: [ERROR] Handler for '{action}' failed: {e}")
            return {
                "action": action,
                "success": False,
                "response_text": "I encountered an error processing your request. Please try again.",
                "items": [],
                "error": str(e),
            }


# Global singleton for easy import
m2_router = M2ActionRouter()
