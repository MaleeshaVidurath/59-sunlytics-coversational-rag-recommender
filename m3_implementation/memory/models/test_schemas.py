# m3_implementation/memory/models/test_schemas.py
# Quick sanity check that all schemas instantiate correctly.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from memory.models.schemas import (
    UserDocument, SessionDocument, ConversationTurn,
    TurnClassification, DialogueState, PreferenceEntry,
    ItemInContext, ExplanationDocument, ExplanationClaim
)

print("Testing schema creation...\n")

# 1. Create a user
user = UserDocument(customer_id="0001d44dbe7f6c4b35200...")
print(f"User created: {user.user_id}")
print(f"  Preferences: {len(user.attribute_preferences)} (should be 0)")
print(f"  Style profile: {user.style_profile}")

# 2. Add a preference
pref = PreferenceEntry(
    category="colour",
    attribute_name="colour_group_name",
    attribute_value="Black",
    sentiment=0.92,
    confidence=0.97,
    source="explicit"
)
user.attribute_preferences.append(pref)
print(f"\nPreference added: {pref.pref_id} — {pref.attribute_value} ({pref.sentiment})")

# 3. Create a session
session = SessionDocument(user_id=user.user_id)
print(f"\nSession created: {session.session_id}")
print(f"  Status: {session.status}")
print(f"  Dialogue state hard constraints: {session.dialogue_state.hard_constraints}")

# 4. Add a conversation turn with classification
turn = ConversationTurn(
    turn_number=1,
    role="user",
    content="I want a black dress under £50",
    classification=TurnClassification(
        label="INITIAL_REQUEST",
        retrieval_strategy="FULL",
        confidence=0.993
    ),
    entities={
        "colour_group_name": "Black",
        "product_type_name": "Dress",
        "price_max": 50.0
    }
)
session.turns.append(turn)
session.turn_count = 1
print(f"\nTurn added: {turn.turn_id}")
print(f"  Content: {turn.content}")
print(f"  Classification: {turn.classification.label} → {turn.classification.retrieval_strategy}")
print(f"  Entities: {turn.entities}")

# 5. Create an explanation with claims
claim = ExplanationClaim(
    claim_text="This item matches the user's preference for black colour",
    claim_type="preference_match",
    attribute="colour_group_name",
    evidence_value="Black",
    user_preference_ref=pref.pref_id,
    confidence=0.97
)
explanation = ExplanationDocument(
    recommendation_id="rec_test001",
    article_id="108775015",
    session_id=session.session_id,
    user_id=user.user_id,
    turn_id=turn.turn_id,
    full_explanation="I recommend this black dress because it matches your colour preference.",
    claims=[claim]
)
print(f"\nExplanation created: {explanation.explanation_id}")
print(f"  Claims: {len(explanation.claims)}")
print(f"  First claim: {explanation.claims[0].claim_text}")
print(f"  Claim status: {explanation.claims[0].status}")

print("\nAll schema tests PASSED.")