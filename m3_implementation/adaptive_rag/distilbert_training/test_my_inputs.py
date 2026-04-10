# test_my_inputs.py
# ─────────────────────────────────────────────────────────────────
# Use this file to test the trained model with your own custom
# conversations. Edit the TEST_CASES list below and run:
#     python test_my_inputs.py
# ─────────────────────────────────────────────────────────────────

from predict import Predictor

# Load the model once — this takes a few seconds
predictor = Predictor()

# ── Define your own test conversations here ────────────────────────
# Each test case is a dict with:
#   "description" : a label so you know what you are testing
#   "history"     : list of prior turns (can be empty for fresh sessions)
#   "message"     : the current user message you want to classify
#
# history turns must be dicts with "role" ("user" or "bot") and "content"

TEST_CASES = [

    # ── Test 1: Fresh request with no history ──────────────────────
    {
        "description": "Fresh request, should be INITIAL_REQUEST → FULL",
        "history": [],
        "message": "Can you recommend me a casual dress for summer?"
    },

    # ── Test 2: Preference change after seeing results ─────────────
    {
        "description": "User changes colour preference → REFINEMENT → FULL",
        "history": [
            {"role": "user", "content": "Show me some dresses for a party"},
            {"role": "bot",  "content": "Here are two options. Option 1 is the Valerie dress (black, dress): Short dress in a crisp cotton weave. Option 2 is the Angel (dark pink, dress): Short A-line dress in cotton."},
        ],
        "message": "Can you show me red ones instead?"
    },

    # ── Test 3: Asking about a property of the recommended item ────
    {
        "description": "Asking about material → ATTRIBUTE_QUESTION → PARTIAL",
        "history": [
            {"role": "user", "content": "I need a blouse for work"},
            {"role": "bot",  "content": "Option 1 is the Lucy blouse (white, blouse): Long-sleeved blouse in woven fabric. Option 2 is the Django (blue, blouse): Blouse in woven fabric with a V-neck."},
        ],
        "message": "What material is the first one made of?"
    },

    # ── Test 4: User asks why a specific item was recommended ──────
    {
        "description": "Why question → EXPLANATION_WHY → PARTIAL",
        "history": [
            {"role": "user", "content": "Find me something elegant for a dinner"},
            {"role": "bot",  "content": "I suggest the Manson blazer (dark grey, blazer) and the Parma Dress (black, dress)."},
        ],
        "message": "Why did you recommend the Manson blazer?"
    },

    # ── Test 5: Comparing two items already shown ──────────────────
    {
        "description": "Comparing items → COMPARISON → PARTIAL",
        "history": [
            {"role": "user", "content": "Show me some trousers"},
            {"role": "bot",  "content": "Option 1 is the Vino trousers (off white). Option 2 is the Enter trousers (black, superstretch twill)."},
        ],
        "message": "Which one is more casual?"
    },

    # ── Test 6: User refers to one item by pronoun ─────────────────
    {
        "description": "Vague reference → SELECTION_REFERENCE → PARTIAL",
        "history": [
            {"role": "user", "content": "I want a sweater for winter"},
            {"role": "bot",  "content": "Take a look at these two: the Montana (beige, sweater) and the Cora (light pink, sweater)."},
        ],
        "message": "Tell me more about the second one"
    },

    # ── Test 7: Positive feedback after seeing items ───────────────
    {
        "description": "Positive reaction → FEEDBACK → NO retrieval",
        "history": [
            {"role": "user", "content": "Show me black jackets"},
            {"role": "bot",  "content": "Option 1 is the DIV Xena jacket (black). Option 2 is the Rick Coat (dark blue)."},
        ],
        "message": "Perfect, I love the first one!"
    },

    # ── Test 8: Chitchat / greeting ───────────────────────────────
    {
        "description": "Simple greeting → CHITCHAT → NO retrieval",
        "history": [],
        "message": "Hello! How are you?"
    },

    # ── Test 9: Your own custom example — edit this freely ─────────
    {
        "description": "My custom test",
        "history": [
            {"role": "user", "content": "I need shoes for a wedding"},
            {"role": "bot",  "content": "Option 1 is the Lenora ballerina (dark beige). Option 2 is the Henry court (grey, trainers)."},
            {"role": "user", "content": "What material are the ballerinas?"},
            {"role": "bot",  "content": "The Lenora ballerina is made from leather with leather insoles."},
        ],
        "message": "Is the leather one easy to clean?"
    },
    # Add more test cases as needed — just copy and paste the structure above my ones


]

# ── Run all test cases and print results ───────────────────────────
print("\n" + "=" * 65)
print("CUSTOM INPUT TEST RESULTS")
print("=" * 65)

for i, case in enumerate(TEST_CASES, 1):
    result = predictor.predict(case["history"], case["message"])

    print(f"\nTest {i}: {case['description']}")

    # Print the conversation history so you can see the full context
    if case["history"]:
        for turn in case["history"]:
            role = turn["role"].upper()
            print(f"  {role:5}: {turn['content'][:80]}")
    else:
        print("  [No history — fresh session]")

    print(f"  USER : {case['message']}")
    print(f"  ──────────────────────────────────────")
    print(f"  Predicted : {result['label_name']}")
    print(f"  Strategy  : {result['retrieval_strategy']}")
    print(f"  Confidence: {result['confidence']:.1%}")

    # Show the top 3 probabilities so you can see how confident
    # the model is across all classes, not just the top one
    sorted_probs = sorted(
        result["all_probabilities"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    print("  Top 3 class probabilities:")
    for label, prob in sorted_probs[:3]:
        bar = "█" * int(prob * 20)   # simple ASCII bar chart
        print(f"    {label:<25} {prob:.1%}  {bar}")