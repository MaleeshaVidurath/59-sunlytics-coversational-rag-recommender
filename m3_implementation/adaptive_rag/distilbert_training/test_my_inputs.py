# test_my_inputs.py
# Run from distilbert_training folder:
#     python test_my_inputs.py

from predict import Predictor

predictor = Predictor()

TEST_CASES = [

    # ── ORIGINAL 9 TESTS ──────────────────────────────────────────

    {
        "description": "1. Fresh request → INITIAL_REQUEST → FULL",
        "history": [],
        "message": "Can you recommend me a casual dress for summer?"
    },
    {
        "description": "2. Colour preference change → REFINEMENT → FULL",
        "history": [
            {"role": "user", "content": "Show me some dresses for a party"},
            {"role": "bot",  "content": "Option 1: Valerie dress (black, dress). Option 2: Angel (dark pink, dress)."},
        ],
        "message": "Can you show me red ones instead?"
    },
    {
        "description": "3. Material question → ATTRIBUTE_QUESTION → PARTIAL",
        "history": [
            {"role": "user", "content": "I need a blouse for work"},
            {"role": "bot",  "content": "Option 1: Lucy blouse (white). Option 2: Django (blue)."},
        ],
        "message": "What material is the first one made of?"
    },
    {
        "description": "4. Why question → EXPLANATION_WHY → PARTIAL",
        "history": [
            {"role": "user", "content": "Find me something elegant for a dinner"},
            {"role": "bot",  "content": "Option 1: Manson blazer (dark grey). Option 2: Parma dress (black)."},
        ],
        "message": "Why did you recommend the Manson blazer?"
    },
    {
        "description": "5. Compare items → COMPARISON → PARTIAL",
        "history": [
            {"role": "user", "content": "Show me some trousers"},
            {"role": "bot",  "content": "Option 1: Vino trousers (off white). Option 2: Enter trousers (black)."},
        ],
        "message": "Which one is more casual?"
    },
    {
        "description": "6. Vague reference → SELECTION_REFERENCE → PARTIAL",
        "history": [
            {"role": "user", "content": "I want a sweater for winter"},
            {"role": "bot",  "content": "Option 1: Montana (beige, sweater). Option 2: Cora (light pink, sweater)."},
        ],
        "message": "Tell me more about the second one"
    },
    {
        "description": "7. Positive feedback → FEEDBACK → NO",
        "history": [
            {"role": "user", "content": "Show me black jackets"},
            {"role": "bot",  "content": "Option 1: DIV Xena jacket (black). Option 2: Rick Coat (dark blue)."},
        ],
        "message": "Perfect, I love the first one!"
    },
    {
        "description": "8. Greeting → CHITCHAT → NO",
        "history": [],
        "message": "Hello! How are you?"
    },
    {
        "description": "9. Multi-turn attribute → ATTRIBUTE_QUESTION → PARTIAL",
        "history": [
            {"role": "user", "content": "I need shoes for a wedding"},
            {"role": "bot",  "content": "Option 1: Lenora ballerina (dark beige). Option 2: Henry court (grey)."},
            {"role": "user", "content": "What material are the ballerinas?"},
            {"role": "bot",  "content": "The Lenora ballerina is made from leather."},
        ],
        "message": "Is the leather one easy to clean?"
    },

    # ── PREVIOUSLY FAILING CASES ──────────────────────────────────

    {
        "description": "10. Short colour+product → REFINEMENT (was CHITCHAT)",
        "history": [
            {"role": "user", "content": "I need a short"},
            {"role": "bot",  "content": "Option 1: whisper shorts (white) £11.74. Option 2: whisper shorts (white) £7.25."},
        ],
        "message": "red short"
    },
    {
        "description": "11. Colour+one → REFINEMENT (was CHITCHAT)",
        "history": [
            {"role": "user", "content": "I need a short"},
            {"role": "bot",  "content": "Option 1: whisper shorts (white) £11.74. Option 2: whisper shorts (white) £7.25."},
        ],
        "message": "red one"
    },
    {
        "description": "12. Need+colour+one → REFINEMENT (was CHITCHAT)",
        "history": [
            {"role": "user", "content": "I need a short"},
            {"role": "bot",  "content": "Option 1: whisper shorts (white) £11.74. Option 2: whisper shorts (white) £7.25."},
            {"role": "user", "content": "I like it"},
            {"role": "bot",  "content": "Great choice! Would you like to see more?"},
        ],
        "message": "need red one"
    },
    {
        "description": "13. After feedback colour change → REFINEMENT (was CHITCHAT)",
        "history": [
            {"role": "user", "content": "I need a short"},
            {"role": "bot",  "content": "Option 1: whisper shorts (white) £11.74. Option 2: whisper shorts (white) £7.25."},
            {"role": "user", "content": "I like red one"},
            {"role": "bot",  "content": "Great choice! Would you like to see more?"},
        ],
        "message": "red short need"
    },
    {
        "description": "14. Casual wear → REFINEMENT (was CHITCHAT)",
        "history": [
            {"role": "user", "content": "I need something for work"},
            {"role": "bot",  "content": "Option 1: Office blazer (black). Option 2: Work dress (navy)."},
        ],
        "message": "casual wear need"
    },
    {
        "description": "15. Need branded → REFINEMENT (was FEEDBACK)",
        "history": [
            {"role": "user", "content": "Show me white trousers"},
            {"role": "bot",  "content": "Option 1: Bitten trousers (white) £17.84. Option 2: Nut trousers (white) £33.27."},
        ],
        "message": "need branded trouser"
    },
    {
        "description": "16. Simple yes → FEEDBACK (was missing)",
        "history": [
            {"role": "user", "content": "Show me black jackets"},
            {"role": "bot",  "content": "Option 1: City jacket (black). Option 2: Slim jacket (dark blue)."},
        ],
        "message": "yes please"
    },
    {
        "description": "17. Option 1 → FEEDBACK (was missing)",
        "history": [
            {"role": "user", "content": "Show me dresses"},
            {"role": "bot",  "content": "Option 1: London dress (black) £11.08. Option 2: SS London dress (black) £15.12."},
        ],
        "message": "option 1"
    },
    {
        "description": "18. Take it → FEEDBACK (was missing)",
        "history": [
            {"role": "user", "content": "Show me tops"},
            {"role": "bot",  "content": "Option 1: Basic top (black). Option 2: Rib top (white)."},
        ],
        "message": "take it"
    },
    {
        "description": "19. Thank you farewell → CHITCHAT (was greeting)",
        "history": [
            {"role": "user", "content": "Show me dresses"},
            {"role": "bot",  "content": "Option 1: London dress. Option 2: SS London dress."},
            {"role": "user", "content": "I'll take the first one"},
            {"role": "bot",  "content": "Great choice!"},
        ],
        "message": "thanks that is really helpful"
    },
    {
        "description": "20. Typo thank you → CHITCHAT (was missing)",
        "history": [
            {"role": "user", "content": "Show me tops"},
            {"role": "bot",  "content": "Option 1: Basic top. Option 2: Rib top."},
        ],
        "message": "thnak you"
    },
    {
        "description": "21. Price question → ATTRIBUTE_QUESTION (was missing)",
        "history": [
            {"role": "user", "content": "Show me dresses"},
            {"role": "bot",  "content": "Option 1: London dress (black) £11.08. Option 2: SS London dress (black) £15.12."},
        ],
        "message": "how much is it?"
    },
    {
        "description": "22. Short why → EXPLANATION_WHY (was missing)",
        "history": [
            {"role": "user", "content": "Show me dresses"},
            {"role": "bot",  "content": "Option 1: London dress (black). Option 2: SS London dress (black)."},
        ],
        "message": "why?"
    },
    {
        "description": "23. Which is cheaper → COMPARISON (was missing)",
        "history": [
            {"role": "user", "content": "Show me shirts"},
            {"role": "bot",  "content": "Option 1: Jonas shirt (dark blue) £10.08. Option 2: Rob shirt (black) £8.56."},
        ],
        "message": "which is cheaper?"
    },
    {
        "description": "24. That one → SELECTION_REFERENCE (was missing)",
        "history": [
            {"role": "user", "content": "Show me sweaters"},
            {"role": "bot",  "content": "Option 1: Montana (beige). Option 2: Cora (light pink)."},
        ],
        "message": "that one"
    },
    {
        "description": "25. T-shirt request → INITIAL_REQUEST (was Shirt)",
        "history": [],
        "message": "I want a t shirt"
    },
    {
        "description": "26. Formal wear → INITIAL_REQUEST (was CHITCHAT)",
        "history": [],
        "message": "I need formal wear for my interview"
    },
    {
        "description": "27. Short colour+product no context → INITIAL_REQUEST",
        "history": [],
        "message": "red dress"
    },
    {
        "description": "28. Nope negative feedback → FEEDBACK",
        "history": [
            {"role": "user", "content": "Show me coats"},
            {"role": "bot",  "content": "Option 1: Trench coat (camel). Option 2: Puffer jacket (black)."},
        ],
        "message": "nope"
    },
    {
        "description": "29. The first → SELECTION_REFERENCE",
        "history": [
            {"role": "user", "content": "Show me skirts"},
            {"role": "bot",  "content": "Option 1: Mini skirt (black). Option 2: Midi skirt (navy)."},
        ],
        "message": "the first"
    },
    {
        "description": "30. Bye farewell → CHITCHAT",
        "history": [
            {"role": "user", "content": "Show me dresses"},
            {"role": "bot",  "content": "Great choice! Enjoy your purchase."},
        ],
        "message": "bye"
    },
]

# ── Run all tests ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLASSIFICATION TEST RESULTS — 30 TEST CASES")
print("=" * 70)

EXPECTED = {
    1: "INITIAL_REQUEST", 2: "REFINEMENT", 3: "ATTRIBUTE_QUESTION",
    4: "EXPLANATION_WHY", 5: "COMPARISON", 6: "SELECTION_REFERENCE",
    7: "FEEDBACK", 8: "CHITCHAT", 9: "ATTRIBUTE_QUESTION",
    10: "REFINEMENT", 11: "REFINEMENT", 12: "REFINEMENT",
    13: "REFINEMENT", 14: "REFINEMENT", 15: "REFINEMENT",
    16: "FEEDBACK", 17: "FEEDBACK", 18: "FEEDBACK",
    19: "CHITCHAT", 20: "CHITCHAT", 21: "ATTRIBUTE_QUESTION",
    22: "EXPLANATION_WHY", 23: "COMPARISON", 24: "SELECTION_REFERENCE",
    25: "INITIAL_REQUEST", 26: "INITIAL_REQUEST", 27: "INITIAL_REQUEST",
    28: "FEEDBACK", 29: "SELECTION_REFERENCE", 30: "CHITCHAT",
}

passed = 0
failed = 0
failed_cases = []

for i, case in enumerate(TEST_CASES, 1):
    result = predictor.predict(case["history"], case["message"])
    predicted = result["label_name"]
    expected  = EXPECTED.get(i, "?")
    correct   = predicted == expected
    if correct:
        passed += 1
        status = "✓"
    else:
        failed += 1
        failed_cases.append(i)
        status = "✗"

    conf = result["confidence"]
    conf_bar = "█" * int(conf * 10)
    print(f"{status} Test {i:2}: {case['message'][:35]:<35} "
          f"→ {predicted:<20} {conf:.0%} {conf_bar}")
    if not correct:
        print(f"         Expected: {expected}")

print(f"\n{'='*70}")
print(f"SCORE: {passed}/30 passed  ({passed/30*100:.0f}%)")
if failed_cases:
    print(f"Failed tests: {failed_cases}")
else:
    print("All 30 tests passed!")
print("="*70)
