import requests
import json
import time
import os
import subprocess
import tempfile

URL = "http://127.0.0.1:8000/api/process"
IMAGE_BASE = "http://127.0.0.1:8000/api/images"

# Directory to cache downloaded images
IMAGE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "m2_test_images")
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)


def download_and_show_image(article_id: str) -> str | None:
    """Downloads the product image from the M2 API and opens it in the default viewer."""
    image_url = f"{IMAGE_BASE}/{article_id}"
    image_path = os.path.join(IMAGE_CACHE_DIR, f"{article_id}.jpg")

    # Use cache if already downloaded
    if os.path.exists(image_path):
        _open_image(image_path)
        return image_path

    try:
        resp = requests.get(image_url, timeout=15)
        if resp.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(resp.content)
            _open_image(image_path)
            return image_path
        else:
            print(f"      [Image] Could not fetch image (HTTP {resp.status_code})")
            return None
    except Exception as e:
        print(f"      [Image] Download failed: {e}")
        return None


def _open_image(path: str):
    """Opens an image file using the OS default viewer."""
    try:
        if os.name == "nt":
            os.startfile(path)
        elif os.name == "posix":
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print(f"      [Image] Saved to: {path}  (auto-open failed)")


def print_item_card(index: int, item: dict):
    """Pretty-prints a single recommended item with image, features, and justification."""
    article_id = item.get("article_id", "N/A")
    prod_name = item.get("prod_name", "Unknown")
    colour = item.get("colour_group_name", "—")
    product_type = item.get("product_type_name", "—")
    product_group = item.get("product_group_name", "—")
    department = item.get("department_name", "—")
    index_group = item.get("index_group_name", "—")
    appearance = item.get("graphical_appearance_name", "—")
    detail_desc = item.get("detail_desc", "")
    explanation = item.get("explanation", "")
    score = item.get("score")

    print(f"\n  ┌─────────────────────────────────────────────────────")
    print(f"  │  [{index}]  {prod_name}")
    print(f"  │       Article ID : {article_id}")
    print(f"  ├─────────────────────────────────────────────────────")
    print(f"  │  FEATURES")
    print(f"  │    Colour        : {colour}")
    print(f"  │    Type          : {product_type}")
    print(f"  │    Group         : {product_group}")
    print(f"  │    Department    : {department}")
    print(f"  │    Index Group   : {index_group}")
    print(f"  │    Appearance    : {appearance}")
    if score is not None:
        print(f"  │    Match Score   : {score:.4f}")
    if detail_desc:
        # Wrap long descriptions nicely
        print(f"  ├─────────────────────────────────────────────────────")
        print(f"  │  DESCRIPTION")
        wrapped = _wrap_text(str(detail_desc), width=50)
        for line in wrapped:
            print(f"  │    {line}")
    if explanation:
        print(f"  ├─────────────────────────────────────────────────────")
        print(f"  │  WHY THIS WAS RECOMMENDED")
        wrapped = _wrap_text(explanation, width=50)
        for line in wrapped:
            print(f"  │    {line}")
    print(f"  └─────────────────────────────────────────────────────")

    # Download & display the product image
    print(f"      Opening product image...")
    download_and_show_image(article_id)


def _wrap_text(text: str, width: int = 50) -> list[str]:
    """Simple word-wrap for terminal output."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = f"{current_line} {word}".strip()
    if current_line:
        lines.append(current_line)
    return lines if lines else ["(empty)"]


def run_interactive_test():
    print("=" * 58)
    print("  INTERACTIVE M2 BACKEND TESTER")
    print("  • Shows product IMAGE, FEATURES, and JUSTIFICATION")
    print("  • Type 'quit' or 'exit' to stop.")
    print("=" * 58)

    while True:
        print("\n" + "-" * 58)
        user_message = input("Enter your search query: ").strip()

        if user_message.lower() in ['quit', 'exit']:
            print("Exiting...")
            break

        if not user_message:
            continue

        # Build the M3 payload assuming a 'catalog_search' action
        payload = {
            "retrieval_input": {
                "action": "catalog_search",
                "retrieval_strategy": "FULL",
                "user_message": user_message,
                "items_in_context": {},
                "exclude_ids": [],
                "payload": {
                    "filters": {},
                    "preference_boosts": [],
                    "penalties": {}
                }
            },
            "memory_context": {}
        }

        print("\nSending request to M2 backend... (Please wait)")
        start_time = time.time()

        try:
            response = requests.post(URL, json=payload, timeout=120)
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                print(f"\n[HTTP 200] Time Taken: {round(end_time - start_time, 2)}s")

                # ── AI Response Summary ──
                print(f"\n{'─' * 58}")
                print(f"  AI RESPONSE")
                print(f"{'─' * 58}")
                print(f"  Action : {result.get('action')}")
                print(f"  Status : {'✓ Success' if result.get('success') else '✗ Failed'}")
                ai_text = result.get('response_text', '')
                if ai_text:
                    wrapped = _wrap_text(ai_text, width=52)
                    print(f"  Summary:")
                    for line in wrapped:
                        print(f"    {line}")

                # ── Recommended Items ──
                items = result.get('items', [])
                if not items:
                    print(f"\n  No items matched your query.")
                else:
                    print(f"\n{'─' * 58}")
                    print(f"  RECOMMENDED ITEMS ({len(items)})")
                    print(f"{'─' * 58}")
                    for i, item in enumerate(items, 1):
                        print_item_card(i, item)

                # ── Error info ──
                if result.get('error'):
                    print(f"\n  ⚠  Error: {result['error']}")

            else:
                print(f"\n[ERROR {response.status_code}]")
                print(response.text)

        except requests.exceptions.ConnectionError:
            print("\n[ERROR]: Could not connect. Make sure uvicorn is running on port 8000!")
        except requests.exceptions.Timeout:
            print("\n[ERROR]: Request timed out (120s limit).")


if __name__ == "__main__":
    try:
        run_interactive_test()
    except KeyboardInterrupt:
        print("\nExiting...")
