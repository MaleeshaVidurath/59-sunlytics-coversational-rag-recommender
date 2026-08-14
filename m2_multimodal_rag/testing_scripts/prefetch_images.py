"""
Bulk image cache warmer for M2.

Downloads product images into data/images/ ahead of time so live requests
never pay the per-file Kaggle download cost. Safe to interrupt and re-run —
already-cached images are skipped instantly.

Usage (from repo root):
    python m2_multimodal_rag/testing_scripts/prefetch_images.py --limit 500
    python m2_multimodal_rag/testing_scripts/prefetch_images.py            # all catalog articles
    python m2_multimodal_rag/testing_scripts/prefetch_images.py --workers 8

Notes:
    - The full catalog is ~42k images (several GB) — start with --limit to
      warm the most common items, or leave it running overnight for all.
    - Requires Kaggle API credentials (same as the live server).
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.data_loader import data_loader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download M2 product images.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of images to fetch (default: all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent downloads (default 4 — be kind to Kaggle)")
    args = parser.parse_args()

    articles_df = data_loader.load_articles()
    article_ids = [str(a).zfill(10) for a in articles_df["article_id"].tolist()]
    if args.limit:
        article_ids = article_ids[: args.limit]

    cache_dir = data_loader.image_cache_dir
    pending = [
        aid for aid in article_ids
        if not (cache_dir / aid[:3] / f"{aid}.jpg").exists()
    ]
    print(f"[prefetch] {len(article_ids)} articles requested, "
          f"{len(article_ids) - len(pending)} already cached, "
          f"{len(pending)} to download.")
    if not pending:
        print("[prefetch] Nothing to do.")
        return

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(data_loader.get_image, aid): aid for aid in pending}
        for fut in as_completed(futures):
            aid = futures[fut]
            try:
                ok = fut.result() is not None
            except Exception:
                ok = False
            done += 1
            failed += 0 if ok else 1
            if done % 25 == 0 or done == len(pending):
                print(f"[prefetch] {done}/{len(pending)} done ({failed} failed)")

    print(f"[prefetch] Finished: {done - failed} downloaded, {failed} failed "
          f"(failures are usually articles with no image in the dataset).")


if __name__ == "__main__":
    main()
