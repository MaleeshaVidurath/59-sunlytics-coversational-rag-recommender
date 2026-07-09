# m3_implementation/test_result/hallucination_result/make_colab_bundle.py
#
# Builds colab_bundle.zip — the minimal file set the Colab notebook
# (hallucination_eval_colab.ipynb) needs to reproduce the evaluation:
#
#   m3_implementation/text_rag/                    (checker + config code)
#   m3_implementation/test_result/hallucination_result/
#       *.py, captured_cases.jsonl,                (pipeline + raw cases)
#       results_detector_eval_v1.json, _v2.json    (archived v1/v2 for the
#                                                   progression figure/summary)
#   shared/main_data_set/sample_articles.csv       (catalog names for the
#                                                   checker's name gate)
#
# Deliberately EXCLUDED: databases, models, venv, figures, logs, backups.
# The labeled test set is regenerated in Colab (deterministic, seed 42).
#
# Run:  python test_result/hallucination_result/make_colab_bundle.py
# Output: colab_bundle.zip in this folder — upload it to the notebook.

import os
import zipfile

_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_DIR, "..", "..", ".."))
OUT       = os.path.join(_DIR, "colab_bundle.zip")

INCLUDE_HERE = [
    "capture.py", "corrupt_cases.py", "run_detector_eval.py",
    "build_summary.py", "make_figures.py", "__init__.py",
    "captured_cases.jsonl",
    "results_detector_eval_v1.json", "results_detector_eval_v2.json",
]


def main():
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        # text_rag package (checker + config), .py files only
        text_rag = os.path.join(REPO_ROOT, "m3_implementation", "text_rag")
        for root, _dirs, fnames in os.walk(text_rag):
            for fn in fnames:
                if fn.endswith(".py"):
                    full = os.path.join(root, fn)
                    arc = os.path.relpath(full, REPO_ROOT).replace(os.sep, "/")
                    z.write(full, arc)
                    n += 1

        # evaluation pipeline + raw cases + archived results
        for fn in INCLUDE_HERE:
            full = os.path.join(_DIR, fn)
            if os.path.exists(full):
                arc = f"m3_implementation/test_result/hallucination_result/{fn}"
                z.write(full, arc)
                n += 1
            else:
                print(f"  WARNING missing: {fn}")

        # package markers so imports resolve
        for arc in ("m3_implementation/test_result/__init__.py",):
            full = os.path.join(REPO_ROOT, arc.replace("/", os.sep))
            if os.path.exists(full):
                z.write(full, arc)
                n += 1

        # catalog CSV for the name gate
        csv = os.path.join(REPO_ROOT, "shared", "main_data_set", "sample_articles.csv")
        if os.path.exists(csv):
            z.write(csv, "shared/main_data_set/sample_articles.csv")
            n += 1
        else:
            print("  WARNING: sample_articles.csv not found — name gate "
                  "will run without the catalog list in Colab")

    size_mb = os.path.getsize(OUT) / (1024 * 1024)
    print(f"Written {os.path.basename(OUT)} — {n} files, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
