# m3_implementation/test_result/hallucination_result/expanded_eval/make_expanded_colab_bundle.py
#
# Builds expanded_colab_bundle.zip for expanded_eval_colab.ipynb — the
# minimal file set to run the EXPANDED evaluation's slow stages on a Colab
# GPU (naive NLI, HHEM, LettuceDetect, SummaC — all FULL, no sampling).
#
# Run:  python test_result/hallucination_result/expanded_eval/make_expanded_colab_bundle.py

import os
import zipfile

_DIR      = os.path.dirname(os.path.abspath(__file__))
HR        = os.path.normpath(os.path.join(_DIR, ".."))
REPO_ROOT = os.path.normpath(os.path.join(HR, "..", "..", ".."))
OUT       = os.path.join(_DIR, "expanded_colab_bundle.zip")

FILES = [
    # (absolute path, archive path)
    (os.path.join(HR, "__init__.py"),
     "m3_implementation/test_result/hallucination_result/__init__.py"),
    (os.path.join(HR, "run_detector_eval.py"),
     "m3_implementation/test_result/hallucination_result/run_detector_eval.py"),
    (os.path.join(HR, "external_baselines", "__init__.py"),
     "m3_implementation/test_result/hallucination_result/external_baselines/__init__.py"),
    (os.path.join(HR, "external_baselines", "run_external_baselines.py"),
     "m3_implementation/test_result/hallucination_result/external_baselines/run_external_baselines.py"),
    (os.path.join(HR, "external_baselines", "summac_conv_vitc_sent_perc_e.bin"),
     "m3_implementation/test_result/hallucination_result/external_baselines/summac_conv_vitc_sent_perc_e.bin"),
    (os.path.join(_DIR, "__init__.py"),
     "m3_implementation/test_result/hallucination_result/expanded_eval/__init__.py"),
    (os.path.join(_DIR, "labeled_test_set_expanded.jsonl"),
     "m3_implementation/test_result/hallucination_result/expanded_eval/labeled_test_set_expanded.jsonl"),
    (os.path.join(_DIR, "hard_set", "labeled_hard_set.jsonl"),
     "m3_implementation/test_result/hallucination_result/expanded_eval/hard_set/labeled_hard_set.jsonl"),
    (os.path.join(REPO_ROOT, "m3_implementation", "test_result", "__init__.py"),
     "m3_implementation/test_result/__init__.py"),
    (os.path.join(REPO_ROOT, "shared", "main_data_set", "sample_articles.csv"),
     "shared/main_data_set/sample_articles.csv"),
]


def main():
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        # text_rag package (checker + config), .py only
        text_rag = os.path.join(REPO_ROOT, "m3_implementation", "text_rag")
        for root, _d, fnames in os.walk(text_rag):
            for fn in fnames:
                if fn.endswith(".py"):
                    full = os.path.join(root, fn)
                    arc = os.path.relpath(full, REPO_ROOT).replace(os.sep, "/")
                    z.write(full, arc)
                    n += 1
        for full, arc in FILES:
            if os.path.exists(full):
                z.write(full, arc)
                n += 1
            else:
                print(f"  WARNING missing: {full}")
    print(f"Written {os.path.basename(OUT)} — {n} files, "
          f"{os.path.getsize(OUT)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
