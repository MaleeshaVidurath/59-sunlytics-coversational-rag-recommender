# Runs eval_guard --stage run repeatedly until it either succeeds fully
# (exit 0) or the daily quota abort recurs with no forward progress.
# Not currently invoked automatically — kept as a convenience script the
# user can run standalone across days.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent | Split-Path -Parent)
$env:PYTHONIOENCODING = "utf-8"
python -m m2_multimodal_rag.evaluation.eval_guard --stage run
