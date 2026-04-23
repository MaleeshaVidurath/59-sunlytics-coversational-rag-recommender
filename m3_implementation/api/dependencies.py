# m3_implementation/api/dependencies.py
# Pipeline singletons — initialised once at startup, shared across requests.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memory.core.pipeline import MemoryPipeline
from text_rag.core.rag_pipeline import TextRAGPipeline

_memory_pipeline: MemoryPipeline = None
_rag_pipeline:    TextRAGPipeline = None

def get_memory_pipeline() -> MemoryPipeline:
    return _memory_pipeline

def get_rag_pipeline() -> TextRAGPipeline:
    return _rag_pipeline

async def init_pipelines():
    global _memory_pipeline, _rag_pipeline
    _memory_pipeline = MemoryPipeline()
    _rag_pipeline    = TextRAGPipeline()
    print("[API] Pipelines initialised.")
