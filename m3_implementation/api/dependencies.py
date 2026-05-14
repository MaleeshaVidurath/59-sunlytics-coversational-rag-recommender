# m3_implementation/api/dependencies.py
# Pipeline singletons — initialised once at startup, shared across requests.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memory.core.pipeline import MemoryPipeline
from text_rag.core.rag_pipeline import TextRAGPipeline
from memory.core.rl_signal_collector import RLSignalCollector

_memory_pipeline: MemoryPipeline = None
_rag_pipeline:    TextRAGPipeline = None
_rl_collector:    RLSignalCollector = None

def get_memory_pipeline() -> MemoryPipeline:
    return _memory_pipeline

def get_rag_pipeline() -> TextRAGPipeline:
    return _rag_pipeline

def get_rl_collector_dep() -> RLSignalCollector:
    return _rl_collector

async def init_pipelines():
    global _memory_pipeline, _rag_pipeline, _rl_collector
    _memory_pipeline = MemoryPipeline()
    _rag_pipeline    = TextRAGPipeline()
    _rl_collector    = RLSignalCollector()
    print("[API] Pipelines initialised.")
    print("[API] RL signal collector initialised.")
