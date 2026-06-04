"""Optional sqlite-backed persistence for cross-session / cross-process resume.

This module hosts the unified sqlite store that backs a workflow run registry,
its per-run journal, and the durable LangGraph checkpointer. It depends on the
optional ``[sqlite]`` extra; the base install stays dependency-free and falls
back to the in-memory store in ``_run_store``.
"""
