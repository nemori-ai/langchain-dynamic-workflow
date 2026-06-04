"""Host-facing run registry abstraction for workflow launches and resumes.

This module hosts the ``WorkflowRunStore`` protocol and its in-memory default,
which back the workflow tool's run registry. The persistent sqlite-backed
implementation lives in the sibling ``_persistence`` module.
"""
