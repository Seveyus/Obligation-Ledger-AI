"""Obligation Ledger — RAG component.

Scope of this package (Yoann's responsibility):

    PDF -> page-aware text -> chunks -> hybrid retrieval -> local LLM extraction
        -> deterministic evidence verification -> deterministic date math -> JSON

Explicitly out of scope: frontend, approval UI, the main application backend,
the committed ledger, OpenClaw / NemoClaw / OpenShell, and the model server.
"""

__version__ = "0.1.0"
