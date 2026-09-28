"""Minimal Track 1 (Agenthon 2026) coding agent.

The agent reads one task directory, asks the House model for a solution
script, runs it, checks the deliverables deterministically and repairs the
script until the checks pass or the request/time budget runs out.
"""

__version__ = "0.1.0"
