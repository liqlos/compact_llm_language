"""Live-model evaluation harness (provider-neutral).

Replays the five benchmark scenarios as baseline vs compiled contexts, asks a
model per-scenario questions, and scores exact-fact recall, citation presence
and constraint adherence. Works with any OpenAI-compatible chat endpoint or a
deterministic fixture client for offline testing.
"""
