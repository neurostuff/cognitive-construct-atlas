"""Cognitive Construct Atlas.

Hexagonal layout:
- domain: the rules (corpus, hierarchy, audit, release tables); pure, no I/O
- ports: what the domain and use cases need from outside, as Protocols
- adapters: files, the legacy contrast hub, the Cognitive Atlas dump, TSV output
- application: use cases that wire ports to the domain
"""
__version__ = "0.1.0"
