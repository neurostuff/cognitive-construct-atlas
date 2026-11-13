"""The contrast hub, through the legacy modules (legacy/contrast_hub/hub.py and friends).

The hub's own code (normalizing sides into signatures) is not ported yet; this adapter is the one place that imports
it, so the rest of the package depends only on the ContrastCatalog port.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .files import read_json


class LegacyHubCatalog:
    def __init__(self, hub_json: str | Path, legacy_root: str | Path):
        root = Path(legacy_root)
        for p in (root, root / "contrast_hub"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from atlas_data import label                      # noqa: E402  (legacy)
        from contrast_construct_authors import kind       # noqa: E402  (legacy)
        hub = read_json(hub_json)
        self._entries = {**hub.get("parents", {}), **hub["contrasts"]}
        self._label, self._kind = label, kind

    def label(self, key: str, sign: int) -> str:
        return self._label(key, sign, self._entries)

    def kind(self, key: str) -> str:
        return self._kind(key, self._entries)
