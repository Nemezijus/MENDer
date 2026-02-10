from __future__ import annotations
from engine.components.interfaces import SanityChecker
from engine.components.sanity.sanity import BasicClassificationSanity

def make_sanity_checker() -> SanityChecker:
    return BasicClassificationSanity()