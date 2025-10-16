from __future__ import annotations
from utils.strategies.interfaces import SanityChecker
from utils.strategies.sanity import BasicClassificationSanity

def make_sanity_checker() -> SanityChecker:
    return BasicClassificationSanity()