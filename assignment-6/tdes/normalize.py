"""
tdes.normalize
==============
Normalization pipeline used ONLY for near-duplicate shingling (dedup.py).
It never touches the raw bytes used for content-hashing / tokenizer
training -- those stay untouched so the integrity chain certifies the
true original bytes, not a normalized derivative.

Pipeline: NFC -> whitespace/punctuation tokenize -> stopword filter ->
trie-based longest-suffix stripping -> (caller shingles the result).

The suffix trie targets Kannada's agglutinative morphology: functional
load that English carries in separate words (case, plurality, ...) is
carried in Kannada largely by suffixes fused onto the stem. Stripping a
curated set of common inflectional suffixes lets shingles match across
ಮನೆಗೆ / ಮನೆಯಲ್ಲಿ / ಮನೆಗಳು -- surface variants of the same content word --
which a plain stopword filter cannot do.

LIMITATION (documented deliberately, not hidden): this is a curated
approximation of Kannada suffix morphology, not a linguistically
complete morphological analyzer. Over-stripping is as harmful as
under-stripping -- it can make genuinely different documents shingle
identically. The suffix list in config.KANNADA_SUFFIXES is intentionally
conservative.
"""

from __future__ import annotations
import re
import unicodedata
from typing import List, Set

from . import config


_WORD_RE = re.compile(r"[^\s.,!?;:\"'()\[\]{}।॥]+", re.UNICODE)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def tokenize_words(text: str) -> List[str]:
    """Whitespace/punctuation word tokenization (NOT the model tokenizer --
    this is only for dedup shingling)."""
    return _WORD_RE.findall(nfc(text))


# ---------------------------------------------------------------------------
# Suffix trie: built over REVERSED suffix strings so that a single
# forward walk over a REVERSED word performs longest-suffix matching in
# O(len(word)) with no backtracking.
# ---------------------------------------------------------------------------

class _TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children = {}
        self.is_end = False


class SuffixTrie:
    def __init__(self, suffixes):
        self.root = _TrieNode()
        for suf in suffixes:
            self._insert(suf)

    def _insert(self, suffix: str) -> None:
        node = self.root
        for ch in reversed(suffix):
            node = node.children.setdefault(ch, _TrieNode())
        node.is_end = True

    def strip_longest_suffix(self, word: str, min_stem_len: int = 2) -> str:
        """Walk the reversed word against the trie, remembering the
        deepest (== longest suffix) match point. Returns the stem with
        that suffix removed, or the original word if no match, or if
        removal would leave a stem shorter than min_stem_len (guards
        against stripping a whole short word down to nothing)."""
        node = self.root
        best_depth = 0
        depth = 0
        for ch in reversed(word):
            if ch not in node.children:
                break
            node = node.children[ch]
            depth += 1
            if node.is_end:
                best_depth = depth
        if best_depth == 0:
            return word
        stem_len = len(word) - best_depth
        if stem_len < min_stem_len:
            return word
        return word[:stem_len]


_SUFFIX_TRIE = SuffixTrie(config.KANNADA_SUFFIXES)
_STOPWORDS: Set[str] = set(config.STOPWORDS_KN)


def normalize_for_shingling(text: str) -> List[str]:
    """Full pipeline: NFC -> tokenize -> drop stopwords -> strip longest
    known suffix per remaining word. Returns the list of normalized
    content words, in original order (order matters -- shingles are
    over ordered windows)."""
    words = tokenize_words(text)
    out = []
    for w in words:
        if w in _STOPWORDS:
            continue
        stem = _SUFFIX_TRIE.strip_longest_suffix(w)
        if stem:
            out.append(stem)
    return out
