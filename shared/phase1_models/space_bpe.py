# Space-attached byte-level BPE (GPT-2 style pre-tokenisation) built on your A1 tokenizer.
#
# Difference from A1: A1 splits text into  \S+ | \s+  so every space is its own token
# (~2.3 tokens per word). Here a single leading space is attached to the following word
# (" the", " Court"), so a word usually costs ~1.3 tokens and a 512-word prompt fits in the
# 1024-token context. Merge rule is unchanged (most frequent pair, ties -> lexicographically
# smallest merged bytes, never across a pre-token boundary), so decode() is still exact.
# train() is an incremental version of the A1 algorithm: it only re-counts pairs in words that
# contain the merged pair (plus a heap), which gives the same merges but runs minutes, not hours.
import heapq
import re
from collections import Counter, defaultdict
from a1_tokenizer import BPETokenizer, BASE, merge_sequence

# " ?\S+"      a word with at most one leading space
# "\s+(?!\S)"  a whitespace run, leaving its last space for the next word
# "\s+"        trailing whitespace at the very end
SPACE_SPLIT = re.compile(r" ?\S+|\s+(?!\S)|\s+")


class SpaceBPETokenizer(BPETokenizer):
    space_prefix = True          # tells the pipeline not to append a trailing space to prompts

    def train(self, text, vocab_size):
        self.itos = {i: b"" for i in range(BASE)}
        for byte in range(256):
            self.itos[BASE + byte] = bytes((byte,))
        counts = Counter(SPACE_SPLIT.findall(text))
        words = [tuple(BASE + b for b in w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())

        pair_counts = defaultdict(int)          # pair -> total frequency
        where = defaultdict(set)                # pair -> indices of words containing it
        for wi, ids in enumerate(words):
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)
        merged = lambda p: self.itos[p[0]] + self.itos[p[1]]
        heap = [(-c, merged(p), p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        n_merges = max(0, vocab_size - BASE - 256)
        next_id = BASE + 256
        while len(self.merges) < n_merges and heap:
            neg_c, _, pair = heapq.heappop(heap)
            c = pair_counts.get(pair, 0)
            if c != -neg_c or c <= 0:
                continue                         # stale heap entry
            if c < 2:
                break
            self.itos[next_id] = merged(pair)
            self.merges[pair] = next_id
            self.ranks[pair] = len(self.ranks)
            delta = defaultdict(int)
            for wi in where.pop(pair, ()):
                old = words[wi]
                new = merge_sequence(old, pair, next_id)
                if new == old:
                    continue
                f = freqs[wi]
                for p in zip(old, old[1:]):
                    delta[p] -= f
                for p in zip(new, new[1:]):
                    delta[p] += f
                    where[p].add(wi)
                words[wi] = new
            pair_counts.pop(pair, None)
            for p, d in delta.items():
                if d == 0 or p == pair:
                    continue
                pair_counts[p] += d
                if pair_counts[p] > 0:
                    heapq.heappush(heap, (-pair_counts[p], merged(p), p))
            next_id += 1
        self.vocab_size = BASE + 256 + len(self.merges)

    def encode(self, text):
        ids = []
        for chunk in SPACE_SPLIT.findall(text):
            ids.extend(self.encode_chunk(chunk.encode("utf-8")))
        return ids


def train_space_tokenizer(train_text, vocab_size):
    tok = SpaceBPETokenizer()
    tok.train(train_text, vocab_size)
    return tok
