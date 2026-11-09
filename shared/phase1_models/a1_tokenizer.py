"""Assignment-1 byte-level BPE tokenizer, retrained on the Assignment-3 corpus.

Byte-level BPE (A-1 Q5 spec): byte symbols so no <unk> is needed, reserved tokens
<pad> <unk> <s> </s> at ids 0..3, each merge takes the most frequent adjacent pair
(ties broken by the lexicographically smallest merged bytes), and merges never cross
a pre-token (word / whitespace) boundary so decode() reconstructs the text exactly.
"""
import re
from collections import Counter

RESERVED = ("<pad>", "<unk>", "<s>", "</s>")
BASE = len(RESERVED)
SPLIT = re.compile(r"\S+|\s+")


def merge_sequence(ids, pair, new_id):
    a, b = pair
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.ranks = {}
        self.itos = {}
        self.vocab_size = 0
        self.cache = {}

    def train(self, text, vocab_size):
        self.itos = {i: b"" for i in range(BASE)}
        for byte in range(256):
            self.itos[BASE + byte] = bytes((byte,))

        counts = Counter(SPLIT.findall(text))
        corpus = Counter()
        for word, freq in counts.items():
            ids = tuple(BASE + byte for byte in word.encode("utf-8"))
            corpus[ids] += freq

        n_merges = max(0, vocab_size - BASE - 256)
        next_id = BASE + 256
        while len(self.merges) < n_merges:
            pair_freq = Counter()
            for ids, freq in corpus.items():
                for j in range(len(ids) - 1):
                    pair_freq[(ids[j], ids[j + 1])] += freq
            if not pair_freq:
                break
            top = max(pair_freq.values())
            if top < 2:
                break
            best_pairs = [p for p, f in pair_freq.items() if f == top]
            pair = min(best_pairs, key=lambda p: self.itos[p[0]] + self.itos[p[1]])

            self.itos[next_id] = self.itos[pair[0]] + self.itos[pair[1]]
            self.merges[pair] = next_id
            self.ranks[pair] = len(self.ranks)

            new_corpus = Counter()
            for ids, freq in corpus.items():
                new_corpus[merge_sequence(ids, pair, next_id)] += freq
            corpus = new_corpus
            next_id += 1

        self.vocab_size = BASE + 256 + len(self.merges)

    def encode_chunk(self, raw_bytes):
        if raw_bytes in self.cache:
            return self.cache[raw_bytes]
        ids = [BASE + byte for byte in raw_bytes]
        while len(ids) >= 2:
            candidates = {}
            for j in range(len(ids) - 1):
                pair = (ids[j], ids[j + 1])
                if pair in self.ranks:
                    candidates[pair] = self.ranks[pair]
            if not candidates:
                break
            pair = min(candidates, key=candidates.get)
            ids = list(merge_sequence(ids, pair, self.merges[pair]))
        self.cache[raw_bytes] = ids
        return ids

    def encode(self, text):
        ids = []
        for chunk in SPLIT.findall(text):
            ids.extend(self.encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids):
        data = b""
        for i in ids:
            data += self.itos.get(i, b"")
        return data.decode("utf-8", errors="replace")

    def get_vocab_size(self):
        return self.vocab_size


def train_tokenizer(train_text, vocab_size):
    tokenizer = BPETokenizer()
    tokenizer.train(train_text, vocab_size)
    return tokenizer
