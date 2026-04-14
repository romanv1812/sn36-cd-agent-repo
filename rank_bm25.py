"""Minimal BM25Okapi — no numpy. Drop-in for rank_bm25 package."""
import math

class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 1.0
        self.idf = {}
        self.corpus = corpus
        nd = {}
        for doc in corpus:
            seen = set()
            for word in doc:
                if word not in seen:
                    nd[word] = nd.get(word, 0) + 1
                    seen.add(word)
        for word, freq in nd.items():
            self.idf[word] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def get_scores(self, query):
        scores = [0.0] * self.corpus_size
        for q in query:
            idf = self.idf.get(q, 0.0)
            for i, doc in enumerate(self.corpus):
                tf = doc.count(q)
                dl = self.doc_lens[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom if denom else 0.0
        return scores
