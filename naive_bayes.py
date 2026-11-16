"""A small Multinomial Naive Bayes text classifier implemented with Python only."""
import math
import re
from collections import Counter, defaultdict

def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", str(text).lower())

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.total_words = Counter()
        self.vocabulary = set()
        self.trained = False

    def fit(self, texts, labels):
        if len(texts) != len(labels) or not texts:
            raise ValueError("Texts and labels must have the same non-zero length.")
        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            self.class_counts[label] += 1
            self.word_counts[label].update(tokens)
            self.total_words[label] += len(tokens)
            self.vocabulary.update(tokens)
        self.trained = True
        return self

    def predict_one(self, text):
        if not self.trained:
            raise RuntimeError("Train the classifier before making predictions.")
        tokens = tokenize(text)
        total_docs = sum(self.class_counts.values())
        vocab_size = max(1, len(self.vocabulary))
        log_scores = {}
        for label, count in self.class_counts.items():
            # Laplace-smoothed prior and word likelihoods.
            score = math.log(count / total_docs)
            denominator = self.total_words[label] + self.alpha * vocab_size
            for token in tokens:
                word_count = self.word_counts[label][token]
                probability = (word_count + self.alpha) / denominator
                score += math.log(probability)
            log_scores[label] = score

        # Convert log scores into normalized confidence-like probabilities.
        max_score = max(log_scores.values())
        exp_scores = {label: math.exp(score - max_score)
                      for label, score in log_scores.items()}
        total = sum(exp_scores.values())
        probabilities = {label: value / total for label, value in exp_scores.items()}
        best_label = max(probabilities, key=probabilities.get)
        return best_label, probabilities[best_label], probabilities

    def predict(self, texts):
        return [self.predict_one(text)[0] for text in texts]
