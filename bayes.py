import operator
import math
from collections import defaultdict, namedtuple

with open('words.txt') as f:
    words = list(map(str.strip, f))

def get_word(word_id):
    return words[word_id - 1]


class NaiveBayesClassifier():
    def __init__(self, label_freqs, word_freqs):
        self._label_freqs = label_freqs
        self._word_freqs = word_freqs

    def _label_prob(self, label, article):
        ans = math.log(self._label_freqs[label])

        for word, freqs in self._word_freqs.items():
            contained = word in article
            f = freqs[contained][label]
            ans += math.log(freqs[contained][label])
        return ans

    def classify(self, article):
        best_label = None
        best_prob = None
        for label in self._label_freqs:
            p = self._label_prob(label, article)
            if best_prob is None or p >= best_prob:
                best_prob = p
                best_label = label
        return best_label


def learn_bayes_net(examples, words):
    ''' Learn a bayesian network lol. '''
    label_freqs = dict()
    word_freqs = dict()
    VALS = (True, False)
    LABELS = (1, 2)
    for word in words:
        word_freqs[word] = dict()
        for t in VALS:
            word_freqs[word][t] = defaultdict(int)

    for label, article in examples.values():
        if label not in label_freqs:
            label_freqs[label] = 0
        label_freqs[label] += 1
        for word in words:
            t = word in article
            if label not in word_freqs[word][t]:
                word_freqs[word][t][label] = 0
            word_freqs[word][t][label] += 1

    for word in words:
        freq = word_freqs[word]
        for t in VALS:
            for label in LABELS:
                freq[t][label] = (freq[t][label] + 1) / (label_freqs[label] + 2)

    for label in label_freqs:
        # Make the labels into frequencies as well
        label_freqs[label] = (label_freqs[label] + 1) / (len(examples) + 2)

    word_discrimination = []
    for word in words:
        f = word_freqs[word][True]
        disc = abs(math.log(f[1]) - math.log(f[2]))
        word_discrimination.append((word, disc))
    word_discrimination.sort(key=operator.itemgetter(1), reverse=True)
    for idx, t in enumerate(word_discrimination[:10]):
        word, disc = t
        print('#{}: {} ({})'.format(idx, get_word(word), disc))


    return NaiveBayesClassifier(label_freqs, word_freqs)


def get_data(name):
    Example = namedtuple('Example', ['label', 'article'])
    # Create a decision tree with given maximum depth
    words = set()
    articles = defaultdict(set)
    examples = dict()
    # An article will be a set() of words (ids)
    with open(name+'Data.txt') as f:
        for article_id, word_id in map(lambda l: map(int, l.split()), f):
            articles[article_id].add(word_id)
            # All words :O
            words.add(word_id)

    with open(name+'Label.txt') as f:
        for article_id, label in enumerate(f):
            article_id += 1
            examples[article_id] = Example(int(label), articles[article_id])

    return examples, words


def main():
    examples, words = get_data('train')
    test_examples, _ = get_data('test')

    bayes_net = learn_bayes_net(examples, words)
    correct = 0
    for label, article in examples.values():
        c = bayes_net.classify(article)
        if c == label:
            correct += 1

    test_correct = 0
    for label, article in test_examples.values():
        c = bayes_net.classify(article)
        if c == label:
            test_correct += 1
    print('Training: {} correct'.format(correct / len(examples)))
    print('Testing: {} correct'.format(test_correct / len(test_examples)))

if __name__ == '__main__':
    main()
