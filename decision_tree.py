import math
from collections import defaultdict, Counter, namedtuple

with open('words.txt') as f:
    words = list(map(str.strip, f))

def get_word(word_id):
    return words[word_id - 1]


def all_equal(vals):
    i = iter(vals)
    v = next(i)
    return all(val == v for val in i)


def most_frequent(vals):
    ''' Return the most frequent value. '''
    counter = Counter(vals)
    value, count = counter.most_common(1)[0]
    return value


def _choose_word(examples, words):
    def entropy(p, n):
        # No samples? No entropy.
        if p + n == 0: return 0
        a = p / (p + n)
        b = n / (p + n)
        return -1 * sum(n * math.log2(n) for n in (a, b) if n > 0)

    def remainder(word, p, n):
        ''' So-called remainder for a word. '''
        # Now, one_true is the number of 1s in examples containing this word
        true_counts = {1: 0, 2: 0}
        false_counts = {1: 0, 2: 0}

        for label, article in examples.values():
            if word in article:
                true_counts[label] += 1
            else:
                false_counts[label] += 1
        labels = [1, 2]
        counts = [list(true_counts[l] for l in labels),
                  list(false_counts[l] for l in labels)]
        return sum((pi + ni) / (p + n) * entropy(pi, ni) for pi, ni in counts)

    def ig(word):
        ''' Information gain for a particular word. '''
        # So p is the number of examples in 1
        # and n is the number of examples that are 2
        counts = Counter(label for label, _ in examples.values())
        p = counts[1]
        n = counts[2]
        return entropy(p, n) - remainder(word, p, n)

    best_word = next(iter(words))
    best_ig = ig(best_word)
    for word in words:
        word_ig = ig(word)
        if word_ig > best_ig:
            best_ig = word_ig
            best_word = word
    return best_word, best_ig


def learn_decision_tree(examples, words, depth=1, default=None):
    ''' Learn a decision tree. '''
    if depth <= 0 or not examples:
        # Default classifier
        return Classifier(default)

    # All remaining examples are the same: Return that classification
    if all_equal(ex.label for ex in examples.values()):
        # Classifier is simply that value: No decision here.
        classifier = next(iter(examples.values())).label
        return Classifier(classifier)

    mode = most_frequent(ex.label for ex in examples.values())
    if not words:
        # No more attributes
        return Classifier(mode)
    # Best word to choose
    best_word, best_ig = _choose_word(examples, words)

    # Examples containing (True) or not containing (False) the word
    true_examples = dict()
    false_examples = dict()
    for id, ex in examples.items():
        d = true_examples if best_word in ex.article else false_examples
        d[id] = ex

    new_words = words - {best_word}
    true_subtree = learn_decision_tree(true_examples, new_words,
                                       depth=depth - 1, default=mode)
    false_subtree = learn_decision_tree(false_examples, new_words,
                                        depth=depth - 1, default=mode)
    return DecisionNode(best_word, best_ig, true_subtree, false_subtree)


class Classifier():
    ''' A single classifier. '''
    def __init__(self, label):
        self._label = label

    def classify(self, article):
        return self._label

    def __str__(self):
        return self._to_string(0)

    def _to_string(self, depth):
        pad = '-' * depth
        return '{}Leaf: {}'.format(pad, self._label)


class DecisionNode():
    ''' A test/decision node. '''
    def __init__(self, word, ig, true_child, false_child):
        self._word = word
        self.true_child = true_child
        self.false_child = false_child
        self._ig = ig

    def classify(self, article):
        ''' Classify an article (recursively). '''
        if self._word in article:
            return self.true_child.classify(article)
        return self.false_child.classify(article)

    def _to_string(self, depth):
        pad = '-' * depth
        s = pad + ('Decision on word: {} (ig: {})'
                   .format(get_word(self._word), self._ig))
        s += '\n' + pad + 'True\n' + self.true_child._to_string(depth + 1)
        s += '\n' + pad + 'False\n' + self.false_child._to_string(depth + 1)
        return s

    def __str__(self):
        return self._to_string(0)


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

    num_test = len(test_examples)
    num_train = len(examples)

    decisionTree = learn_decision_tree(examples, words, depth=5)
    print(decisionTree)
    return
    depth = 1
    best_depth = 1
    best_test = 0
    while True:
        # Build a tree for this depth
        decisionTree = learn_decision_tree(examples, words, depth=depth)
        correct = 0
        test_correct = 0
        # Check the training data
        for id, ex in examples.items():
            label, article = ex
            decision = decisionTree.classify(article)
            if decision == label:
                correct += 1

        # Check the test data
        for id, ex in test_examples.items():
            label, article = ex
            decision = decisionTree.classify(article)
            if decision == label:
                test_correct += 1
        print('At depth {} we got {} correct on training'
              .format(depth, correct / num_train))
        print('At depth {} we got {} correct on test data'
              .format(depth, test_correct / num_test))
        if test_correct > best_test:
            best_test = test_correct
            best_depth = depth
        print('At depth {} we got {} correct in total'
              .format(depth, (correct + test_correct) / (num_train + num_test)))
        if correct == num_train:
            break
        depth += 1
    print('best depth is {}'.format(best_depth))
    print(learn_decision_tree(examples, words, depth=best_depth))

if __name__ == '__main__':
    main()
