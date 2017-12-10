from collections import Counter

import numpy
from nlputils import LexicalTools

import data

sentences2016_train = data.read_semeval2016_restaurant_train("sentence", LexicalTools.lower, "advanced2", None,
                                                             data.opinion_non_null_filter).sentences
sentences2016_test = data.read_semeval2016_restaurant_test("sentence", LexicalTools.lower, "advanced2", None,
                                                           data.opinion_non_null_filter).sentences

for name, sentences in [("2016 train", sentences2016_train), ("2016 test", sentences2016_test)]:
    opinions = [o for s in sentences for o in s.opinions]

    n_opinons = Counter(len(s.opinions) for s in sentences)
    n_opinons2 = [len(s.opinions) for s in sentences]
    phases_lengths = Counter(len(o.target.split(" ")) for o in opinions)
    phases_lengths2 = [len(o.target.split(" ")) for o in opinions]
    phrase_char_lengths = [len(o.target) for o in opinions]

    print name
    print "#Sentences", len(sentences)
    print "#Opinions", len(opinions)
    print "#Opinions per Sentence", n_opinons.most_common()
    print "#Opinions per Sentence", numpy.min(n_opinons2), numpy.max(n_opinons2), numpy.mean(n_opinons2)
    print n_opinons[0], n_opinons[1], sum(c for n, c in n_opinons.iteritems() if n >= 2)
    print "#Words per Opinion", phases_lengths.most_common()
    print "#Words per Opinion", numpy.min(phases_lengths2), numpy.max(phases_lengths2), numpy.mean(phases_lengths2)
    print phases_lengths[1], sum(c for n, c in phases_lengths.iteritems() if n >= 2)

    print "#Chars per Opinion", numpy.min(phrase_char_lengths), numpy.max(phrase_char_lengths), numpy.mean(
        phrase_char_lengths)
