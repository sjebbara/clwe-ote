import sys

from nlputils import DataTools
from nlputils import EvaluationTools

import data

token_output_filename = "/vol/scstaff/sjebbara/data/AspectBasedSentiment/experiments/Final EMNLP SCLeM Test/best_epochs/epoch=35_predicted_aspects_2016_token.xml"
char_output_filename = "/vol/scstaff/sjebbara/data/AspectBasedSentiment/experiments/Final EMNLP SCLeM Test/best_epochs/epoch=73_predicted_aspects_2016_char.xml"

word_embeddings = DataTools.Embedding()
word_embeddings.load("../res/embeddings/amazon_review_corpus_en_100D_advanced_top-50000_W.npy",
                     "../res/embeddings/amazon_review_corpus_en_100D_advanced_top-50000_vocab.txt")
word_embeddings.vocabulary.set_padding(word_embeddings.vocabulary.get_index("<pad>"))
word_embeddings.vocabulary.set_unknown(word_embeddings.vocabulary.get_index("<UNK>"))

text_prerpocessing = lambda t: t.strip().lower()
print "read token documents..."
token_documents = data.read_semeval2016(token_output_filename, "sentence", text_prerpocessing, "advanced2", None,
                                        data.opinion_non_null_filter).sentences
token_document_map = dict((d.id, d) for d in token_documents)

print "read char documents..."
char_documents = data.read_semeval2016(char_output_filename, "sentence", text_prerpocessing, "advanced2", None,
                                       data.opinion_non_null_filter).sentences
char_document_map = dict((d.id, d) for d in char_documents)

print "read original documents..."
documents = data.read_semeval2016_restaurant_test("sentence", text_prerpocessing, str("advanced2"), None,
                                                  data.opinion_non_null_filter).sentences

documents = [d for d in documents if d.id in char_document_map]
document_map = dict((d.id, d) for d in documents)


def contains_unk(tokens):
    return any(t not in word_embeddings.vocabulary for t in tokens)


def score(gold_documents, predicted_documents):
    gold = set((d.id, o.start, o.end) for d in gold_documents for o in d.opinions)
    predicted = set((d.id, o.start, o.end) for d in predicted_documents for o in d.opinions)
    f1, p, r = EvaluationTools.f1(targets=gold, predictions=predicted)
    return f1


def score_ops(gold_ops, predicted_ops):
    gold = set((d.id, o.start, o.end) for o in gold_ops)
    predicted = set((d.id, o.start, o.end) for o in predicted_ops)
    f1, p, r = EvaluationTools.f1(targets=gold, predictions=predicted)
    return f1


w_color = "#00aaaa"
c_color = "#aa00aa"

T = []
C = []
L = []
print "all"
print len(documents), len(token_documents), len(char_documents)
t = score(documents, token_documents)
c = score(documents, char_documents)
T.append(t)
C.append(c)
L.append("all")

print "not unk"
not_unk_documents = [d for d in documents if not contains_unk(d.tokens)]
not_unk_token_documents = [d for d in token_documents if not contains_unk(d.tokens)]
not_unk_char_documents = [d for d in char_documents if not contains_unk(d.tokens)]
print len(not_unk_documents), len(not_unk_token_documents), len(not_unk_char_documents)

t = score(not_unk_documents, not_unk_token_documents)
c = score(not_unk_documents, not_unk_char_documents)
T.append(t)
C.append(c)
L.append("no OOV")

print "unk"
unk_documents = [d for d in documents if contains_unk(d.tokens)]
unk_token_documents = [d for d in token_documents if contains_unk(d.tokens)]
unk_char_documents = [d for d in char_documents if contains_unk(d.tokens)]
print len(unk_documents), len(unk_token_documents), len(unk_char_documents)

t = score(unk_documents, unk_token_documents)
c = score(unk_documents, unk_char_documents)
T.append(t)
C.append(c)
L.append("OOV sent.")

print "unk ops"
unk_ops_documents = [d for d in documents if any(contains_unk(o.tokens) for o in d.opinions)]
unk_ops_document_ids = set(d.id for d in unk_ops_documents)
unk_ops_token_documents = [d for d in token_documents if d.id in unk_ops_document_ids]
unk_ops_char_documents = [d for d in char_documents if d.id in unk_ops_document_ids]
print len(unk_ops_documents), len(unk_ops_token_documents), len(unk_ops_char_documents)

t = score(unk_ops_documents, unk_ops_token_documents)
c = score(unk_ops_documents, unk_ops_char_documents)
T.append(t)
C.append(c)
L.append("OOV op.")

for t, c in zip(T, C):
    print t, c, c - t

print T
print C
print L

# font = {'family': 'normal', 'size': 14}
#
# pylab.rc('font', **font)
#
# fig, ax = pylab.subplots()
# width = 0.35
# ind = numpy.array(range(len(T))) + width
# ax.bar(ind, T, width=width, color=w_color, label="word-only")
# ax.bar(ind + width, C, width=width, color=c_color, label="char+word")
#
# ax.set_xticks(ind + width)
# ax.set_xticklabels(L)
#
# pylab.ylim([0, 1])
# ax.legend()
# pylab.grid(True)
#
# pylab.savefig('../results/oov_test_model.pdf', bbox_inches='tight')
# pylab.show()

T = [T[0]]
C = [C[0]]
L = [L[0]]

print "multi word ops"
for k in [2, 3, 4]:
    print "@", k
    multi_ops_documents = [d for d in documents if any(len(o.tokens) >= k for o in d.opinions)]
    multi_ops_document_ids = set(d.id for d in multi_ops_documents)
    multi_ops_token_documents = [d for d in token_documents if d.id in multi_ops_document_ids]
    multi_ops_char_documents = [d for d in char_documents if d.id in multi_ops_document_ids]
    # unk_ops_token_documents = [d for d in token_documents if any(contains_unk(o.tokens) for o in d.opinions)]
    # unk_ops_char_documents = [d for d in char_documents if any(contains_unk(o.tokens) for o in d.opinions)]
    print len(multi_ops_documents), len(multi_ops_token_documents), len(multi_ops_char_documents)

    t = score(multi_ops_documents, multi_ops_token_documents)
    c = score(multi_ops_documents, multi_ops_char_documents)
    T.append(t)
    C.append(c)
    L.append("tokens $\geq$ {}".format(k))

for t, c in zip(T, C):
    print t, c, c - t

print T
print C
print L

# fig, ax = pylab.subplots()
# width = 0.35
# ind = numpy.array(range(len(T))) + width
# ax.bar(ind, T, width=width, color=w_color, label="word-only")
# ax.bar(ind + width, C, width=width, color=c_color, label="char+word")
#
# ax.set_xticks(ind + width)
# ax.set_xticklabels(L)
#
# pylab.ylim([0, 1])
# ax.legend()
# pylab.grid(True)
# pylab.savefig('../results/multi_word_test_model.pdf', bbox_inches='tight')
# pylab.show()
#
sys.exit()
t_only_pairs = []
c_only_pairs = []
other_pairs = []
unk_aspect_pairs = []
t_correct = []
c_correct = []
t_unk_correct = []
c_unk_correct = []

t_correct_unk_ops = []
c_correct_unk_ops = []

t_missed_unk_ops = []
c_missed_unk_ops = []

t_correct_c_missed_unk_ops = []
c_correct_t_missed_unk_ops = []

for d_id in token_document_map.keys():
    d = document_map[d_id]
    td = token_document_map[d_id]
    cd = char_document_map[d_id]

    do = set((o.start, o.end) for o in d.opinions)
    to = set((o.start, o.end) for o in td.opinions)
    co = set((o.start, o.end) for o in cd.opinions)

    t_only = to - co
    c_only = co - to

    if len(t_only) > 0 and len(c_only) == 0:
        t_only_pairs.append((d, td, cd))

    elif len(c_only) > 0 and len(t_only) == 0:
        c_only_pairs.append((d, td, cd))

    elif len(c_only) > 0 and len(t_only) > 0:
        other_pairs.append((d, td, cd))

    contains_unk_opinions = any(contains_unk(o.tokens) for o in d.opinions)
    if contains_unk_opinions:
        unk_aspect_pairs.append((d, td, cd))

    if do == to and do != co:
        t_correct.append((d, td, cd))
    if do == co and do != to:
        c_correct.append((d, td, cd))

    if contains_unk_opinions and do == to and do != co:
        t_unk_correct.append((d, td, cd))
    if contains_unk_opinions and do == co and do != to:
        c_unk_correct.append((d, td, cd))

    t_correct_unk_ops += [(d, o) for o in td.opinions if contains_unk(o.tokens) and (o.start, o.end) in do]
    c_correct_unk_ops += [(d, o) for o in cd.opinions if contains_unk(o.tokens) and (o.start, o.end) in do]

    t_missed_unk_ops += [(d, o) for o in d.opinions if contains_unk(o.tokens) and (o.start, o.end) not in to]
    c_missed_unk_ops += [(d, o) for o in d.opinions if contains_unk(o.tokens) and (o.start, o.end) not in co]

    t_correct_c_missed_unk_ops += [(d, o) for o in td.opinions if
                                   contains_unk(o.tokens) and (o.start, o.end) in do and (o.start, o.end) not in co]
    c_correct_t_missed_unk_ops += [(d, o) for o in cd.opinions if
                                   contains_unk(o.tokens) and (o.start, o.end) in do and (o.start, o.end) not in to]

print len(t_only_pairs)
print len(c_only_pairs)
print len(other_pairs)
print len(unk_aspect_pairs)
print len(t_correct)
print len(c_correct)
print len(t_unk_correct)
print len(c_unk_correct)

print len(t_correct_unk_ops)
print len(c_correct_unk_ops)

print len(t_missed_unk_ops)
print len(c_missed_unk_ops)

print len(t_correct_c_missed_unk_ops)
print len(c_correct_t_missed_unk_ops)


def _ops2str(opinions):
    return "|".join(["({}-{}): '{}'".format(o.start, o.end, " ".join(
        DataTools.mark_unknown(o.tokens, word_embeddings.vocabulary.word2index))) for o in opinions])


def _print_list(l):
    for d, td, cd in l:
        d.opinions = sorted(d.opinions, key=lambda o: o.start)
        td.opinions = sorted(td.opinions, key=lambda o: o.start)
        cd.opinions = sorted(cd.opinions, key=lambda o: o.start)

        print "#############################################"
        print " ".join(DataTools.mark_unknown(d.tokens, word_embeddings.vocabulary.word2index))
        print ""
        print "ORIG:\t", _ops2str(d.opinions)
        print ""
        print "TOKN:\t", _ops2str(td.opinions)
        print ""
        print "CHAR:\t", _ops2str(cd.opinions)


def _print_op_list(ops, dmap):
    for d, o in ops:
        print "#############################################"
        print " ".join(DataTools.mark_unknown(d.tokens, word_embeddings.vocabulary.word2index))
        print "ORIG:\t", _ops2str(d.opinions)
        print o
        print ""
        print "    :\t", _ops2str(dmap[d.id].opinions)


_print_op_list(t_correct_c_missed_unk_ops, char_document_map)
print "#################"
print "#################"
print "#################"
print ""
_print_op_list(c_correct_t_missed_unk_ops, token_document_map)
