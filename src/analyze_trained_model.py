from __future__ import print_function

import os
from collections import defaultdict

import numpy
from nlputils import DataTools
from nlputils import DatasetTools
from nlputils import LearningTools
from unidecode import unidecode

import AspectExtraction
import models

# If you want to uses this script your probably have to modify the code to some extend.
# It is still strongly tied to the environment of my workstation in the university

# specify the model that you want to visualize
cv_dirname = "cv-1"
experiment_base_dirpath = os.path.join(AspectExtraction.EXPERIMENTS_OUTPUT_DIR,
                                       "Final EMNLP SCLeM Test/AspectBasedSentiment_Configuration_2017-06-07_17:22:22_2016_char")
# load the configuration that specifies the network topology
conf = LearningTools.Configuration.load(os.path.join(experiment_base_dirpath, cv_dirname, "configuration.conf"))

print(conf)

model_fn = models.__dict__[conf.model]
modelz = model_fn(word_embedding_weights=None, **conf)
char_model = modelz[1]

# load the trained weights
weights = char_model.load_weights(os.path.join(experiment_base_dirpath, cv_dirname, "models/best_model.h5"),
                                  by_name=True)

# load resources: character vocabulary and pretrained word embeddings
char_vocabulary = DataTools.Vocabulary()
char_vocabulary.load(os.path.join(experiment_base_dirpath, "char_vocabulary.txt"))
char_vocabulary.set_padding(char_vocabulary.get_index("<0>"))
char_vocabulary.set_unknown(char_vocabulary.get_index("<?>"))
print(char_vocabulary)

word_embeddings = DataTools.Embedding()
word_embeddings.load("../res/embeddings/amazon_review_corpus_en_100D_advanced_top-100000_W.npy",
                     "../res/embeddings/amazon_review_corpus_en_100D_advanced_top-100000_vocab.txt")
word_embeddings.vocabulary.set_padding(word_embeddings.vocabulary.get_index("<pad>"))
word_embeddings.vocabulary.set_unknown(word_embeddings.vocabulary.get_index("<UNK>"))

vocab = LearningTools.load_as_list("../res/embeddings/amazon_review_corpus_en_100D_advanced_top-100000_vocab.txt",
                                   to_object=lambda line: line.split(" ")[0])

# specify the suffixes and their respective colors
suffix_colors = dict([("ing", "r"), ("ly", "g"), ("able", "b"), ("ish", "c"), ("less", "m"), ("ize", "y")])
suffixes = list(suffix_colors.keys())
suffix_colors["other"] = "k"

# specify the words to visualize
words = [w for w in vocab if any(w.endswith(s) for s in suffixes)]
if len(words) > 2000:
    words = words[:2000]
custom_words = ["service", "serivce", "food", "atmosphere", "delicious", "delicus", "place", "waiter"]
all_words = words + custom_words

# collect all char-level wor embeddings by applying the character level model to each word in "all_words"
Wc = []
char_vectorizer = DatasetTools.LambdaVectorizer(lambda w: [c for c in unidecode(w)])
char_vectorizer = DatasetTools.Padded1DSequenceVectorizer(char_vocabulary, "pre")(char_vectorizer)
union = DatasetTools.VectorizerUnion()
union = union("char_word_input", char_vectorizer)
batch_generator = DatasetTools.BatchGenerator(all_words, batch_size=20, vectorizer=union, raw_data_name="word")
for i, batches in enumerate(batch_generator):
    print("batch", i)
    char_word_vectors = char_model.predict_on_batch(batches)
    batches.char_word_output = char_word_vectors
    for instance in DatasetTools.BatchIterator([batches]):
        Wc.append(instance.char_word_output)

Wc = numpy.array(Wc)

Ww = word_embeddings.get_vectors(words, drop_unknown=True)

# Write the vectors to the respective files
for name, W in [("char", Wc), ("word", Ww)]:
    groups = defaultdict(list)
    for vec, w in zip(W, words):
        s = "other"
        for suffix in suffixes:
            if w.endswith(suffix):
                s = suffix
                break

        groups[s].append(vec)

    for suffix, vecs in groups.items():
        with open("../results/{}_{}.txt".format(name, suffix), "w") as f:
            vecs = numpy.array(vecs)
            print(suffix, len(vecs))
            f.write(str(vecs.tolist()))
