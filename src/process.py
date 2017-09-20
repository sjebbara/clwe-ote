import random
from collections import defaultdict

import numpy
from nlputils import AnnotationTools
from nlputils import DatasetTools

from nlputils import EvaluationTools
from nlputils import LearningTools
from nlputils import DataTools
from unidecode import unidecode

import xml.sax.saxutils
from bs4 import BeautifulSoup


def sorted_shuffle(documents):
    token_variation = 3
    random.shuffle(documents)
    return sorted(documents, key=lambda d: (len(d.tokens) + numpy.random.randint(0, token_variation),))


def get_vectorizer(word_vocabulary, pos_vocabulary, char_vocabulary, tagging_scheme, input_only=False):
    text_vectorizer = DatasetTools.DocumentLevelTokenExtractor()
    text_vectorizer = DatasetTools.Padded1DSequenceVectorizer(word_vocabulary, "pre")(text_vectorizer)

    if pos_vocabulary:
        pos_vectorizer = DatasetTools.DocumentLevelPosTagExtractor()
        pos_vectorizer = DatasetTools.VocabSequenceVectorizer1D(pos_vocabulary, mode="sequence")(pos_vectorizer)
        pos_vectorizer = DatasetTools.VocabularyPaddingVectorizer(pos_vocabulary, "pre")(pos_vectorizer)

    if char_vocabulary:
        char_vectorizer = DatasetTools.LambdaVectorizer(lambda d: [[c for c in unidecode(t)] for t in d.tokens])
        char_vectorizer = DatasetTools.Padded2DSequenceVectorizer(char_vocabulary, "pre")(char_vectorizer)

    expected_aspects_vectorizer = DatasetTools.LambdaVectorizer(lambda d: [float(len(d.opinions))])
    expected_aspects_vectorizer = DatasetTools.ArrayVectorizer()(expected_aspects_vectorizer)

    if not input_only:
        aspect_vectorizer = DatasetTools.Opinions2TagSequence(tagging_scheme)
        aspect_vectorizer = DatasetTools.PaddingVectorizer1D("O", "pre")(aspect_vectorizer)
        aspect_vectorizer = DatasetTools.TagSequenceVectorizer(tagging_scheme)(aspect_vectorizer)
        aspect_vectorizer = DatasetTools.ArrayVectorizer()(aspect_vectorizer)

    union = DatasetTools.VectorizerUnion()
    union = union("text_input", text_vectorizer)
    if char_vocabulary:
        union = union("char_input", char_vectorizer)
    if pos_vocabulary:
        union = union("pos_input", pos_vectorizer)
    union("n_aspects_input", expected_aspects_vectorizer)
    if not input_only:
        union = union("aspect_output", aspect_vectorizer)
    return union


def train_aspects(model, documents, word_vocabulary, pos_vocabulary, char_vocabulary, conf, e=None, n_epochs=None,
                  class_weights=None):
    tagging_scheme = AnnotationTools.get_tagging_scheme(conf.tagging_scheme)
    LearningTools.log(None, "pretrain model ...")
    LearningTools.log(None, "## Epoch %d/%d ##" % (e + 1, n_epochs))

    documents = filter(lambda d: not d.out_of_scope, documents)
    documents = filter(lambda d: len(d.opinions) > 0, documents)
    documents = sorted_shuffle(documents)
    batch_generator = DatasetTools.BatchGenerator(documents, conf.batch_size,
                                                  get_vectorizer(word_vocabulary, pos_vocabulary, char_vocabulary,
                                                                 tagging_scheme), raw_data_name="document")

    timer = LearningTools.TrainingTimer()
    timer.init(1, len(documents))
    for i, batches in enumerate(batch_generator):
        actual_batch_size = len(batches.text_input)
        i_text = i * conf.batch_size + actual_batch_size
        print("Pretrain Epoch %d/%d; Batch %d; Text %d:" % (e + 1, n_epochs, i + 1, i_text))
        LearningTools.print_batch_shapes(batches)
        history = model.train_on_batch(batches, batches)

        print "loss: %s" % history
        print timer.time_now()
        timer.process(actual_batch_size)
    return model


def evaluate_aspects(model, documents, word_vocabulary, pos_vocabulary, char_vocabulary, conf, verbose=True):
    print "Evaluate model ..."
    tagging_scheme = AnnotationTools.get_tagging_scheme(conf.tagging_scheme)
    documents = filter(lambda d: not d.out_of_scope, documents)

    batch_generator = DatasetTools.BatchGenerator(documents, 100,
                                                  get_vectorizer(word_vocabulary, pos_vocabulary, char_vocabulary,
                                                                 tagging_scheme), raw_data_name="document")

    results = LearningTools.ExperimentSnapshotResults()
    errors = 0
    for i, batches in enumerate(batch_generator):
        actual_batch_size = len(batches.text_input)
        i_text = i * conf.batch_size + actual_batch_size
        print("Pretrain Batch %d; Text %d:" % (i + 1, i_text))

        predicted_aspect_batch = model.predict_on_batch(batches)
        batches["predicted_aspect_output"] = predicted_aspect_batch

        for instance in DatasetTools.BatchIterator([batches]):
            d = instance["document"]
            true_aspects = instance["aspect_output"]
            predicted_aspect_probas = instance["predicted_aspect_output"]

            if verbose:
                print u"#### Sentence: [{}]: '{}'".format(d.id, d.text)
            tokens = DataTools.mark_unknown(d.tokens, word_vocabulary.word2index)

            true_aspects = true_aspects[-len(tokens):, :]  # remove padding
            predicted_aspect_probas = predicted_aspect_probas[-len(tokens):, :]  # remove padding

            true_aspect_spans_orig = set([(o.token_start, o.token_end) for o in d.opinions])
            true_aspect_spans = set(tagging_scheme.encoding2spans(true_aspects))
            if true_aspect_spans != true_aspect_spans_orig:
                print "ERROR: {} vs. {}".format(true_aspect_spans_orig, true_aspect_spans)
                errors += 1
            predicted_aspect_spans = set(tagging_scheme.encoding2spans(predicted_aspect_probas))

            tokens_proba = [u"{} ({:.2f},{:.2f},{:.2f})".format(t, pb, pi, po) for t, (pb, pi, po) in
                            zip(tokens, predicted_aspect_probas)]

            if verbose:
                print "TRUE:   ", tagging_scheme.visualize_tags(tokens_proba, tagging_scheme.spans2tags(len(tokens),
                                                                                                        true_aspect_spans),
                                                                spacer=" ")
                print "PRED:   ", tagging_scheme.visualize_tags(tokens_proba, tagging_scheme.spans2tags(len(tokens),
                                                                                                        predicted_aspect_spans),
                                                                spacer=" ")

            data_sample = DataTools.DataSample()
            data_sample.document = d
            data_sample.true_aspect_spans = true_aspect_spans
            data_sample.predicted_aspect_spans = predicted_aspect_spans
            data_sample.predicted_aspect_probas = predicted_aspect_probas
            results.add(data_sample)

    def extract_aspects(min_confidence=0.75):
        all_true_aspects = set()
        all_predicted_aspects = set()
        for ds in results.data_samples:
            for a in ds.true_aspect_spans:
                all_true_aspects.add((ds.document.id,) + a)

            for a in ds.predicted_aspect_spans:
                probas = numpy.max(ds.predicted_aspect_probas[a[0]:a[1]], axis=1)
                if numpy.mean(probas) > min_confidence:
                    all_predicted_aspects.add((ds.document.id,) + a)

        return all_true_aspects, all_predicted_aspects

    def score(beta=1, min_confidence=0.):
        all_true_aspects, all_predicted_aspects = results.extract_aspects(min_confidence)

        return EvaluationTools.f1(beta=beta, targets=all_true_aspects, predictions=all_predicted_aspects)

    results.extract_aspects = extract_aspects
    results.score = score

    f1, p, r = results.score(min_confidence=0)
    print "F1: {:.3f}".format(f1)
    print "P:  {:.3f}".format(p)
    print "R:  {:.3f}".format(r)

    print "#Errors:", errors
    return results


def predict_and_write(output_filepath, model, documents, word_vocabulary, pos_vocabulary, char_vocabulary, conf,
                      verbose=True):
    print "Evaluate model ..."
    tagging_scheme = AnnotationTools.get_tagging_scheme(conf.tagging_scheme)
    batch_generator = DatasetTools.BatchGenerator(documents, 100,
                                                  get_vectorizer(word_vocabulary, pos_vocabulary, char_vocabulary,
                                                                 tagging_scheme, input_only=True),
                                                  raw_data_name="document")
    reviews = defaultdict(list)
    for i, batches in enumerate(batch_generator):
        actual_batch_size = len(batches.text_input)
        i_text = i * conf.batch_size + actual_batch_size
        print("Pretrain Batch %d; Text %d:" % (i + 1, i_text))

        predicted_aspect_batch = model.predict_on_batch(batches)
        batches["predicted_aspect_output"] = predicted_aspect_batch

        for instance in DatasetTools.BatchIterator([batches]):
            sentence = instance["document"]
            predicted_aspect_probas = instance["predicted_aspect_output"]

            predicted_aspect_probas = predicted_aspect_probas[-len(sentence.tokens):, :]  # remove padding

            predicted_aspect_spans = set(tagging_scheme.encoding2spans(predicted_aspect_probas))

            reviews[sentence.review_id].append((sentence, predicted_aspect_spans))

    write_reviews_bs4(reviews, output_filepath)


def write_reviews_bs4(reviews, output_filepath):
    soup = BeautifulSoup(features='xml')
    reviews_tag = soup.new_tag("Reviews")
    soup.append(reviews_tag)
    for review_id, sentences in reviews.iteritems():
        review_tag = soup.new_tag("Review", rid=review_id)
        reviews_tag.append(review_tag)

        sentences_tag = soup.new_tag("sentences")
        review_tag.append(sentences_tag)
        for sentence, predicted_aspect_spans in sentences:
            sentence_tag = soup.new_tag("sentence", id=sentence.id)
            if sentence.out_of_scope:
                sentence_tag["OutOfScope"] = "TRUE"
            sentences_tag.append(sentence_tag)

            text_tag = soup.new_tag("text")
            text_tag.string = sentence.text
            sentence_tag.append(text_tag)

            opinions_tag = soup.new_tag("Opinions")
            sentence_tag.append(opinions_tag)

            if not sentence.out_of_scope:
                for predicted_span in predicted_aspect_spans:
                    token_start, token_end = predicted_span
                    char_start = sentence.tokenization[token_start][0]
                    char_end = sentence.tokenization[token_end - 1][1]
                    aspect_text = xml.sax.saxutils.escape(sentence.text[char_start:char_end])
                    if len(aspect_text) == 0:
                        print "########################################"
                        print "########################################"
                        print "########################################"
                        print "ASPECT TEXT IS EMPTY. PUT IN ANYTHING."
                        print sentence.id, predicted_span
                        print "########################################"
                        print "########################################"
                        print "########################################"
                        aspect_text = "--"
                    opinion_tag = soup.new_tag("Opinion", target=aspect_text)
                    opinion_tag["from"] = char_start
                    opinion_tag["to"] = char_end
                    opinions_tag.append(opinion_tag)

    with open(output_filepath, "w") as f:
        f.write(soup.prettify(encoding="utf-8"))
