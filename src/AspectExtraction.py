import io
import os
import numpy
import sys
import data
import models
import process

from collections import Counter
from nlputils import DataTools
from nlputils import LearningTools
from nlputils import LexicalTools
from unidecode import unidecode

EXPERIMENTS_OUTPUT_DIR = "/home/sjebbara/data/AspectBasedSentiment/experiments/"


def get_base_config():
    conf = LearningTools.Configuration()
    conf.w2v_type = "amazon_review_corpus_en_100D_advanced"
    conf.tokenization_style = "advanced2"
    conf.text_preprocessing = LexicalTools.lower
    conf.max_documents = None
    conf.batch_size = 5
    conf.n_cross_validation = 5
    conf.opinion_filter = data.opinion_non_null_filter
    conf.n_epochs = 100
    conf.top_k_vocab = 50000
    conf.dataset = "semeval2016restaurant"
    conf.data_split = "cv"
    conf.scope = "sentence"
    conf.word_dropout = 0.5
    conf.dropout = 0.5
    conf.rnn_dropout_W = 0.5
    conf.rnn_dropout_U = 0.5
    conf.cnn_dropout = 0.5
    conf.tagging_scheme = "IOB"
    conf.n_tags = 3
    conf.pos_context_size = 3
    conf.pos_embedding_size = 20
    conf.sequence_embedding_size = 100
    conf.word_context_size = 3
    conf.tag_embedding_size = None
    conf.use_pos = False
    conf.alternate_optimizers = False
    conf.char_embedding_size = 20
    conf.l2 = 1e-5
    conf.depth = 3
    conf.threshold = 0.5
    return conf


def get_test_configurations():
    configs = []

    ###############################################
    conf = get_base_config()
    conf.model = models.char_bielugru.func_name
    conf.use_pos = False
    conf.data_split = "original"
    conf.top_k_vocab = 50000
    conf.sequence_embedding_size = 100
    conf.char_embedding_size = 100
    configs.append(conf)

    ###############################################
    conf = get_base_config()
    conf.model = models.bielugru.func_name
    conf.use_pos = False
    conf.data_split = "original"
    conf.top_k_vocab = 50000
    conf.sequence_embedding_size = 60
    conf.char_embedding_size = 100
    configs.append(conf)
    return configs


def get_configurations():
    configs = []

    for sequence_embedding_size in [50, 100, 200]:
        for top_k_vocab in [10000, 20000, 50000]:
            for model in [models.char_bielugru.func_name]:
                for char_embedding_size in [20, 50, 100]:
                    conf = get_base_config()
                    conf.model = model
                    conf.sequence_embedding_size = sequence_embedding_size
                    conf.char_embedding_size = char_embedding_size
                    conf.top_k_vocab = top_k_vocab
                    configs.append(conf)

            for model in [models.bielugru.func_name]:
                conf = get_base_config()
                conf.model = model
                conf.sequence_embedding_size = sequence_embedding_size
                conf.char_embedding_size = 100
                conf.top_k_vocab = top_k_vocab
                configs.append(conf)

    return configs


def main(conf, plot_scores=True):
    conf.experiment_id = "AspectBasedSentiment_Configuration_" + LearningTools.get_timestamp()
    print conf
    base_dirpath = os.path.join(EXPERIMENTS_OUTPUT_DIR, "AspectBasedSentiment_" + conf.timestamp, conf.experiment_id)
    os.makedirs(base_dirpath)
    print "read dataset..."

    if conf.data_split == "original":
        train_dataset = data.read_semeval2016_restaurant_train(conf.scope, conf.text_preprocessing,
                                                               conf.tokenization_style, conf.sentence_filter,
                                                               conf.opinion_filter)
        blind_test_documents = data.read_semeval2016_restaurant_blind_test(conf.scope, conf.text_preprocessing,
                                                                           conf.tokenization_style).sentences
        train_documents, val_documents = DataTools.custom_split(train_dataset.sentences, 0.8, seed=7)
        train_test_splits = [(train_documents, val_documents)]
    elif conf.data_split == "custom":
        dataset = data.read_semeval2016_restaurant_train(conf.scope, conf.text_preprocessing, conf.tokenization_style,
                                                         conf.sentence_filter, conf.opinion_filter)
        train_documents, test_documents = DataTools.custom_split(dataset.sentences, 0.8, seed=7)
        train_test_splits = [(train_documents, test_documents)]
    elif conf.data_split == "cv":
        train_dataset = data.read_semeval2016_restaurant_train(conf.scope, conf.text_preprocessing,
                                                               conf.tokenization_style, conf.sentence_filter,
                                                               conf.opinion_filter)
        train_test_splits = DataTools.cross_validation_split(train_dataset.sentences, conf.n_cross_validation, seed=7)

    word_embeddings = DataTools.Embedding()
    word_embeddings.load(
        "../res/embeddings/amazon_review_corpus_en_100D_advanced_top-{}_W.npy".format(conf.top_k_vocab),
        "../res/embeddings/amazon_review_corpus_en_100D_advanced_top-{}_vocab.txt".format(conf.top_k_vocab))
    word_embeddings.vocabulary.set_padding(word_embeddings.vocabulary.get_index("<pad>"))
    word_embeddings.vocabulary.set_unknown(word_embeddings.vocabulary.get_index("<UNK>"))

    conf.word_input_size = len(word_embeddings.vocabulary)
    conf.word_embedding_size = word_embeddings.W.shape[1]

    char_vocabulary = DataTools.Vocabulary()
    char_vocab = Counter(c for w in word_embeddings.vocabulary.vocab for c in unidecode(w) if c != " ")
    print char_vocab.most_common()
    char_vocabulary.init_from_vocab(char_vocab)
    char_vocabulary.add_padding("<0>", 0)
    char_vocabulary.add_unknown("<?>", 1)
    char_vocabulary.save(os.path.join(base_dirpath, "char_vocabulary.txt"))

    conf.char_input_size = len(char_vocabulary)

    pos_vocabulary = LexicalTools.pos_vocabulary
    conf.pos_input_size = len(pos_vocabulary)
    if not conf.use_pos:
        pos_vocabulary = None

    if plot_scores:
        score_plot = LearningTools.ScorePlot("Aspect Extraction", n_cross_validation=len(train_test_splits),
                                             n_epochs=conf.n_epochs)

    for n, (train_documents, val_documents) in enumerate(train_test_splits):
        cv_dirpath = os.path.join(base_dirpath, "cv-{}".format(n + 1))
        os.makedirs(cv_dirpath)
        conf.save(os.path.join(cv_dirpath, "configuration.conf"))

        best_epoch = 0
        best_score = 0

        model_name = "{}_{}_n-docs={}_batch-size={}_epochs={}_s-size={}_c-size={}_topK={}".format(conf.model,
                                                                                                  conf.dataset,
                                                                                                  conf.max_documents,
                                                                                                  conf.batch_size,
                                                                                                  conf.n_epochs,
                                                                                                  conf.sequence_embedding_size,
                                                                                                  conf.char_embedding_size,
                                                                                                  conf.top_k_vocab)
        print "Model:", model_name
        print conf
        model_fn = models.__dict__[conf.model]
        modelz = model_fn(word_embedding_weights=[word_embeddings.W], **conf)
        model = modelz[0]

        model.summary()

        models_dirpath = os.path.join(cv_dirpath, "models")
        os.makedirs(models_dirpath)

        best_model = (0, 0, None)
        for e in range(conf.n_epochs):
            process.train_aspects(model, train_documents, word_embeddings.vocabulary, pos_vocabulary, char_vocabulary,
                                  conf, e, n_epochs=conf.n_epochs)
            print "\n\nEvaluate on TRAIN"
            train_results = process.evaluate_aspects(model, train_documents, word_embeddings.vocabulary, pos_vocabulary,
                                                     char_vocabulary, conf, verbose=False)
            print "\n\nEvaluate on VAL"
            val_results = process.evaluate_aspects(model, val_documents, word_embeddings.vocabulary, pos_vocabulary,
                                                   char_vocabulary, conf)

            if conf.data_split == "original":
                predict_documents = blind_test_documents
            else:
                predict_documents = val_documents

            process.predict_and_write(os.path.join(cv_dirpath, "epoch={}_predicted_aspects.xml".format(e + 1)), model,
                                      predict_documents, word_embeddings.vocabulary, pos_vocabulary, char_vocabulary,
                                      conf)

            f1_train, p_train, r_train = train_results.score(min_confidence=0)
            f1, p, r = val_results.score(min_confidence=0)

            if plot_scores:
                score_plot.add(n, e, f1_train, "F1-Train")
                score_plot.add(n, e, f1, "F1")
                score_plot.add(n, e, p, "P")
                score_plot.add(n, e, r, "R")

                score_plot.print_scores("F1")

            if e > 1:
                if best_model is None or f1 > best_model[0]:
                    model.save_weights(os.path.join(models_dirpath, "weights@{}.h5".format(e + 1)))
                    best_model = (f1, e)

            with io.open(os.path.join(cv_dirpath, "scores.txt".format(e + 1)), "a") as f:
                f.write(u"{:.6f}\n".format(f1))

        print [w.shape for w in model.get_weights()]

        print "best model:", best_model
        ############ Save Model Weights ############
        model.save_weights(os.path.join(models_dirpath, "final_weights.h5"))

    if plot_scores:
        numpy.save("../results/scores_{}.npy".format(conf.model), score_plot.scores["F1"])
    print "Best Epoch {} with score {}".format(best_epoch, best_score)


if __name__ == "__main__":
    plot_scores = True
    if len(sys.argv) > 1:
        plot_scores = not sys.argv[1].lower() == "--no-plot"
        print "Plot: ", plot_scores
    timestamp = LearningTools.get_timestamp()

    configs = get_test_configurations()
    # configs = get_configurations()
    for config in configs:
        config.timestamp = timestamp
        print timestamp
        main(config, plot_scores)
