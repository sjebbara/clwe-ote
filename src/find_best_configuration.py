import os
import io
import ujson

import numpy
from nlputils import LearningTools

import AspectExtraction


def read_results(base_dirpath, conditions=None):
    experiments_dirnames = os.listdir(base_dirpath)
    experiments_dirnames = [d for d in experiments_dirnames if d.startswith("AspectBasedSentiment_Configuration_")]

    experiments = []
    for exp_dirname in experiments_dirnames:
        skip = False
        cv_dirnames = os.listdir(os.path.join(base_dirpath, exp_dirname))
        cv_dirnames = [d for d in cv_dirnames if d.startswith("cv-")]
        all_scores = []
        if len(cv_dirnames) != 5:
            print "WARNING: unexpected number of CVs:", exp_dirname, cv_dirnames
            skip = True
        else:
            for cv_dirname in cv_dirnames:
                try:
                    with io.open(os.path.join(base_dirpath, exp_dirname, cv_dirname, "scores.txt")) as f:
                        scores = [float(s.strip()) for s in f]
                        all_scores.append(scores)
                    with io.open(os.path.join(base_dirpath, exp_dirname, cv_dirname, "configuration.conf")) as f:
                        conf = "".join(f.readlines())
                        conf = ujson.loads(conf)
                        conf = LearningTools.Configuration(conf)

                        if conditions is not None and not all(conf[k] == v for k, v in conditions):
                            skip = True

                except IOError as er:
                    print "WARNING: necessary files not available:", exp_dirname, cv_dirnames
                    skip = True

        if len(set(len(scores) for scores in all_scores)) != 1:
            print "Skip", exp_dirname, ". Unfinished training."
            print conf
            skip = True

        if not skip:
            all_scores = numpy.array(all_scores)
            cv_scores = numpy.nanmean(all_scores, axis=0)
            cv_max_epoch = numpy.nanargmax(cv_scores)
            cv_max = cv_scores[cv_max_epoch]

            experiments.append((exp_dirname, conf, cv_max_epoch, cv_max, cv_scores, all_scores.T))

    return experiments


def list_results(base_dirpath, parameter_names, conditions=None):
    experiments = read_results(base_dirpath, conditions)
    experiments = sorted(experiments, key=lambda e: e[3], reverse=True)
    print "+++ RESULTS: +++"
    print conditions
    for exp in experiments:
        print exp[3], exp[0], exp[2], "\t".join(["{}: {}".format(p, exp[1][p]) for p in parameter_names])


def find_best(base_dirpath, conditions=None):
    experiments_dirnames = os.listdir(base_dirpath)
    experiments_dirnames = [d for d in experiments_dirnames if d.startswith("AspectBasedSentiment_Configuration_")]

    best_score = 0
    best_experiment = tuple()
    for exp_dirname in experiments_dirnames:
        skip = False
        cv_dirnames = os.listdir(os.path.join(base_dirpath, exp_dirname))
        cv_dirnames = [d for d in cv_dirnames if d.startswith("cv-")]
        all_scores = []
        if len(cv_dirnames) != 5:
            print "WARNING: unexpected number of CVs:", exp_dirname, cv_dirnames
            skip = True
        else:
            for cv_dirname in cv_dirnames:
                try:
                    with io.open(os.path.join(base_dirpath, exp_dirname, cv_dirname, "scores.txt")) as f:
                        scores = [float(s.strip()) for s in f]
                        all_scores.append(scores)
                    with io.open(os.path.join(base_dirpath, exp_dirname, cv_dirname, "configuration.conf")) as f:
                        conf = "".join(f.readlines())
                        conf = ujson.loads(conf)
                        conf = LearningTools.Configuration(conf)

                        if conditions is not None and not all(conf[k] == v for k, v in conditions):
                            skip = True

                except IOError as er:
                    print "WARNING: necessary files not available:", exp_dirname, cv_dirnames
                    skip = True

        if not skip:
            all_scores = numpy.array(all_scores)
            cv_scores = numpy.nanmean(all_scores, axis=0)
            cv_max_epoch = numpy.nanargmax(cv_scores)
            cv_max = cv_scores[cv_max_epoch]

            if cv_max > best_score:
                best_score = cv_max
                best_experiment = (exp_dirname, conf, cv_max_epoch, cv_max, cv_scores, all_scores.T)

    print "+++ RESULTS: +++"
    print conditions
    print best_experiment[0]
    print best_score, best_experiment[2]
    print best_experiment[1]
    # for x in best_experiment:
    #     print x


if __name__ == "__main__":
    base_dirpath = os.path.join(AspectExtraction.EXPERIMENTS_OUTPUT_DIR, "AspectBasedSentiment_2017-09-20_13:11:06")
    list_results(base_dirpath, ["sequence_embedding_size", "char_embedding_size", "top_k_vocab"],
                 [("dataset", "semeval2016restaurant"), ("model", "bielugru")])
    list_results(base_dirpath, ["sequence_embedding_size", "char_embedding_size", "top_k_vocab"],
                 [("dataset", "semeval2016restaurant"), ("model", "char_bielugru")])
