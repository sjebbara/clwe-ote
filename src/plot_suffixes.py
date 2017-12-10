import numpy
import pylab

# this script simply visualizes the vectors in the files "tsne_char_<suffix>.txt" and "tsne_word_<suffix>.txt" which are
# generated in "analyze_trained_model".

suffix_colors = dict([("ing", "r"), ("ly", "g"), ("able", "b"), ("ish", "c"), ("less", "m"), ("ize", "y")])
suffix_markers = dict([("ing", "x"), ("ly", "o"), ("able", "+"), ("ish", "d"), ("less", "*"), ("ize", "v")])
suffixes = suffix_colors.keys()
suffix_colors["other"] = "k"

for model in ["char", "word"]:
    fig, ax = pylab.subplots()
    suffixes = ["ing", "ize", "less", "able", "ish", "ly"]
    for s in suffixes:
        with open("../results/tsne_{}_{}.txt".format(model, s)) as f:
            X = f.readline()
            X = eval(X)
            X = numpy.array(X)
            c = suffix_colors[s]
            m = suffix_markers[s]
            ax.scatter(X[:, 0], X[:, 1], facecolor=c, edgecolor=c, label="-" + s, marker=m)
    pylab.tight_layout()
    pylab.legend(loc="lower right")
    pylab.show()
