import numpy
import pylab
from collections import defaultdict
from sklearn.manifold import TSNE

# this script applies T-SNE to the vectors in the files "char_<suffix>.txt" and "word_<suffix>.txt"

for model in ["char", "word"]:
    fig, ax = pylab.subplots()
    suffixes = ["ing", "ize", "less", "able", "ish", "ly"]
    Z = []
    S = []
    for s in suffixes:
        with open("../results/{}_{}.txt".format(model, s)) as f:
            X = f.readline()
            X = eval(X)
            Z += X

            S += [s] * len(X)
    Z = numpy.array(Z)
    print(Z.shape)
    tsne = TSNE()
    Z = tsne.fit_transform(Z)
    print(Z.shape)

    grouped_low_dim_vectors = defaultdict(list)
    for s, z in zip(S, Z):
        grouped_low_dim_vectors[s].append(z)

    for suffix, vecs in grouped_low_dim_vectors.iteritems():
        with open("../results/tsne_{}_{}.txt".format(model, suffix), "w") as f:
            vecs = numpy.array(vecs)
            print suffix, len(vecs)
            f.write(str(vecs.tolist()))
