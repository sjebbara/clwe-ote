import data
from nlputils import DataTools

for top_k in [10000, 20000, 50000]:
    word_embeddings = DataTools.Embedding()
    word_embeddings.load("/vol/scstaff/sjebbara/data/embeddings/amazon_review_corpus_en_100D_advanced_W.npy",
                         "/vol/scstaff/sjebbara/data/embeddings/amazon_review_corpus_en_100D_advanced_vocab.txt")
    word_embeddings.trim_embeddings(vocab_trim=["<UNK>"], top_k=top_k)
    word_embeddings.vocabulary.set_unknown(word_embeddings.vocabulary.get_index("<UNK>"))
    word_embeddings.add("<pad>", 0, vector_init="zeros")
    word_embeddings.vocabulary.set_padding(0)
    word_embeddings.add(data.SENTENCE_START_TOKEN, 1, vector_init="zeros")
    word_embeddings.add(data.SENTENCE_END_TOKEN, 2, vector_init="zeros")
    word_embeddings.save("../res/embeddings/", "amazon_review_corpus_en_100D_advanced_top-{}".format(top_k))
