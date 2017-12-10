# Improving Opinion-Traget Extraction with Character-Level Word Embeddings

Research code for the paper "Improving Opinion-Traget Extraction with Character-Level Word Embeddings", to be published at the workshop on "Subword and Character LEvel Models in NLP" at the EMNLP 2017 (https://sites.google.com/view/sclem2017/home)

---

## Abstract
> Fine-grained sentiment analysis is receiving increasing attention in recent
years. Extracting opinion target expressions (OTE) in reviews is often an important
step in fine-grained, aspect-based sentiment analysis. Retrieving this information from user-generated text, however, can be
difficult. Customer reviews, for instance, are prone to contain misspelled words and are
difficult to process due to their domain-specific language.
> In this work, we investigate whether character-level models can improve the
performance for the identification of opinion target expressions. We integrate information about the character structure of a word into a
sequence labeling system using character-level word embeddings and show their positive impact on the system's performance. Specifically, we obtain an increase by 3.3 points F1-score with respect to our
baseline model. In further experiments, we reveal encoded character patterns of the learned embeddings and give a nuanced view of the performance differences of both models.

## Paper
The Paper can be found here:
<https://pub.uni-bielefeld.de/publication/2913711>
and
<https://arxiv.org/abs/1709.06317>

---

## Bibtex:
```
@InProceedings{jebbara-cimiano:2017:SCLeM,
  author    = {Jebbara, Soufian  and  Cimiano, Philipp},
  title     = {Improving Opinion-Target Extraction with Character-Level Word Embeddings},
  booktitle = {Proceedings of the First Workshop on Subword and Character Level Models in NLP},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  pages     = {159--167},
  url       = {http://www.aclweb.org/anthology/W17-4124}
}

```

## The Code
The code is still in rough shape. I try to clean it soon.
If you have any qeustion, send me a message on Github or an email.

### Dependencies
* Keras (Version 1.2.0)
* NLPUtils <https://github.com/sjebbara/NLPUtils> (Commit a4a506af653e119385d096a9e26ddfbc42de93ac)
* Theano
