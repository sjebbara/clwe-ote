import semevalabsa.datasets
from nlputils import AnnotationTools
from nlputils import DataTools
from nlputils import LexicalTools
from semevalabsa.datasets import Opinion, Sentence

SENTENCE_START_TOKEN = "<start>"
SENTENCE_END_TOKEN = "<end>"
UNDEFINED_ASPECT = "<NULL>"


def opinion_non_null_filter(o):
    return o.target is not None


def document_filter_min_max_characters(min_chars, max_chars):
    def fn(d):
        return min_chars <= len(d.text) <= max_chars

    return fn


def document_filter_max_length(max_sentences, max_tokens):
    def fn(d):
        return len(d.sentences) <= max_sentences and max(len(s.tokens) for s in d.sentences) <= max_tokens

    return fn


def document_filter_aspects(required_aspects):
    def fn(d):
        return all(a in d.ratings and d.ratings[a] != -1 for a in required_aspects)

    return fn


def _build_sentence_dataset(reviews, text_preprocessing, tokenization_style, opinion_filter):
    dataset = SentenceDataset()

    for review in reviews:
        for sentence in review.sentences:
            s = SemevalSentence(sentence)
            s.preprocess_text(preprocessing_function=text_preprocessing)
            s.tokenize(tokenization_style=tokenization_style)
            s.opinions = filter(opinion_filter, s.opinions)
            dataset.add(s)

    dataset.build()
    dataset.compute_pos_tags()
    return dataset


def read_semeval2016_restaurant_train(scope, text_preprocessing, tokenization_style, sentence_filter, opinion_filter,
                                      filename="/home/sjebbara/datasets/SemEval2016 Task5/en/ABSA16_Restaurants_Train_SB1_v2.xml"):
    reviews = semevalabsa.datasets.read_semeval2016_task5_subtask1(filename)

    dataset = _build_sentence_dataset(reviews, text_preprocessing, tokenization_style, opinion_filter)
    return dataset


def read_semeval2016_restaurant_test(scope, text_preprocessing, tokenization_style, sentence_filter, opinion_filter,
                                     filename="/home/sjebbara/datasets/SemEval2016 Task5/en/EN_REST_SB1_TEST.xml.gold"):
    reviews = semevalabsa.datasets.read_semeval2016_task5_subtask1(filename)

    dataset = _build_sentence_dataset(reviews, text_preprocessing, tokenization_style, opinion_filter)
    return dataset


def read_semeval2016_restaurant_blind_test(scope, text_preprocessing, tokenization_style,
                                           filename="/home/sjebbara/datasets/SemEval2016 Task5/en/EN_REST_SB1_TEST.xml.A"):
    reviews = semevalabsa.datasets.read_semeval2016_task5_subtask1(filename)

    dataset = _build_sentence_dataset(reviews, text_preprocessing, tokenization_style, None)
    return dataset


def read_semeval2016(filepath, scope, text_preprocessing, tokenization_style, sentence_filter, opinion_filter):
    reviews = semevalabsa.datasets.read_semeval2016_task5_subtask1(filepath)
    dataset = _build_sentence_dataset(reviews, text_preprocessing, tokenization_style, opinion_filter)
    return dataset


class SemevalSentence(Sentence):
    def __init__(self, sentence=None):
        Sentence.__init__(self)
        self.review_id = None
        self.id = None
        self.text = None
        self.normalized_text = None
        self.tokens = None
        self.token_pos_tags = None
        self.opinions = []
        self.tokenization = []
        self.out_of_scope = None
        if sentence:
            self.review_id = sentence.review_id
            self.id = sentence.id
            self.text = sentence.text
            self.opinions = []
            for o in sentence.opinions:
                self.opinions.append(SemevalOpinion(o))
            self.out_of_scope = sentence.out_of_scope

    def preprocess_text(self, preprocessing_function):
        if preprocessing_function and self.text:
            self.normalized_text = preprocessing_function(self.text)
        for o in self.opinions:
            o.preprocess_text(preprocessing_function)

    def tokenize(self, tokenization_style, stopwords=None):
        if self.normalized_text:
            text = self.normalized_text
        else:
            text = self.text

        starts, ends, self.tokens = LexicalTools.tokenization(text, tokenization_style=tokenization_style,
                                                              stopwords=stopwords)
        self.tokenization = zip(starts, ends)
        for o in self.opinions:
            if o.target:
                o.tokenize(tokenization_style, stopwords, self.tokenization)

                doc_token_text = " ".join(self.tokens[o.token_start: o.token_end])
                opinion_token_text = " ".join(o.tokens)
                if doc_token_text != opinion_token_text:
                    print "TOKEN ERROR: ", doc_token_text, "vs.", opinion_token_text

    def get_number_of_opinions(self):
        return len(self.opinions)


class SemevalOpinion(Opinion):
    def __init__(self, opinion=None):
        Opinion.__init__(self)
        self.target = None
        self.category = None
        self.entity = None
        self.attribute = None
        self.polarity = None
        self.start = None
        self.end = None

        self.normalized_target = None
        self.tokens = None
        self.token_start = None
        self.token_end = None
        self.confidences = None

        if opinion:
            self.target = opinion.target
            self.category = opinion.category
            self.entity = opinion.entity
            self.attribute = opinion.attribute
            self.polarity = opinion.polarity
            self.start = opinion.start
            self.end = opinion.end

    def __str__(self):
        if self.target:
            s = u"[{}; {}] '{}' ({}-{}, {})".format(self.category, self.polarity, self.target, self.start, self.end,
                                                    self.confidences)
        else:
            s = u"[{}; {}]".format(self.category, self.polarity)
        return s.encode("utf-8")

    def preprocess_text(self, preprocessing_function):
        if preprocessing_function and self.target:
            self.normalized_target = preprocessing_function(self.target)

    def tokenize(self, tokenization_style, stopwords=None, document_tokenization=None):
        if self.normalized_target:
            target = self.normalized_target
        else:
            target = self.target

        if target:
            self.tokens = LexicalTools.tokenize(target, tokenization_style=tokenization_style, stopwords=stopwords)

            self.token_start, self.token_end = AnnotationTools.find_matching_spans(document_tokenization, self.start,
                                                                                   self.end)


class SentenceDataset:
    def __init__(self):
        self.sentences = []
        self.polarity_table = None
        self.category_table = None
        self.entity_table = None
        self.attribute_table = None

    def add(self, s):
        self.sentences.append(s)

    def add_all(self, sentences):
        self.sentences += sentences

    def build(self):
        polarities = set()
        categories = set()
        entities = set()
        attributes = set()
        for s in self.sentences:
            for o in s.opinions:
                polarities.add(o.polarity)
                categories.add(o.category)
                entities.add(o.entity)
                attributes.add(o.attribute)

        self.polarity_table = DataTools.Vocabulary()
        self.polarity_table.init_from_vocab(polarities)
        self.category_table = DataTools.Vocabulary()
        self.category_table.init_from_vocab(categories)
        self.entity_table = DataTools.Vocabulary()
        self.entity_table.init_from_vocab(entities)
        self.attribute_table = DataTools.Vocabulary()
        self.attribute_table.init_from_vocab(attributes)

    def compute_pos_tags(self):
        all_sentences = [s.tokens for s in self.sentences]
        all_pos = LexicalTools.get_pos_tags(all_sentences, as_index=False)

        for i, (s, pos) in enumerate(zip(self.sentences, all_pos)):
            s.token_pos_tags = pos
            for o in s.opinions:
                if o.target:
                    o.token_pos_tags = pos[o.token_start:o.token_end]
                else:
                    o.token_pos_tags = None
