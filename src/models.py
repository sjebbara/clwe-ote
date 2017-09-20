from keras.engine import Input
from keras.engine import Model
from keras.layers import merge
from keras.regularizers import WeightRegularizer
from keras.layers import Embedding, TimeDistributed, Dense, Dropout, ELU, Bidirectional
from keras.optimizers import Adam

from custom_layers import BetterTimeDistributed, ELUGRU


def bielugru(word_input_size, word_embedding_size, sequence_embedding_size, n_tags, word_dropout, rnn_dropout_W,
             rnn_dropout_U, l2, embedding_weights=None, **kwargs):
    default_initialization = "he_normal"
    default_activation = ELU

    text_input = Input(shape=(None,), dtype='int32', name='text_input')

    word_embeddings = Embedding(input_dim=word_input_size, output_dim=word_embedding_size, weights=embedding_weights,
                                dropout=word_dropout, name="word_embeddings")(text_input)

    sequence_embedding = word_embeddings
    # TEXT RNN

    bidirectional_tag_sequence_output = Bidirectional(
        ELUGRU(output_dim=sequence_embedding_size / 2, return_sequences=True, init=default_initialization,
               dropout_W=rnn_dropout_W, dropout_U=rnn_dropout_U), merge_mode="concat")(sequence_embedding)

    tag_sequence_output = TimeDistributed(Dense(n_tags, activation='softmax', W_regularizer=WeightRegularizer(l2=l2)),
                                          name="aspect_output")(bidirectional_tag_sequence_output)
    model = Model(input=[text_input], output=[tag_sequence_output])
    adam = Adam(clipnorm=5.0)
    model.compile(optimizer=adam, loss={'aspect_output': "categorical_crossentropy"}, sample_weight_mode="temporal")
    model._make_train_function()
    model._make_predict_function()
    return model,


def char_bielugru(word_input_size, word_embedding_size, char_input_size, char_embedding_size, sequence_embedding_size,
                  n_tags, word_dropout, rnn_dropout_W, rnn_dropout_U, dropout, l2, embedding_weights=None, **kwargs):
    default_initialization = "he_normal"
    default_activation = ELU

    text_input = Input(shape=(None,), dtype='int32', name='text_input')
    char_input = Input(shape=(None, None,), dtype='int32', name='char_input')

    char_word_input = Input(shape=(None,), dtype='int32', name='char_word_input')

    word_embeddings = Embedding(input_dim=word_input_size, output_dim=word_embedding_size, weights=embedding_weights,
                                dropout=word_dropout, name="word_embeddings")(text_input)

    char_embedding_layer = Embedding(input_dim=char_input_size, output_dim=char_embedding_size, dropout=word_dropout,
                                     name="char_embeddings")
    char_embeddings = BetterTimeDistributed(char_embedding_layer)(char_input)

    ##################
    ##################
    char_word_model = Bidirectional(ELUGRU(output_dim=char_embedding_size, return_sequences=False), merge_mode="concat")

    char_word_embeddings = BetterTimeDistributed(char_word_model)(char_embeddings)
    char_word_embeddings = Dropout(dropout)(char_word_embeddings)
    char_word_embeddings = BetterTimeDistributed(Dense(char_embedding_size, W_regularizer=WeightRegularizer(l2=l2)))(
        char_word_embeddings)
    ##################
    ##################

    sequence_embedding = merge([word_embeddings, char_word_embeddings])
    # TEXT RNN

    bidirectional_tag_sequence_output = Bidirectional(
        ELUGRU(output_dim=sequence_embedding_size / 2, return_sequences=True, init=default_initialization,
               dropout_W=rnn_dropout_W, dropout_U=rnn_dropout_U), merge_mode="concat")(sequence_embedding)

    tag_sequence_output = TimeDistributed(Dense(n_tags, activation='softmax', W_regularizer=WeightRegularizer(l2=l2)),
                                          name="aspect_output")(bidirectional_tag_sequence_output)
    model = Model(input=[text_input, char_input], output=[tag_sequence_output])
    adam = Adam(clipnorm=5.0)
    model.compile(optimizer=adam, loss={'aspect_output': "categorical_crossentropy"}, sample_weight_mode="temporal")
    model._make_train_function()
    model._make_predict_function()

    ### CHAR WORD MODEL ###
    char_word_embedding = char_embedding_layer(char_word_input)
    char_word_embedding = char_word_model(char_word_embedding)

    char_word_model = Model(input=[char_word_input], output=[char_word_embedding])
    char_word_model._make_predict_function()
    return model, char_word_model
