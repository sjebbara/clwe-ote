from keras.engine import Input
from keras.engine import Model
from keras.regularizers import L1L2
from keras.layers import Embedding, TimeDistributed, Dense, Dropout, ELU, Bidirectional, concatenate
from keras.optimizers import Adam

from custom_layers import BetterTimeDistributed, ELUGRU


### Baseline model - Plain bidirectional GRU with ELU activation
def bielugru(word_input_size, word_embedding_size, sequence_embedding_size, n_tags, word_dropout, rnn_dropout_W,
             rnn_dropout_U, l2, word_embedding_weights, **kwargs):
    # define network inputs: words only
    text_input = Input(shape=(None,), dtype='int32', name='text_input')

    # map word indices to vector representation
    word_embeddings = Embedding(input_dim=word_input_size, output_dim=word_embedding_size,
                                weights=word_embedding_weights, name="word_embeddings")(text_input)
    # drop small portion of input vectors
    word_embeddings = Dropout(word_dropout)(word_embeddings)
    sequence_embedding = word_embeddings

    # apply text level BIGRU
    bidirectional_tag_sequence_output = Bidirectional(
        ELUGRU(sequence_embedding_size / 2, return_sequences=True, dropout=rnn_dropout_W,
               recurrent_dropout=rnn_dropout_U), merge_mode="concat")(sequence_embedding)

    # project hidden states to IOB tags
    tag_sequence_output = TimeDistributed(Dense(n_tags, activation='softmax', kernel_regularizer=L1L2(l2=l2)),
                                          name="aspect_output")(bidirectional_tag_sequence_output)

    # construct Model object and compile
    model = Model(inputs=[text_input], outputs=[tag_sequence_output])
    adam = Adam()
    model.compile(optimizer=adam, loss={'aspect_output': "categorical_crossentropy"}, sample_weight_mode="temporal")
    model._make_train_function()
    model._make_predict_function()
    return model,


### Proposed character enhanced model - include character level Bi-GRU to model word structure
def char_bielugru(word_input_size, word_embedding_size, char_input_size, char_embedding_size, sequence_embedding_size,
                  n_tags, word_dropout, rnn_dropout_W, rnn_dropout_U, dropout, l2, word_embedding_weights, **kwargs):
    # define network inputs: words and character indices
    text_input = Input(shape=(None,), dtype='int32', name='text_input')
    char_input = Input(shape=(None, None,), dtype='int32', name='char_input')

    # map word indices to vector representations
    word_embeddings = Embedding(input_dim=word_input_size, output_dim=word_embedding_size,
                                weights=word_embedding_weights, name="word_embeddings")(text_input)
    word_embeddings = Dropout(word_dropout)(word_embeddings)

    # map each character for each word to its vector representation
    char_embedding_layer = Embedding(input_dim=char_input_size, output_dim=char_embedding_size, name="char_embeddings")
    char_embeddings = BetterTimeDistributed(char_embedding_layer)(char_input)
    char_embeddings = Dropout(word_dropout)(char_embeddings)

    ##################
    # apply char-level BiGRU to every word
    char_word_model = Bidirectional(ELUGRU(char_embedding_size, return_sequences=False), merge_mode="concat")
    char_word_embeddings = BetterTimeDistributed(char_word_model)(char_embeddings)
    char_word_embeddings = Dropout(dropout)(char_word_embeddings)

    # project final states to fixed size representation
    char_word_embeddings = BetterTimeDistributed(Dense(char_embedding_size, kernel_regularizer=L1L2(l2=l2)))(
        char_word_embeddings)
    ##################

    # combine word and character emebeddings
    sequence_embedding = concatenate([word_embeddings, char_word_embeddings])

    # apply text level BIGRU
    bidirectional_tag_sequence_output = Bidirectional(
        ELUGRU(sequence_embedding_size / 2, return_sequences=True, dropout=rnn_dropout_W,
               recurrent_dropout=rnn_dropout_U), merge_mode="concat")(sequence_embedding)

    # project hidden states to IOB tags
    tag_sequence_output = TimeDistributed(Dense(n_tags, activation='softmax', kernel_regularizer=L1L2(l2=l2)),
                                          name="aspect_output")(bidirectional_tag_sequence_output)

    # construct Model object and compile
    model = Model(inputs=[text_input, char_input], outputs=[tag_sequence_output])
    adam = Adam()
    model.compile(optimizer=adam, loss={'aspect_output': "categorical_crossentropy"}, sample_weight_mode="temporal")
    model._make_train_function()
    model._make_predict_function()

    # construct Model object to obtain the character-level verctor representation for a single word
    char_word_input = Input(shape=(None,), dtype='int32', name='char_word_input')
    char_word_embedding = char_embedding_layer(char_word_input)
    char_word_embedding = char_word_model(char_word_embedding)

    char_word_model = Model(input=[char_word_input], output=[char_word_embedding])
    char_word_model._make_predict_function()
    return model, char_word_model
