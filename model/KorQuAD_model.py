import modeling_electra as modeling
import tensorflow as tf

import DataHolder3 as DataHolder

from utils import Fully_Connected
from utils import Highway_Network_Fullyconnceted

import numpy as np

import optimization

import tokenization

from HTML_Utils import *
import os
import json

from evaluate2 import f1_score
from evaluate2 import exact_match_score

import Chuncker
import Name_Tagging
import Ranking_ids

from time import sleep
from HTML_Processor import process_document
import Table_Holder
from Table_Holder import detect_simple_num_word, detect_num_word, get_space_num_lists

from modeling import get_shape_list
from modeling import transformer_model
from modeling import create_attention_mask_from_input_mask
from transformers import ElectraTokenizer

import DataHolder_test

table_holder = Table_Holder.Holder()
chuncker = Chuncker.Chuncker()


def embedding_postprocessor(input_tensor,
                            col_ids,
                            row_type_ids,
                            hidden_size=768,
                            initializer_range=0.02,):
    """Performs various post-processing on a word embedding tensor.

    Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

    Returns:
    float tensor with same shape as `input_tensor`.

    Raises:
    ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    #cols
    cols_table = tf.get_variable(
        name='col_embedding',
        shape=[50, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(col_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=50)
    token_type_embeddings = tf.matmul(one_hot_ids, cols_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

    #rows
    rows_table = tf.get_variable(
        name='row_embedding',
        shape=[250, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(row_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=250)
    token_type_embeddings = tf.matmul(one_hot_ids, rows_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

    return output


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def masked_softmax(logits, mask, dim):
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30)  # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask)  # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


def get_variables_with_name(name, train_only=True, printable=False):
    """Get variable list by a given name scope.
    Examples
    ---------
    >>> dense_vars = tl.layers.get_variable_with_name('dense', True, True)
    """
    print("  [*] geting variables with %s" % name)
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try: # TF1.0
            t_vars = tf.global_variables()
        except: # TF0.12
            t_vars = tf.all_variables()

    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars


class KoNET:
    def __init__(self, firstTraining, testCase=False):
        self.chuncker = Chuncker.Chuncker()
        self.first_training = firstTraining

        self.save_path = 'E:\\cell_bert_qa_koelectra\\cell_bert_model.ckpt'
        self.bert_path = 'E:\\cell_bert_koelectra\\cell_bert_model.ckpt'

        #self.save_path = 'E:\\dual_mrc\\dual_mrc.ckpt'
        #self.bert_path = 'F:\\cell_bert_att\\bert_model.ckpt'

        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_segments = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_cols = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_rows = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_names = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_rankings = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.query_mask = tf.placeholder(shape=[None, None], dtype=tf.float32)

        self.column_label = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.row_label = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.input_mask_rows = tf.placeholder(shape=[None, 50], dtype=tf.int32, name='mask_rows')
        self.input_mask_cols = tf.placeholder(shape=[None, 30], dtype=tf.int32, name='mask_cols')

        self.numeric_space = tf.placeholder(shape=[None, None, None], dtype=tf.int32)
        self.numeric_mask = tf.placeholder(shape=[None, None, None], dtype=tf.int32)

        self.start_label = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.stop_label = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.start_label2 = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.stop_label2 = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.rank_label = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.adjacency_mat_bigger = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
        self.adjacency_mat_smaller = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])

        self.processor = DataHolder.DataHolder()

        self.keep_prob = 0.9
        if testCase is True:
            self.keep_prob = 1.0

        self.testCase = testCase

        self.sess = None
        self.prediction_start = None
        self.prediction_stop = None

        self.column_size = 15
        self.row_size = 50

    def model_setting(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        self.sess = tf.Session(config=config)

        model, bert_variables, seqeunce_output = self.create_model(input_ids=self.input_ids,
                                                                   input_mask=self.input_mask,
                                                                   input_segments=self.input_segments,
                                                                   )
        probs_start, _ = \
            self.get_qa_probs2(seqeunce_output, model.get_pooled_output(), is_training=False)

        vars_ = get_variables_with_name('pointer_net1', True, True)
        vars2_ = get_variables_with_name('MRC_block', True, True)

        bert_vars = get_variables_with_name('bert', True, True)
        for var in vars_:
            bert_vars.append(var)
        for var in vars2_:
            bert_vars.append(var)

        self.prediction_start = probs_start
        self.prediction_start = tf.nn.softmax(self.prediction_start, axis=-1)

        self.sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver(var_list=bert_vars)
        saver.restore(self.sess, self.save_path)

    def create_model(self, input_ids, input_mask, input_segments, is_training=True):
        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base.json')

        if self.testCase is True:
            is_training = False

        input_mask = tf.where(input_ids > 0, tf.ones_like(input_ids), tf.zeros_like(input_ids))

        model = modeling.BertModel(
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_segments,
            scope='electra',
            bert_config=bert_config
        )

        bert_variables = tf.global_variables()

        return model, bert_variables, model.get_sequence_output()

    def get_qa_loss2(self, logits, label):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label)
            loss = tf.reduce_mean(loss1)
        return loss

    def get_qa_loss(self, logit1, logit2):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit1, labels=self.start_label)
            loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit2, labels=self.stop_label)

            loss = tf.reduce_mean(loss1) + tf.reduce_mean(loss2)
        return loss, loss1, loss2

    def get_qa_probs2(self, model_output, pooled_output, is_training=False):
        """Get loss and log probs for the next sentence prediction."""

        keep_prob = 0.9

        if is_training is False:
            keep_prob = 1.0

        with tf.variable_scope("MRC_block"):
            model_output = Fully_Connected(model_output, output=768, name='hidden1', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=512, name='hidden2', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=256, name='hidden3', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

        with tf.variable_scope("pointer_net1"):
            model_output1 = Fully_Connected(model_output, output=128, name='hidden4', activation=gelu)
            model_output1 = tf.nn.dropout(model_output1, keep_prob=keep_prob)

            log_probs_s = Fully_Connected(model_output1, output=1, name='pointer_start', activation=gelu, reuse=False)
            log_probs_e = Fully_Connected(model_output1, output=1, name='pointer_stop', activation=gelu, reuse=False)
            log_probs_s = tf.squeeze(log_probs_s, axis=-1)
            log_probs_e = tf.squeeze(log_probs_e, axis=-1)

        bert_variables = tf.global_variables()

        return log_probs_s, bert_variables

    def get_qa_probs(self, model_output, is_training=False):
        """Get loss and log probs for the next sentence prediction."""

        keep_prob = 0.9

        if is_training is False:
            keep_prob = 1.0

        with tf.variable_scope("MRC_block"):
            model_output = Fully_Connected(model_output, output=768, name='hidden1', activation=tf.nn.leaky_relu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=512, name='hidden2', activation=tf.nn.leaky_relu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

        with tf.variable_scope("pointer_net1"):
            log_probs_s = Fully_Connected(model_output, output=1, name='pointer_start1', activation=None, reuse=False)
            log_probs_e = Fully_Connected(model_output, output=1, name='pointer_stop1', activation=None, reuse=False)
            log_probs_s = tf.squeeze(log_probs_s, axis=2)
            log_probs_e = tf.squeeze(log_probs_e, axis=2)

        return log_probs_s, log_probs_e

    def Table_Memory_Network(self, sequence_output, hops=3, dropout=0.2):
        #sequence_output = sequence_output + space_states

        cell_fw_col = tf.nn.rnn_cell.GRUCell(768)
        cell_fw_col = tf.nn.rnn_cell.DropoutWrapper(cell_fw_col, input_keep_prob=self.keep_prob,
                                                   output_keep_prob=self.keep_prob)
        cell_fw_row = tf.nn.rnn_cell.GRUCell(768)
        cell_fw_row = tf.nn.rnn_cell.DropoutWrapper(cell_fw_row, input_keep_prob=self.keep_prob,
                                                   output_keep_prob=self.keep_prob)

        row_one_hot = tf.one_hot(self.input_rows, depth=100)
        row_one_hot = tf.transpose(row_one_hot, perm=[0, 2, 1])

        column_one_hot = tf.one_hot(self.input_cols, depth=50)
        column_one_hot = tf.transpose(column_one_hot, perm=[0, 2, 1])

        column_wise_memory = tf.matmul(column_one_hot, sequence_output)
        row_wise_memory = tf.matmul(row_one_hot, sequence_output)

        reuse = False

        for h in range(hops):
            print('hop:', h)

            with tf.variable_scope("column_memory_block" + str(h)):
                column_wise_memory = modeling.attention_layer_modified(
                                                  from_tensor=column_wise_memory,
                                                  to_tensor=sequence_output,
                                                  attention_mask=column_one_hot,
                                                  )

            column_wise_memory = Fully_Connected(column_wise_memory, 768, 'hidden_col' + str(0), gelu, reuse=reuse)
            column_wise_memory = modeling.dropout(column_wise_memory, dropout)

            with tf.variable_scope("row_memory_block" + str(h)):
                row_wise_memory = modeling.attention_layer_modified(
                                                from_tensor=row_wise_memory,
                                                to_tensor=sequence_output,
                                                attention_mask=row_one_hot)

            row_wise_memory = Fully_Connected(row_wise_memory, 768, 'hidden_row' + str(0), gelu, reuse=reuse)
            row_wise_memory = modeling.dropout(row_wise_memory, dropout)

            reuse = True

        # todo: RNN Code for Context Encoding
        with tf.variable_scope("rnn_model_col"):
            column_wise_memory, state_fw = tf.nn.dynamic_rnn(
                inputs=column_wise_memory, cell=cell_fw_col,
                sequence_length=seq_length(column_wise_memory), dtype=tf.float32, time_major=False)

        with tf.variable_scope("rnn_model_row"):
            row_wise_memory, state_fw = tf.nn.dynamic_rnn(
                inputs=row_wise_memory, cell=cell_fw_row,
                sequence_length=seq_length(row_wise_memory), dtype=tf.float32, time_major=False)

        return column_wise_memory, row_wise_memory

    def Training(self, is_Continue, training_epoch):
        dropout = 0.2

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.99

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids, self.input_mask,
                                                                       self.input_segments, is_training=True)
            """
            initializer_range = 0.02
            space_embedding = tf.get_variable(
                name='space_embedding',
                shape=[150, 768],
                initializer=create_initializer(initializer_range))
            
            space_ids = tf.reshape(self.numeric_space, shape=[-1, 500])
            space_mask = tf.expand_dims(self.numeric_mask, axis=-1)

            space_representation = tf.nn.embedding_lookup(space_embedding, space_ids)
            space_representation = tf.reshape(space_representation, shape=[-1, 10, 500, 768])

            space_representation = space_representation * tf.cast(space_mask, dtype=tf.float32)
            space_representation = tf.reduce_sum(space_representation, axis=1)

            input_shape = get_shape_list(sequence_output, expected_rank=3)
            batch_size = input_shape[0]
            seq_length = input_shape[1]
            """
            column_memory, row_memory = self.Table_Memory_Network(sequence_output=sequence_output,
                                                                  hops=5)
            #bert_variables = tf.global_variables()

            row_one_hot = tf.one_hot(self.input_rows, depth=100)
            column_one_hot = tf.one_hot(self.input_cols, depth=50)

            column_memory = tf.matmul(column_one_hot, column_memory)
            row_memory = tf.matmul(row_one_hot, row_memory)

            ### NumNet Codes
            """
            row_one_hot_ = tf.expand_dims(row_one_hot, axis=3)
            col_one_hot_ = tf.expand_dims(column_one_hot, axis=2)

            col_row_onehot = tf.einsum(col_one_hot_, row_one_hot_, equation='bsca,bsar->bscr')
            col_row_onehot = tf.reshape(col_row_onehot, shape=[-1, 500, 50 * 15])
            col_row_onehot_ = tf.transpose(col_row_onehot, perm=[0, 2, 1])

            col_row_memory = tf.matmul(col_row_onehot_, sequence_output)
            col_row_memory = tf.reshape(col_row_memory, shape=[-1, 50, 768])

            adjacency_mat_bigger = tf.reshape(-1, 50, 50)
            adjacency_mat_smaller = tf.reshape(-1, 50, 50)

            #[b, c, r, r] X [b, c, r, h]
            #how many hops to propagate
            with tf.variable_scope("bigger_graph"):
                num_states_big = modeling.attention_layer(from_tensor=col_row_memory,
                                         to_tensor=col_row_memory,
                                         attention_mask=adjacency_mat_bigger)
                num_states_big = tf.reshape(num_states_big, shape=[-1, 15 * 50, 768])
                num_states_big = tf.matmul(col_row_onehot, num_states_big)

            with tf.variable_scope("smaller_graph"):
                num_states_small = modeling.attention_layer(from_tensor=col_row_memory,
                                         to_tensor=col_row_memory,
                                         attention_mask=adjacency_mat_smaller)
                num_states_small = tf.reshape(num_states_small, shape=[-1, 15 * 50, 768])
                num_states_small = tf.matmul(col_row_onehot, num_states_small)

            num_states = tf.concat([num_states_big, num_states_small], axis=2)
            num_states = Fully_Connected(num_states, output=768, name='num_hidden', activation=tf.nn.leaky_relu)
            num_states = tf.nn.dropout(num_states, keep_prob=1 - dropout)
            """
            ###

            sequence_output = tf.concat([column_memory, row_memory, sequence_output],
                                        axis=2)
            probs_start, probs_stop = self.get_qa_probs(sequence_output, is_training=True)

            loss, loss2, loss3 = self.get_qa_loss(probs_start, probs_stop)
            total_loss = loss

            learning_rate = 2e-5

            optimizer = optimization.create_optimizer(loss=total_loss, init_lr=learning_rate, num_train_steps=25000,
                                                      num_warmup_steps=500, use_tpu=False)
            sess.run(tf.initialize_all_variables())

            if self.first_training is True:
                saver = tf.train.Saver(bert_variables)
                saver.restore(sess, self.bert_path)
                print('BERT restored')

            if is_Continue is True:
                saver = tf.train.Saver()
                saver.restore(sess, self.save_path)

            for i in range(training_epoch):
                ##
                # 라벨 정보 바뀌어야함
                ##
                input_ids, input_mask, input_segments, input_rows, input_cols, \
                input_numeric_sapce, input_numeric_mask, \
                row_mask, col_mask,\
                start_label, stop_label = \
                    self.processor.next_batch()

                #input_ids, input_mask, input_segments, input_positions, input_rows, \
                #input_cols, start_label, stop_label, input_names, input_rankings = \
                #    self.processor.next_batch2()

                feed_dict = {self.input_ids: input_ids, self.input_mask: input_mask,
                             self.input_segments: input_segments,
                             self.input_cols: input_cols, self.input_rows: input_rows,
                             self.input_mask_rows: row_mask, self.input_mask_cols: col_mask,
                             self.numeric_space: np.zeros(shape=[3, 10, 512], dtype=np.int32),
                             self.numeric_mask: np.zeros(shape=[3, 10, 512], dtype=np.float32),
                             self.start_label: start_label, self.stop_label: stop_label
                             }

                loss_, _ = sess.run([total_loss, optimizer], feed_dict=feed_dict)
                print(i, loss_)
                if i % 1000 == 0 and i > 100:
                    print('saved!')
                    saver = tf.train.Saver()
                    saver.save(sess, self.save_path)

    def eval_with_span(self):
        self.keep_prob = 1.0

        name_tagger = Name_Tagging.Name_tagger()
        chuncker = Chuncker.Chuncker()

        path_dir = 'F:\\korquad2_dev'

        file_list = os.listdir(path_dir)
        file_list.sort()

        vocab = tokenization.load_vocab(vocab_file='vocab.txt')
        tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        f_tokenizer = tokenization.FullTokenizer(vocab_file='vocab.txt')

        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        tokenizer.add_tokens('[table]')
        tokenizer.add_tokens('[/table]')
        tokenizer.add_tokens('[list]')
        tokenizer.add_tokens('[/list]')
        tokenizer.add_tokens('[h3]')
        tokenizer.add_tokens('[td]')

        em_total = 0
        f1_total = 0
        epo = 0

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids,
            self.input_mask,
            self.input_segments,
            is_training=False)
            """
            initializer_range = 0.02
            space_embedding = tf.get_variable(
                name='space_embedding',
                shape=[150, 768],
                initializer=create_initializer(initializer_range))

            space_ids = tf.reshape(self.numeric_space, shape=[-1, 500])
            space_mask = tf.expand_dims(self.numeric_mask, axis=-1)

            space_representation = tf.nn.embedding_lookup(space_embedding, space_ids)
            space_representation = tf.reshape(space_representation, shape=[-1, 10, 500, 768])

            space_representation = space_representation * tf.cast(space_mask, dtype=tf.float32)
            space_representation = tf.reduce_sum(space_representation, axis=1)
            """
            input_shape = get_shape_list(sequence_output, expected_rank=3)
            batch_size = input_shape[0]
            seq_length = input_shape[1]

            column_memory, row_memory = self.Table_Memory_Network(sequence_output=sequence_output,
                                                                  hops=5,
                                                                  dropout=0.0)
            # bert_variables = tf.global_variables()

            row_one_hot = tf.one_hot(self.input_rows, depth=100)
            column_one_hot = tf.one_hot(self.input_cols, depth=50)

            column_memory = tf.matmul(column_one_hot, column_memory)
            row_memory = tf.matmul(row_one_hot, row_memory)

            ### NumNet Codes
            """
            row_one_hot_ = tf.expand_dims(row_one_hot, axis=3)
            col_one_hot_ = tf.expand_dims(column_one_hot, axis=2)

            col_row_onehot = tf.einsum(col_one_hot_, row_one_hot_, equation='bsca,bsar->bscr')
            col_row_onehot = tf.reshape(col_row_onehot, shape=[-1, 500, 50 * 15])
            col_row_onehot_ = tf.transpose(col_row_onehot, perm=[0, 2, 1])

            col_row_memory = tf.matmul(col_row_onehot_, sequence_output)
            col_row_memory = tf.reshape(col_row_memory, shape=[-1, 50, 768])

            adjacency_mat_bigger = tf.reshape(-1, 50, 50)
            adjacency_mat_smaller = tf.reshape(-1, 50, 50)

            #[b, c, r, r] X [b, c, r, h]
            #how many hops to propagate
            with tf.variable_scope("bigger_graph"):
                num_states_big = modeling.attention_layer(from_tensor=col_row_memory,
                                         to_tensor=col_row_memory,
                                         attention_mask=adjacency_mat_bigger)
                num_states_big = tf.reshape(num_states_big, shape=[-1, 15 * 50, 768])
                num_states_big = tf.matmul(col_row_onehot, num_states_big)

            with tf.variable_scope("smaller_graph"):
                num_states_small = modeling.attention_layer(from_tensor=col_row_memory,
                                         to_tensor=col_row_memory,
                                         attention_mask=adjacency_mat_smaller)
                num_states_small = tf.reshape(num_states_small, shape=[-1, 15 * 50, 768])
                num_states_small = tf.matmul(col_row_onehot, num_states_small)

            num_states = tf.concat([num_states_big, num_states_small], axis=2)
            num_states = Fully_Connected(num_states, output=768, name='num_hidden', activation=tf.nn.leaky_relu)
            num_states = tf.nn.dropout(num_states, keep_prob=1 - dropout)
            """
            ###

            sequence_output = tf.concat([column_memory, row_memory, sequence_output],
                                        axis=2)
            prob_start, prob_stop = self.get_qa_probs(sequence_output, is_training=False)

            prob_start = tf.nn.softmax(prob_start, axis=-1)
            prob_stop = tf.nn.softmax(prob_stop, axis=-1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            for file_name in file_list:
                print(file_name, 'processing evaluation')

                in_path = path_dir + '\\' + file_name
                data = json.load(open(in_path, 'r', encoding='utf-8'))

                for article in data['data']:
                    doc = str(article['context'])

                    for qas in article['qas']:
                        error_code = -1

                        answer = qas['answer']
                        answer_start = answer['answer_start']
                        answer_text = answer['text']
                        question = qas['question']

                        answer_text_ = doc[answer_start: answer_start + len(answer_text)]

                        chuncker.get_feautre(query=question)

                        if len(answer_text) > 40:
                            continue

                        query_tokens = []
                        query_tokens.append('[CLS]')
                        q_tokens = tokenizer.tokenize(question.lower())
                        for tk in q_tokens:
                            query_tokens.append(tk)
                        query_tokens.append('[SEP]')

                        ######
                        # 정답에 ans 토큰을 임베딩하기 위한 코드
                        ######

                        ans1 = ''
                        ans2 = ''
                        if doc[answer_start - 1] == ' ':
                            ans1 = ' [answer] '
                        else:
                            ans1 = ' [answer]'

                        if doc[answer_start + len(answer_text)] == ' ':
                            ans2 = ' [/answer] '
                        else:
                            ans2 = ' [/answer]'

                        doc_ = doc[0: answer_start] + ans1 + answer_text + ans2 + doc[
                                                                                  answer_start + len(answer_text): -1]
                        doc_ = str(doc_)
                        #
                        #####

                        paragraphs = doc_.split('<h2>')
                        sequences = []

                        checked = False

                        for paragraph in paragraphs:
                            try:
                                title = paragraph.split('[/h2]')[0]
                                paragraph = paragraph.split('[/h2]')[1]
                            except:
                                title = ''

                            sub_paragraphs = paragraph.split('<h3>')

                            for sub_paragraph in sub_paragraphs:
                                if checked is True:
                                    break

                                paragraph_, table_list = pre_process_document(paragraph, answer_setting=False,
                                                                              a_token1='',
                                                                              a_token2='')
                                paragraph = process_document(paragraph)

                                for table_text in table_list:
                                    if checked is True:
                                        break

                                    if table_text.find('[answer]') != -1:
                                        table_text = table_text.replace('<th', '<td')
                                        table_text = table_text.replace('</th', '</td')

                                        table_text = table_text.replace(' <td>', '<td>')
                                        table_text = table_text.replace(' <td>', '<td>')
                                        table_text = table_text.replace('\n<td>', '<td>')
                                        table_text = table_text.replace('</td> ', '</td>')
                                        table_text = table_text.replace('</td> ', '</td>')
                                        table_text = table_text.replace('\n<td>', '<td>')
                                        table_text = table_text.replace('[answer]<td>', '<td>[answer] ')
                                        table_text = table_text.replace('</td>[/answer]', ' [/answer]</td>')
                                        table_text = table_text.replace('</td>', '  </td>')
                                        table_text = table_text.replace('<td>', '<td> ')
                                        table_text = table_text.replace('[answer]', '')
                                        table_text = table_text.replace('[/answer]', '')

                                        table_text, child_texts = overlap_table_process(table_text=table_text)
                                        table_text = head_process(table_text=table_text)

                                        table_holder.get_table_text(table_text=table_text)
                                        table_data = table_holder.table_data
                                        lengths = []

                                        for data in table_data:
                                            lengths.append(len(data))
                                        if len(lengths) <= 0:
                                            break

                                        length = max(lengths)

                                        rank_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)
                                        col_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)
                                        row_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)

                                        count_arr = np.zeros(shape=[200], dtype=np.int32)
                                        for data in table_data:
                                            count_arr[len(data)] += 1
                                        table_head = get_table_head(table_text=table_text, count_arr=count_arr)

                                        rankings = Ranking_ids.numberToRanking(table_data, table_head)

                                        for j in range(length):
                                            for i in range(len(table_data)):
                                                col_ids[i, j] = j
                                                row_ids[i, j] = i
                                                rank_ids[i, j] = rankings[i][j]
                                            """
                                            check = True
                                            values = np.zeros(shape=[len(table_data)], dtype=np.float)

                                            for i in range(len(table_data)):
                                                try:
                                                    if RepresentsInt(table_data[i][j]) is False:
                                                        #print('check:', table_data[i][j])
                                                        check = False
                                                        break
                                                    values[i] = float(table_data[i][j])
                                                except:
                                                    values[i] = -999

                                            if check is True:
                                                arg_idx = values.argsort()

                                                for i in range(len(table_data)):
                                                    rank_ids[i, j] = arg_idx[i]
                                                    col_ids[i, j] = j
                                                    row_ids[i, j] = i
                                            else:
                                                for i in range(len(table_data)):
                                                    col_ids[i, j] = j
                                                    row_ids[i, j] = i
                                            """
                                        idx = 0
                                        tokens_ = []
                                        rows_ = []
                                        cols_ = []
                                        ranks_ = []
                                        spaces_ = []
                                        #name_tags_ = []

                                        for j in range(len(table_head)):
                                            if table_head[j] is not None:
                                                tokens = tokenizer.tokenize(table_head[j])
                                                #name_tag = name_tagger.get_name_tag(table_head[j])

                                                for k, tk in enumerate(tokens):
                                                    tokens_.append(tk)
                                                    rows_.append(0)
                                                    cols_.append(j)
                                                    ranks_.append(0)
                                                    #name_tags_.append(name_tag)

                                                    if k >= 50:
                                                        break

                                        for i in range(len(table_data)):
                                            for j in range(len(table_data[i])):
                                                if table_data[i][j] is not None:
                                                    tokens = tokenizer.tokenize(table_data[i][j])
                                                    #name_tag = name_tagger.get_name_tag(table_data[i][j])

                                                    is_num, number_value = detect_num_word(table_data[i][j])

                                                    for k, tk in enumerate(tokens):
                                                        tokens_.append(tk)
                                                        rows_.append(i + 1)
                                                        cols_.append(j)
                                                        ranks_.append(rank_ids[i][j])

                                                        if is_num is True:
                                                            if detect_simple_num_word(tk) is True:
                                                                space_lists = get_space_num_lists(number_value)
                                                                spaces_.append(space_lists)
                                                            else:
                                                                spaces_.append(-1)
                                                        else:
                                                            spaces_.append(-1)
                                                        #name_tags_.append(name_tag)

                                                        if k >= 50:
                                                            break

                                                    if len(tokens) > 50 and str(table_data[i][j]).find(
                                                            '[/answer]') != -1:
                                                        tokens_.append('[/answer]')
                                                        rows_.append(i)
                                                        cols_.append(j)
                                                        ranks_.append(rank_ids[i][j])
                                                        #name_tags_.append(name_tag)

                                        start_idx = -1
                                        end_idx = -1

                                        tokens = []
                                        rows = []
                                        cols = []
                                        ranks = []
                                        segments = []
                                        spaces = []
                                        #name_tags = []

                                        for tk in query_tokens:
                                            tokens.append(tk)
                                            rows.append(0)
                                            cols.append(0)
                                            ranks.append(0)
                                            segments.append(0)
                                            #name_tags.append(0)
                                            spaces.append(-1)

                                        for j, tk in enumerate(tokens_):
                                            if tk == '[answer]':
                                                start_idx = len(tokens)
                                            elif tk == '[/answer]':
                                                end_idx = len(tokens) - 1
                                            else:
                                                tokens.append(tk)
                                                rows.append(rows_[j] + 1)
                                                cols.append(cols_[j] + 1)
                                                ranks.append(ranks_[j])
                                                segments.append(1)
                                                spaces.append(spaces_[j])
                                                #name_tags.append(name_tags_[j])

                                        ids = tokenization.convert_tokens_to_ids(vocab=vocab, tokens=tokens)

                                        max_length = 512

                                        length = len(ids)
                                        if length > max_length:
                                            length = max_length

                                        input_ids = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        input_mask = np.zeros(shape=[1, max_length], dtype=np.int32)

                                        segments_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        ranks_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        cols_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        rows_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        names_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        numeric_space = np.zeros(shape=[1, 10, max_length], dtype=np.int32)
                                        numeric_mask = np.zeros(shape=[1, 10, max_length], dtype=np.int32)

                                        count = 0

                                        for j in range(length):
                                            input_ids[count, j] = ids[j]
                                            segments_has_ans[count, j] = segments[j]
                                            cols_has_ans[count, j] = cols[j]
                                            rows_has_ans[count, j] = rows[j]
                                            ranks_has_ans[count, j] = ranks[j]
                                            input_mask[count, j] = 1

                                            if spaces[j] != -1:
                                                for k in range(10):
                                                    numeric_space[count, k, j] = spaces[j][k]
                                                    if spaces[j][k] != 0:
                                                        numeric_mask[count, k, j] = 1
                                            #names_has_ans[count, j] = name_tags[j]

                                        input_mask_rows = np.zeros(shape=[1, 50], dtype=np.int32)
                                        input_mask_cols = np.zeros(shape=[1, 30], dtype=np.int32)

                                        max_rows = rows_has_ans.max(axis=1)
                                        max_cols = cols_has_ans.max(axis=1)

                                        try:
                                            for i in range(1):
                                                if max_rows[i] > 50:
                                                    max_rows[i] = 50
                                                if max_cols[i] > 30:
                                                    max_cols = 30

                                                for j in range(max_rows[i] + 1):
                                                    input_mask_rows[i, j] = 1
                                                for j in range(max_cols[i] + 1):
                                                    input_mask_cols[i, j] = 1
                                        except:
                                            None

                                        feed_dict = {self.input_ids: input_ids, self.input_mask: input_mask,
                                                     self.input_segments: segments_has_ans,
                                                     self.input_names: names_has_ans,
                                                     self.input_rankings: ranks_has_ans,
                                                     self.input_mask_rows: input_mask_rows,
                                                     self.input_mask_cols: input_mask_cols,
                                                     self.input_rows: rows_has_ans, self.input_cols: cols_has_ans,
                                                     self.numeric_space: numeric_space,
                                                     self.numeric_mask: numeric_mask,}

                                        if input_ids.shape[0] > 0:

                                            probs_start, probs_stop = \
                                                sess.run([prob_start, prob_stop], feed_dict=feed_dict)

                                            probs_start = np.array(probs_start, dtype=np.float32)
                                            probs_stop = np.array(probs_stop, dtype=np.float32)

                                            for j in range(input_ids.shape[0]):
                                                for k in range(1, input_ids.shape[1]):
                                                    probs_start[j, k] = 0
                                                    probs_stop[j, k] = 0

                                                    if input_ids[j, k] == 3:
                                                        break

                                            self.chuncker.get_feautre(question)

                                            prob_scores = []
                                            c_scores = []

                                            for j in range(input_ids.shape[0]):
                                                # paragraph ranking을 위한 score 산정기준
                                                # score2 = ev_values[j, 0]
                                                score2 = 2 - (probs_start[j, 0] + probs_stop[j, 0])

                                                prob_scores.append(score2)
                                                #c_scores.append(self.chuncker.get_chunk_score(sequences[j]))

                                            if True:
                                                for j in range(input_ids.shape[0]):
                                                    probs_start[j, 0] = -999
                                                    probs_stop[j, 0] = -999

                                                # CLS 선택 무효화

                                                prediction_start = probs_start.argmax(axis=1)
                                                prediction_stop = probs_stop.argmax(axis=1)

                                                answers = []
                                                scores = []
                                                candi_scores = []

                                                for j in range(input_ids.shape[0]):
                                                    answer_start_idx = prediction_start[j]
                                                    answer_stop_idx = prediction_stop[j]

                                                    if cols_has_ans[0, answer_start_idx] != cols_has_ans[0, answer_stop_idx]:
                                                        answer_stop_idx2 = answer_stop_idx
                                                        answer_stop_idx = answer_start_idx
                                                        answer_start_idx2 = answer_stop_idx2

                                                        for k in range(answer_start_idx + 1, input_ids.shape[1]):
                                                            if cols_has_ans[0, k] == cols_has_ans[0, answer_start_idx]:
                                                                answer_stop_idx = k
                                                            else:
                                                                break

                                                        for k in reversed(list(range(0, answer_stop_idx2 - 1))):
                                                            if cols_has_ans[0, k] == cols_has_ans[0, answer_stop_idx2]:
                                                                answer_start_idx2 = k
                                                            else:
                                                                break

                                                        prob_1 = probs_start[0, answer_start_idx] + \
                                                                 probs_stop[0, answer_stop_idx]

                                                        prob_2 = probs_start[0, answer_start_idx2] + \
                                                                 probs_stop[0, answer_stop_idx2]

                                                        if prob_2 > prob_1:
                                                            answer_start_idx = answer_start_idx2
                                                            answer_stop_idx = answer_stop_idx2

                                                    score = probs_start[j, answer_start_idx]
                                                    scores.append(score * 1)
                                                    candi_scores.append(score * 1)

                                                    if answer_start_idx > answer_stop_idx:
                                                        answer_stop_idx = answer_start_idx + 15
                                                    if f_tokenizer.inv_vocab[input_ids[j, answer_start_idx]] == '[p]':
                                                        for k in range(answer_start_idx, input_ids.shape[1]):
                                                            if f_tokenizer.inv_vocab[input_ids[j, k]] == '[/p]':
                                                                answer_stop_idx = k
                                                                break

                                                    answer = ''

                                                    if answer_stop_idx + 1 >= input_ids.shape[1]:
                                                        answer_stop_idx = input_ids.shape[1] - 2

                                                    for k in range(answer_start_idx, answer_stop_idx + 1):
                                                        tok = f_tokenizer.inv_vocab[input_ids[j, k]]
                                                        if len(tok) > 0:
                                                            if tok[0] != '#':
                                                                answer += ' '
                                                        answer += str(f_tokenizer.inv_vocab[input_ids[j, k]]).replace(
                                                            '##', '')

                                                    answers.append(answer)

                                            if len(answers) > 0:
                                                answer_candidates = []
                                                candidates_scores = []

                                                for _ in range(1):
                                                    m_s = -99
                                                    m_ix = 0

                                                    for q in range(len(scores)):
                                                        if m_s < scores[q]:
                                                            m_s = scores[q]
                                                            m_ix = q

                                                    answer_candidates.append(answer_re_touch(answers[m_ix]))
                                                    candidates_scores.append(candi_scores[m_ix])
                                                    # print('score:', scores[m_ix])
                                                    # scores[m_ix] = -999

                                                a1 = [0]
                                                a2 = [0]

                                                for a_c in answer_candidates:
                                                    if a_c.find('<table>') != -1:
                                                        continue

                                                    a1.append(
                                                        exact_match_score(prediction=a_c, ground_truth=answer_text))
                                                    a2.append(f1_score(prediction=a_c, ground_truth=answer_text))

                                                em_total += max(a1)
                                                f1_total += max(a2)
                                                epo += 1

                                                for j in range(input_ids.shape[0]):
                                                    check = 'None'
                                                    answer_text = answer_text.replace('<a>', '').replace('</a>', '')

                                                    f1_ = f1_score(prediction=answer_re_touch(answers[j]),
                                                                   ground_truth=answer_text)

                                                    text = answers[j]

                                                if f1_ > 0.5:
                                                    continue

                                                print('score:', scores[j], check, type, 'F1:',
                                                      f1_, ' , ',
                                                      text.replace('\n', ' '))
                                                print(table_text.replace('\n', ''))
                                                print('question:', question)
                                                print('answer:', answer_text, ',', answer_text_)
                                                print('EM:', em_total / epo)
                                                print('F1:', f1_total / epo)
                                                print('-----\n', epo)

    def eval_with_span_testset(self):
        dataholder = DataHolder_test.DataHolder()

        name_tagger = Name_Tagging.Name_tagger()
        chuncker = Chuncker.Chuncker()

        vocab = tokenization.load_vocab(vocab_file='vocab.txt')
        tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        f_tokenizer = tokenization.FullTokenizer(vocab_file='vocab.txt')

        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        tokenizer.add_tokens('[table]')
        tokenizer.add_tokens('[/table]')
        tokenizer.add_tokens('[list]')
        tokenizer.add_tokens('[/list]')
        tokenizer.add_tokens('[h3]')
        tokenizer.add_tokens('[td]')

        em_total = 0
        f1_total = 0

        em_total1 = 0
        f1_total1 = 0

        em_total2 = 0
        f1_total2 = 0

        em_total3 = 0
        f1_total3 = 0

        em_total4 = 0
        f1_total4 = 0

        em_total5 = 0
        f1_total5 = 0

        em_total6 = 0
        f1_total6 = 0

        epo = 0
        epo1 = 0.1
        epo2 = 0.1
        epo3 = 0.1
        epo4 = 0.1
        epo5 = 0.1
        epo6 = 0.1

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output, row_memory, column_memory = self.create_model(self.input_ids, self.input_mask,
                                                                       self.input_segments, is_training=False)

            initializer_range = 0.02
            space_embedding = tf.get_variable(
                name='space_embedding',
                shape=[150, 768],
                initializer=create_initializer(initializer_range))

            space_ids = tf.reshape(self.numeric_space, shape=[-1, 512])
            space_mask = tf.expand_dims(self.numeric_mask, axis=-1)

            space_representation = tf.nn.embedding_lookup(space_embedding, space_ids)
            space_representation = tf.reshape(space_representation, shape=[-1, 10, 512, 768])

            space_representation = space_representation * tf.cast(space_mask, dtype=tf.float32)
            space_representation = tf.reduce_sum(space_representation, axis=1)

            input_shape = get_shape_list(sequence_output, expected_rank=3)
            batch_size = input_shape[0]
            seq_length = input_shape[1]

            column_memory, row_memory = self.Table_Memory_Network(sequence_output=sequence_output,
                                                                  space_states=space_representation,
                                                                  row_wise_memory=row_memory,
                                                                  column_wise_memory=column_memory,
                                                                  hops=5, dropout=0.0)

            row_one_hot = tf.one_hot(self.input_rows, depth=50)
            column_one_hot = tf.one_hot(self.input_cols, depth=30)

            column_memory = tf.matmul(column_one_hot, column_memory)
            row_memory = tf.matmul(row_one_hot, row_memory)

            sequence_output = tf.concat([column_memory, row_memory, sequence_output], axis=2)
            prob_start, prob_stop = self.get_qa_probs(sequence_output, is_training=False)

            prob_start = tf.nn.softmax(prob_start, axis=-1)
            prob_stop = tf.nn.softmax(prob_stop, axis=-1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            for b in range(dataholder.input_ids.shape[0]):
                input_ids, input_mask, segments_has_ans, rows_has_ans, cols_has_ans, \
                    answer_text, question_text, answer_type = \
                    dataholder.next_batch()

                input_mask_rows = np.zeros(shape=[input_ids.shape[0], 50], dtype=np.int32)
                input_mask_cols = np.zeros(shape=[input_ids.shape[0], 30], dtype=np.int32)

                max_rows = rows_has_ans.max(axis=1)
                max_cols = cols_has_ans.max(axis=1)

                for i in range(input_ids.shape[0]):
                    for j in range(max_rows[i] + 1):
                        input_mask_rows[i, j] = 1
                    for j in range(max_cols[i] + 1):
                        input_mask_cols[i, j] = 1

                numeric_space = np.zeros(shape=[1, 10, 512], dtype=np.int32)
                numeric_mask = np.zeros(shape=[1, 10, 512], dtype=np.int32)

                answer_text = str(answer_text).replace('?', ' ')

                feed_dict = {self.input_ids: input_ids, self.input_mask: input_mask,
                             self.input_segments: segments_has_ans,
                             self.input_rows: rows_has_ans, self.input_cols: cols_has_ans,
                             self.numeric_space: numeric_space,
                             self.numeric_mask: numeric_mask,
                             self.input_mask_cols: input_mask_cols,
                             self.input_mask_rows: input_mask_rows
                             }

                if input_ids.shape[0] > 0:

                    probs_start, probs_stop = \
                        sess.run([prob_start, prob_stop], feed_dict=feed_dict)

                    probs_start = np.array(probs_start, dtype=np.float32)
                    probs_stop = np.array(probs_stop, dtype=np.float32)

                    for j in range(input_ids.shape[0]):
                        for k in range(1, input_ids.shape[1]):
                            probs_start[j, k] = 0
                            probs_stop[j, k] = 0

                            if input_ids[j, k] == 3:
                                break

                    prob_scores = []
                    c_scores = []

                    for j in range(input_ids.shape[0]):
                        # paragraph ranking을 위한 score 산정기준
                        # score2 = ev_values[j, 0]
                        score2 = 2 - (probs_start[j, 0] + probs_stop[j, 0])

                        prob_scores.append(score2)
                        #c_scores.append(self.chuncker.get_chunk_score(sequences[j]))

                    if True:
                        for j in range(input_ids.shape[0]):
                            probs_start[j, 0] = -999
                            probs_stop[j, 0] = -999

                        # CLS 선택 무효화

                        prediction_start = probs_start.argmax(axis=1)
                        prediction_stop = probs_stop.argmax(axis=1)

                        answers = []
                        scores = []
                        candi_scores = []

                        for j in range(input_ids.shape[0]):
                            answer_start_idx = prediction_start[j]
                            answer_stop_idx = prediction_stop[j]

                            if cols_has_ans[0, answer_start_idx] != cols_has_ans[0, answer_stop_idx]:
                                answer_stop_idx2 = answer_stop_idx
                                answer_stop_idx = answer_start_idx
                                answer_start_idx2 = answer_stop_idx2

                                for k in range(answer_start_idx + 1, input_ids.shape[1]):
                                    if cols_has_ans[0, k] == cols_has_ans[0, answer_start_idx]:
                                        answer_stop_idx = k
                                    else:
                                        break

                                for k in reversed(list(range(0, answer_stop_idx2 - 1))):
                                    if cols_has_ans[0, k] == cols_has_ans[0, answer_stop_idx2]:
                                        answer_start_idx2 = k
                                    else:
                                        break

                                prob_1 = probs_start[0, answer_start_idx] + \
                                         probs_stop[0, answer_stop_idx]

                                prob_2 = probs_start[0, answer_start_idx2] + \
                                         probs_stop[0, answer_stop_idx2]

                                if prob_2 > prob_1:
                                    answer_start_idx = answer_start_idx2
                                    answer_stop_idx = answer_stop_idx2

                            score = probs_start[j, answer_start_idx]
                            scores.append(score * 1)
                            candi_scores.append(score * 1)

                            if answer_start_idx > answer_stop_idx:
                                answer_stop_idx = answer_start_idx + 15
                            if f_tokenizer.inv_vocab[input_ids[j, answer_start_idx]] == '[p]':
                                for k in range(answer_start_idx, input_ids.shape[1]):
                                    if f_tokenizer.inv_vocab[input_ids[j, k]] == '[/p]':
                                        answer_stop_idx = k
                                        break

                            answer = ''

                            if answer_stop_idx + 1 >= input_ids.shape[1]:
                                answer_stop_idx = input_ids.shape[1] - 2

                            for k in range(answer_start_idx, answer_stop_idx + 1):
                                tok = f_tokenizer.inv_vocab[input_ids[j, k]]
                                if len(tok) > 0:
                                    if tok[0] != '#':
                                        answer += ' '
                                answer += str(f_tokenizer.inv_vocab[input_ids[j, k]]).replace(
                                    '##', '')

                            answers.append(answer)

                    if len(answers) > 0:
                        answer_candidates = []
                        candidates_scores = []

                        for _ in range(1):
                            m_s = -99
                            m_ix = 0

                            for q in range(len(scores)):
                                if m_s < scores[q]:
                                    m_s = scores[q]
                                    m_ix = q

                            answer_candidates.append(answer_re_touch(answers[m_ix]))
                            candidates_scores.append(candi_scores[m_ix])
                            # print('score:', scores[m_ix])
                            # scores[m_ix] = -999

                        a1 = [0]
                        a2 = [0]

                        for a_c in answer_candidates:
                            if a_c.find('<table>') != -1:
                                continue

                            a1.append(
                                exact_match_score(prediction=a_c, ground_truth=answer_text))
                            a2.append(f1_score(prediction=a_c, ground_truth=answer_text))

                        em_total += max(a1)
                        f1_total += max(a2)
                        epo += 1

                        if answer_type == "유형1":
                            em_total1 += max(a1)
                            f1_total1 += max(a2)
                            epo1 += 1
                        elif answer_type == "유형2":
                            em_total2 += max(a1)
                            f1_total2 += max(a2)
                            epo2 += 1
                        elif answer_type == "유형3":
                            em_total3 += max(a1)
                            f1_total3 += max(a2)
                            epo3 += 1
                        elif answer_type == "유형4":
                            em_total4 += max(a1)
                            f1_total4 += max(a2)
                            epo4 += 1
                        elif answer_type == "유형 1":
                            em_total1 += max(a1)
                            f1_total1 += max(a2)
                            epo1 += 1
                        elif answer_type == "유형 2":
                            em_total2 += max(a1)
                            f1_total2 += max(a2)
                            epo2 += 1
                        elif answer_type == "유형 3":
                            em_total3 += max(a1)
                            f1_total3 += max(a2)
                            epo3 += 1
                        elif answer_type == "유형4":
                            em_total4 += max(a1)
                            f1_total4 += max(a2)
                            epo4 += 1
                        elif answer_type == "유형5":
                            em_total5 += max(a1)
                            f1_total5 += max(a2)
                            epo5 += 1
                        elif answer_type == "유형6":
                            em_total6 += max(a1)
                            f1_total6 += max(a2)
                            epo6 += 1

                        for j in range(input_ids.shape[0]):
                            check = 'None'
                            answer_text = answer_text.replace('<a>', '').replace('</a>', '')

                            f1_ = f1_score(prediction=answer_re_touch(answers[j]),
                                           ground_truth=answer_text)

                            text = answers[j]

                        #if f1_ > 0.5:
                        #    continue

                        seq_text = ''
                        for i in range(input_ids.shape[1]):
                            seq_text += f_tokenizer.inv_vocab[input_ids[0, i]] + ' '
                        seq_text = seq_text.replace(' ##', '')

                        if epo > 900:
                            print(seq_text)

                        print(answer_start_idx, answer_stop_idx)
                        print(question_text, answer_type)
                        print('ground truth:', answer_text)
                        print('prediction:', a_c)

                        print('F1:', f1_total / epo, 'EM:', em_total / epo, epo)
                        print('유형1 F1:', f1_total1 / epo1, 'EM:', em_total1 / epo1, epo1)
                        print('유형2 F1:', f1_total2 / epo2, 'EM:', em_total2 / epo2, epo2)
                        print('유형3 F1:', f1_total3 / epo3, 'EM:', em_total3 / epo3, epo3)
                        print('유형4 F1:', f1_total4 / epo4, 'EM:', em_total4 / epo4, epo4)
                        print('유형5 F1:', f1_total5 / epo5, 'EM:', em_total5 / epo5, epo5)
                        print('유형6 F1:', f1_total6 / epo6, 'EM:', em_total6 / epo6, epo6)

                        print()
                        print('-----\n', epo)
