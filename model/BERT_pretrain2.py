import modeling as modeling
import tensorflow as tf

import DataHolder as DataHolder
import Dataholder_test

from utils import Fully_Connected
import numpy as np

import optimization

import tokenization
import LM_Utils


def kl(x, y):
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y, allow_nan_stats=False)


def kl_coef(i):
    # coef for KL annealing
    # reaches 1 at i = 22000
    # https://github.com/kefirski/pytorch_RVAE/blob/master/utils/functional.py
    return (tf.tanh((i - 3500) / 1000) + 1) / 2


def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


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


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def masked_softmax(logits, mask, dim):
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30)  # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask)  # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


class KoNET:
    def __init__(self, firstTraining, testCase=False):
        self.first_training = firstTraining

        self.save_path = 'E:\\bert_adv2\\bert_model.ckpt'
        self.bert_path = 'E:\\roberta_base\\roberta_base.ckpt'

        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name='input')
        self.input_segments = tf.placeholder(shape=[None, None], dtype=tf.int32, name='segment')
        self.input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name='mask')
        self.input_cols = tf.placeholder(shape=[None, None], dtype=tf.int32, name='cols')
        self.input_rows = tf.placeholder(shape=[None, None], dtype=tf.int32, name='rows')
        self.input_ranks = tf.placeholder(shape=[None, None], dtype=tf.int32, name='ranks')

        self.input_mask_rows = tf.placeholder(shape=[None, 50], dtype=tf.int32, name='mask_rows')
        self.input_mask_cols = tf.placeholder(shape=[None, 30], dtype=tf.int32, name='mask_cols')

        self.label_position = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.label_weight = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.label_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.next_sentence_label = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.domain_label = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.domain_label_seq = tf.placeholder(shape=[None, None, None], dtype=tf.float32)
        self.domain_weights = tf.placeholder(shape=[None, None], dtype=tf.float32)

        self.processor = DataHolder.DataHolder()
        self.keep_prob = 0.8
        if testCase is True:
            self.keep_prob = 1.0

        self.testCase = testCase

        self.vocab = tokenization.load_vocab("vocab.txt")
        self.tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)
        self.f_tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt")

    def get_masked_lm_output_original(self, bert_config, input_tensor, output_weights, positions,
                             label_ids, label_weights):
        """Get loss and log probs for the masked LM."""
        input_tensor = gather_indexes(input_tensor, positions)

        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            label_ids = tf.reshape(label_ids, [-1])
            label_weights = tf.reshape(label_weights, [-1])

            one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator

        return loss, per_example_loss, log_probs

    def get_masked_lm_output(self, bert_config, input_tensor, output_weights, positions,
                             label_ids, label_weights, name_scope=''):
        """Get loss and log probs for the masked LM."""
        input_tensor = gather_indexes(input_tensor, positions)

        #with tf.variable_scope("cls/predictions"):
        with tf.variable_scope("output_layer"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            label_ids = tf.reshape(label_ids, [-1])
            label_weights = tf.reshape(label_weights, [-1])

            one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator

        return loss, per_example_loss, log_probs

    def get_next_sentence_output(self, bert_config, input_tensor, labels):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.

        #with tf.variable_scope("output"):
        with tf.variable_scope("cls/predictions"):
            output_weights = tf.get_variable(
                "output_weights",
                shape=[2, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range))
            output_bias = tf.get_variable(
                "output_bias", shape=[2], initializer=tf.zeros_initializer())

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            #labels = tf.reshape(labels, [-1])
            #one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return loss, per_example_loss, log_probs

    def get_adv_output_(self, bert_config, input_tensor, positions, domain_label,
                             label_weights, dis_lambda, global_step):
        """Get loss and log probs for the masked LM."""
        input_tensor = gather_indexes(input_tensor, positions)

        with tf.variable_scope("cls/domain_classification"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            dis_lambda = dis_lambda * kl_coef(global_step)
            output_weights = tf.get_variable(
                "output_weights",
                shape=[2, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range))
            output_bias = tf.get_variable(
                "output_bias", shape=[2], initializer=tf.zeros_initializer())

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.softmax(logits, axis=-1)

            label_weights = tf.reshape(label_weights, [-1])
            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            domain_weights = tf.reshape(self.domain_weights, shape=[-1])
            domain_label = tf.ones_like(domain_label, dtype=tf.float32)
            domain_label = tf.reshape(domain_label, shape=[-1, 2])

            per_example_loss = kl(log_probs, tf.ones_like(log_probs, dtype=tf.float32))
            #print(per_example_loss)
            #input()
            loss = per_example_loss * domain_weights * label_weights * tf.cast(dis_lambda, dtype=tf.float32)
            loss = tf.reduce_mean(loss)

        return loss, per_example_loss, log_probs

    def get_get_discrimination_output_(self, bert_config, input_tensor, positions, domain_label,
                             label_weights, dis_lambda, global_step):
        """Get loss and log probs for the masked LM."""
        input_tensor = gather_indexes(input_tensor, positions)

        with tf.variable_scope("cls/domain_classification", reuse=True):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            dis_lambda = dis_lambda * kl_coef(global_step)
            output_weights = tf.get_variable(
                "output_weights",
                shape=[2, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range))
            output_bias = tf.get_variable(
                "output_bias", shape=[2], initializer=tf.zeros_initializer())

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            domain_weights = tf.reshape(self.domain_weights, shape=[-1])
            label_weights = tf.reshape(label_weights, shape=[-1])

            domain_label = tf.reshape(domain_label, shape=[-1, 2])

            per_example_loss = -tf.reduce_sum(log_probs * domain_label, axis=[-1])
            loss = tf.reduce_mean(per_example_loss * domain_weights * label_weights)

        return loss, per_example_loss, log_probs

    def create_model(self, input_ids, is_training=True):
        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base_rr.json')

        if self.testCase is True:
            is_training = False

        input_mask = tf.where(input_ids > 0, tf.ones_like(input_ids), tf.zeros_like(input_ids))

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=self.input_segments,
            scope='roberta'
        )

        bert_variables = tf.global_variables()

        return model, bert_variables, model.get_sequence_output()

    def Table_Memory_Network(self, sequence_output, hops=1, dropout=0.2):
        # sequence_output = sequence_output + space_states
        row_one_hot = tf.one_hot(self.input_rows, depth=100)
        row_one_hot = tf.transpose(row_one_hot, perm=[0, 2, 1])

        column_one_hot = tf.one_hot(self.input_cols, depth=50)
        column_one_hot = tf.transpose(column_one_hot, perm=[0, 2, 1])

        column_wise_memory = tf.matmul(column_one_hot, sequence_output)
        row_wise_memory = tf.matmul(row_one_hot, sequence_output)

        reuse = False

        with tf.variable_scope("table_output_layer"):
            with tf.variable_scope("tab_mem"):
                cell_fw_col = tf.nn.rnn_cell.GRUCell(768)
                cell_fw_col = tf.nn.rnn_cell.DropoutWrapper(cell_fw_col, input_keep_prob=self.keep_prob,
                                                            output_keep_prob=self.keep_prob)
                cell_fw_row = tf.nn.rnn_cell.GRUCell(768)
                cell_fw_row = tf.nn.rnn_cell.DropoutWrapper(cell_fw_row, input_keep_prob=self.keep_prob,
                                                            output_keep_prob=self.keep_prob)

                for h in range(hops):
                    print('hop:', h)
                    with tf.variable_scope("column_memory_block", reuse=reuse):
                        column_wise_memory = modeling.attention_layer_modified(
                            from_tensor=column_wise_memory,
                            to_tensor=sequence_output,
                            attention_mask=column_one_hot,
                        )

                    column_wise_memory = Fully_Connected(column_wise_memory, 768, 'hidden_col' + str(0), gelu, reuse=reuse)
                    column_wise_memory = modeling.dropout(column_wise_memory, dropout)

                    with tf.variable_scope("row_memory_block", reuse=reuse):
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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base_rr.json')

        with tf.Session(config=config) as sess:
            text_weights, table_weights = tf.split(self.domain_label, 2, axis=1)

            model, bert_variables, sequence_output = self.create_model(self.input_ids)

            column_memory, row_memory = self.Table_Memory_Network(sequence_output, hops=5)

            row_one_hot = tf.one_hot(self.input_rows, depth=100)
            column_one_hot = tf.one_hot(self.input_cols, depth=50)

            column_memory = tf.matmul(column_one_hot, column_memory)
            row_memory = tf.matmul(row_one_hot, row_memory)

            sequence_output = tf.concat([column_memory, row_memory, sequence_output], axis=2)

            input_tensor = sequence_output
            output_weights = model.get_embedding_table()
            positions = self.label_position
            label_ids = self.label_ids
            label_weights = self.label_weight
            domain_label = self.domain_label_seq

            global_step = tf.Variable(0, name='global_step', trainable=False)
            global_step_ = tf.Variable(0, name='global_step_', trainable=False)
            global_step2 = tf.Variable(0, name='global_step2', trainable=False)

            loss2, _, _ = self.get_masked_lm_output(bert_config, input_tensor, output_weights,
                                                             positions,
                                                             label_ids, label_weights)

            input_tensor = model.get_sequence_output()
            loss3, _, log_P = self.get_adv_output_(bert_config, input_tensor, positions,
                                                   domain_label, label_weights,
                                                  dis_lambda=0.5,
                                                  global_step=global_step)
            loss4, _, _ = self.get_get_discrimination_output_(bert_config, input_tensor, positions,
                                                   domain_label, label_weights,
                                                  dis_lambda=0.5,
                                                  global_step=global_step)

            bert_vars = get_variables_with_name('bert')
            cls_vars = get_variables_with_name('cls/predictions')

            cls_vars.extend(bert_vars)

            output_vars = get_variables_with_name('output_layer')
            output_vars.extend(bert_vars)

            disc_vars = get_variables_with_name('cls/domain_classification')

            lm_loss = loss2 + loss3
            lm_loss2 = loss4

            optimizer_model2 = optimization.create_optimizer(loss=lm_loss, init_lr=5e-5, num_train_steps=125000,
                                                        num_warmup_steps=5000, use_tpu=False, var_list=output_vars,
                                                        global_step=global_step_)

            optimizer_disc = optimization.create_optimizer(loss=lm_loss2, init_lr=5e-5, num_train_steps=125000,
                                                            num_warmup_steps=5000, use_tpu=False, var_list=disc_vars,
                                                           global_step=global_step2)

            sess.run(tf.initialize_all_variables())

            if self.first_training is True:
                saver = tf.train.Saver(cls_vars)
                saver.restore(sess, self.bert_path)
                print('BERT restored')

            if is_Continue is True:
                saver = tf.train.Saver()
                saver.restore(sess, self.save_path)

            for i in range(training_epoch):
                input_ids, input_segments, input_cols, input_rows, input_masks, \
                label_ids, label_weight, label_position, domain_label, \
                domain_label_seq, domain_weights = self.processor.next_batch_adv2()

                feed_dict = {self.input_ids: input_ids,
                             self.input_segments: input_segments,
                             self.input_cols: input_cols, self.input_rows: input_rows,
                             self.label_position: label_position, self.label_weight: label_weight,
                             self.label_ids: label_ids, self.domain_label: domain_label,
                             self.domain_label_seq: domain_label_seq,
                             self.domain_weights: domain_weights}
                #print(label_position.max())
                #input()
                #log_P = sess.run(log_P, feed_dict=feed_dict)
                #print(log_P)

                try:
                    g1, loss_, _ = sess.run([global_step, lm_loss, optimizer_model2], feed_dict=feed_dict)
                    g2, loss_2, _ = sess.run([global_step, lm_loss, optimizer_disc], feed_dict=feed_dict)
                except:
                    g1 = 0
                    g2 = 0
                    loss_ = 0
                    loss_2 = 0

                print(g1, i)
                print(g2, loss_, loss_2)
                print('--------')

                if i % 10000 == 0 and i != 0:
                    print('saved!')
                    saver = tf.train.Saver()
                    saver.save(sess, self.save_path)

    def eval(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base_rr.json')
        statement_file = open('triples_for_commom_facts', 'r', encoding='utf-8')

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids, is_training=False)

            column_memory, row_memory = self.Table_Memory_Network(sequence_output, hops=5, dropout=0.0)

            row_one_hot = tf.one_hot(self.input_rows, depth=100)
            column_one_hot = tf.one_hot(self.input_cols, depth=50)

            column_memory = tf.matmul(column_one_hot, column_memory)
            row_memory = tf.matmul(row_one_hot, row_memory)

            sequence_output = tf.concat([column_memory, row_memory, sequence_output], axis=2)

            #with tf.variable_scope("table_memory_hidden"):
            #    sequence_output = Fully_Connected(sequence_output, output=768, name='dense', activation=None)

            input_tensor = sequence_output
            output_weights = model.get_embedding_table()
            positions = self.label_position
            label_ids = self.label_ids
            label_weights = self.label_weight

            loss1, input_tensor, log_probs = self.get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                                                    label_ids, label_weights)
            print('BERT restored')
            #input()

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            lines = statement_file.read().split('\n')
            lines.pop(-1)

            print(len(lines))
            exit(111)

            total_rr = 0
            count = 0

            for line in lines:
                tk = line.split('[split]')
                print(tk)
                table_data = [['이름', tk[1]], [tk[0], '[MASK]']]

                input_ids_, input_segments, input_rows, input_cols, label_ids_, label_weight_, label_position_ = \
                    LM_Utils.get_table_input(table_data, self.tokenizer, self.vocab)

                # print(tokenization.convert_ids_to_tokens(self.f_tokenizer.inv_vocab, input_ids_))
                print(label_position_)

                feed_dict = {self.input_ids: input_ids_, self.input_segments: input_segments,
                             self.input_rows: input_rows, self.input_cols: input_cols,
                             self.label_position: label_position_, self.label_weight: label_weight_,
                             self.label_ids: label_ids_}

                lm_probs = sess.run(log_probs, feed_dict=feed_dict)
                lm_probs = np.array(lm_probs)

                rank = 32000

                try:
                    label_idx = self.vocab[tk[2].strip()]
                except:
                    continue

                arg_max = list(reversed(lm_probs.argsort(axis=1)[0]))
                c = 0
                for i in range(32000):
                    """
                    try:
                        int(self.f_tokenizer.inv_vocab[arg_max[i]])
                        continue
                    except:
                        None

                    if len(self.f_tokenizer.inv_vocab[arg_max[i]]) == 1:
                        continue
                    if self.f_tokenizer.inv_vocab[arg_max[i]] == 'D�':
                        continue
                    if self.f_tokenizer.inv_vocab[arg_max[i]] == 't�':
                        continue
                    """
                    c += 1

                    if arg_max[i] == label_idx:
                        rank = c
                        break

                rr = 1 / rank
                total_rr += rr
                count += 1

                for i in range(15):
                    print(self.f_tokenizer.inv_vocab[arg_max[i]], lm_probs[0, arg_max[i]])
                print('rank:', rank, 'word:', tk[2], lm_probs[0, label_idx])
                print('count:', count, 'avg rr:', total_rr / count)
                print('------')

    def count(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base_rr.json')
        statement_file = open('triples_for_commom_facts', 'r', encoding='utf-8')

        with tf.Session(config=config) as sess:
            print('BERT restored')
            # input()

            lines = statement_file.read().split('\n')
            lines.pop(-1)

            print(len(lines))

            total_rr = 0
            count = 0

            for line in lines:
                tk = line.split('[split]')
                try:
                    label_idx = self.vocab[tk[2].strip()]
                except:
                    continue
                count += 1

        print(count)

    def eval2(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base.json')

        statement_file = open('text_statements_for_common', 'r', encoding='utf-8')

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids, is_training=False)

            input_tensor = sequence_output
            output_weights = model.get_embedding_table()
            positions = self.label_position
            label_ids = self.label_ids
            label_weights = self.label_weight

            loss1, input_tensor, log_probs = self.get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                                                    label_ids, label_weights)
            print('BERT restored')
            #input()

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.bert_path)

            lines = statement_file.read().split('\n')
            lines.pop(-1)

            total_rr = 0
            count = 0

            for line in lines:
                tk = line.split('[split]')
                print(tk)
                table_data = [[tk[1]], ['']]

                input_ids_, input_segments, input_rows, input_cols, label_ids_, label_weight_, label_position_ = \
                    LM_Utils.get_table_input(table_data, self.tokenizer, self.vocab)

                #print(tokenization.convert_ids_to_tokens(self.f_tokenizer.inv_vocab, input_ids_))
                print(label_position_)

                feed_dict = {self.input_ids: input_ids_, self.input_segments: input_segments,
                             self.input_rows: input_rows, self.input_cols: input_cols,
                             self.label_position: label_position_, self.label_weight: label_weight_,
                             self.label_ids: label_ids_}

                lm_probs = sess.run(log_probs, feed_dict=feed_dict)
                lm_probs = np.array(lm_probs)

                rank = 32000

                try:
                    label_idx = self.vocab[tk[2].strip()]
                except:
                    continue

                arg_max = list(reversed(lm_probs.argsort(axis=1)[0]))
                for i in range(32000):
                    if arg_max[i] == label_idx:
                        rank = i + 1
                        break

                rr = 1 / rank
                total_rr += rr
                count += 1

                for i in range(15):
                    print(self.f_tokenizer.inv_vocab[arg_max[i]], lm_probs[0, arg_max[i]])
                print('rank:', rank, 'word:', tk[2], lm_probs[0, label_idx])
                print('count:', count, 'avg rr:', total_rr / count)
                print('------')

    def eval_test(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base.json')
        statement_file = open('triples_for_medical_knowledge', 'r', encoding='utf-8')

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids, is_training=False)

            column_memory, row_memory = self.Table_Memory_Network(sequence_output, hops=5, dropout=0.0)

            row_one_hot = tf.one_hot(self.input_rows, depth=100)
            column_one_hot = tf.one_hot(self.input_cols, depth=50)

            column_memory = tf.matmul(column_one_hot, column_memory)
            row_memory = tf.matmul(row_one_hot, row_memory)

            sequence_output = tf.concat([column_memory, row_memory, sequence_output], axis=2)

            with tf.variable_scope("table_memory_hidden"):
                sequence_output = Fully_Connected(sequence_output, output=768, name='dense', activation=None)

            input_tensor = sequence_output
            output_weights = model.get_embedding_table()
            positions = self.label_position
            label_ids = self.label_ids
            label_weights = self.label_weight

            loss1, input_tensor, log_probs = self.get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                                                    label_ids, label_weights)
            print('BERT restored')
            #input()

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            lines = statement_file.read().split('\n')
            lines.pop(-1)

            total_rr = 0
            count = 0

            dataholder = Dataholder_test.DataHolder()
            type_rr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            type_cnt = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]

            for i in range(157):
                input_ids, input_mask, input_segments, input_rows, input_cols, answer_tokens, answer_types, \
                label_position_, label_weight_, label_ids_ = dataholder.next_batch()

                feed_dict = {self.input_ids: input_ids, self.input_segments: input_segments,
                             self.input_rows: input_rows, self.input_cols: input_cols,
                             self.label_position: label_position_, self.label_weight: label_weight_,
                             self.label_ids: label_ids_}

                lm_probs = sess.run(log_probs, feed_dict=feed_dict)
                lm_probs = np.array(lm_probs)

                rank = 128005

                try:
                    label_idx = self.vocab[answer_tokens[0].strip()]
                except:
                    continue

                arg_max = list(reversed(lm_probs.argsort(axis=1)[0]))
                c = 0
                for i in range(1000):
                    """
                    try:
                        int(self.f_tokenizer.inv_vocab[arg_max[i]])
                        continue
                    except:
                        None

                    if len(self.f_tokenizer.inv_vocab[arg_max[i]]) == 1:
                        continue
                    if self.f_tokenizer.inv_vocab[arg_max[i]] == 'D�':
                        continue
                    if self.f_tokenizer.inv_vocab[arg_max[i]] == 't�':
                        continue
                    """
                    c += 1

                    if arg_max[i] == label_idx:
                        rank = c
                        break

                if rank == 128005:
                    continue

                rr = 1 / rank
                total_rr += rr

                answer_type = int(answer_types[0])
                print('type', answer_type)
                type_rr[answer_type] += 1 / rank
                type_cnt[answer_type] += 1

                count += 1

                for i in range(15):
                    print(self.f_tokenizer.inv_vocab[arg_max[i]], lm_probs[0, arg_max[i]])
                print('rank:', rank, 'word:', answer_tokens[0], lm_probs[0, label_idx])
                print('count:', count, 'avg rr:', total_rr / count)
                for i in range(6):
                    print(' ' + str(i) + ' , ', type_rr[i] / type_cnt[i])
                print('------')