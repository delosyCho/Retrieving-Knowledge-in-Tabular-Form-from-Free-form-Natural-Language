import numpy as np
import tokenization

import random
from transformers import AutoTokenizer, AutoModelForMaskedLM

title_file = open('wiki_titles')

max_masking = 20

vocab = tokenization.load_vocab('vocab.txt')
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

eval_file = open('triples_for_koelectra', 'r', encoding='utf-8')
statements_file = open('text_statements_for_common', 'r', encoding='utf-8')

triple_lines = eval_file.read().split('\n')
text_lines = statements_file.read().split('\n')

input_ids = np.zeros(shape=[len(triple_lines) - 1, 2, 256], dtype=np.int32)
input_segments = np.zeros(shape=[len(triple_lines) - 1, 2, 256], dtype=np.int32)
input_cols = np.zeros(shape=[len(triple_lines) - 1, 2, 256], dtype=np.int32)
input_rows = np.zeros(shape=[len(triple_lines) - 1, 2, 256], dtype=np.int32)

label_ids = np.zeros(shape=[len(triple_lines) - 1, 2, max_masking], dtype=np.int32)
label_position = np.zeros(shape=[len(triple_lines) - 1, 2, max_masking], dtype=np.int32)
label_weight = np.zeros(shape=[len(triple_lines) - 1, 2, max_masking], dtype=np.float)

idx = 0
cnt = 0
count = 0

triples = []
texts = []

while True:
    print(count)

    if len(triples) == 0:
        triples.append(triple_lines[count])
        texts.append(text_lines[count])
    else:
        print(triple_lines[count])
        if triples[0][0:2] == triple_lines[count][0:2]:
            triples.append(triple_lines[count])
            texts.append(text_lines[count])
        else:
            for i in range(10):
                rand_idx = int(random.random() * len(triples))

                tks = triples[rand_idx].split('[split]')
                if len(tokenizer.tokenize(tks[2])) <= 5:
                    break

            #triple statement
            tokens = ['[CLS]']
            segments = [0]
            rows = [0]
            cols = [0]

            label_tokens = []
            label_positions = []

            table_data = []

            table_line = ['이름']
            for i in range(len(triples)):
                tks = triples[i].split('[split]')
                table_line.append(tks[1])

            table_data.append(table_line)

            table_line = [triples[0].split('[split]')[0]]
            for i in range(len(triples)):
                tks = triples[i].split('[split]')
                table_line.append(tks[2])

            table_data.append(table_line)

            for i, table_line in enumerate(table_data):
                for j, td in enumerate(table_line):
                    tokens_ = tokenizer.tokenize(td)

                    for token in tokens_:
                        if i == 1 and j == rand_idx + 1:
                            label_tokens.append(token)
                            label_positions.append(len(tokens))

                            token = '[MASK]'

                        tokens.append(token)
                        segments.append(1)
                        rows.append(i + 2)
                        cols.append(j + 1)

            token_ids = tokenization.convert_tokens_to_ids(tokens=tokens, vocab=vocab)
            label_ids_ = tokenization.convert_tokens_to_ids(tokens=label_tokens, vocab=vocab)

            length = len(token_ids)
            if length > 256:
                length = 256

            for i in range(length):
                input_ids[cnt, 0, i] = token_ids[i]
                input_segments[cnt, 0, i] = segments[i]
                input_rows[cnt, 0, i] = rows[i]
                input_cols[cnt, 0, i] = cols[i]

            length = len(label_ids_)
            if length > max_masking:
                length = max_masking

            for i in range(length):
                if label_positions[i] < 256:
                    label_ids[cnt, 0, i] = label_ids_[i]
                    label_position[cnt, 0, i] = label_positions[i]
                    label_weight[cnt, 0, i] = 1

            #text statement
            tokens = ['[CLS]']
            segments = [0]
            rows = [0]
            cols = [0]

            label_tokens = []
            label_positions = []

            table_data = []

            subject = triples[rand_idx].split('[split]')[2]
            for i in range(len(texts)):
                tks = texts[i].split('[split]')
                statement = tks[1]

                if i == rand_idx:
                    mask_tokens = tokenizer.tokenize(tks[2])
                    mask_statement = ''
                    for mask_token in mask_tokens:
                        label_tokens.append(mask_token)
                        mask_statement += ' [MASK]'
                    print(mask_tokens, label_tokens)
                    print(mask_statement)

                    statement = statement.replace('[MASK]', mask_statement)
                else:
                    statement = statement.replace('[MASK]', tks[2])
                #statement = statement.replace('이다', ' ##이다')
                print(statement)
                tokens_ = tokenizer.tokenize(statement)

                for token in tokens_:
                    if token == '[MASK]':
                        label_positions.append(len(tokens))

                    tokens.append(token)
                    segments.append(1)
                    rows.append(0)
                    cols.append(0)

            token_ids = tokenization.convert_tokens_to_ids(tokens=tokens, vocab=vocab)
            label_ids_ = tokenization.convert_tokens_to_ids(tokens=label_tokens, vocab=vocab)

            length = len(token_ids)
            if length > 256:
                length = 256

            for i in range(length):
                input_ids[cnt, 1, i] = token_ids[i]
                input_segments[cnt, 1, i] = segments[i]
                input_rows[cnt, 1, i] = rows[i]
                input_cols[cnt, 1, i] = cols[i]

            length = len(label_ids_)
            if length > max_masking:
                length = max_masking

            for i in range(length):
                print(i, len(label_positions), len(label_ids_), label_tokens)
                label_ids[cnt, 1, i] = label_ids_[i]
                label_position[cnt, 1, i] = label_positions[i]
                label_weight[cnt, 1, i] = 1

            cnt += 1

            triples = [triple_lines[count]]
            texts = [text_lines[count]]

    count += 1

    if count == 600000:
        break

print('count:', cnt, count)

input_ids_ = np.zeros(shape=[cnt, 2, 256], dtype=np.int32)
input_segments_ = np.zeros(shape=[cnt, 2, 256], dtype=np.int32)
input_cols_ = np.zeros(shape=[cnt, 2, 256], dtype=np.int32)
input_rows_ = np.zeros(shape=[cnt, 2, 256], dtype=np.int32)

label_ids_ = np.zeros(shape=[cnt, 2, max_masking], dtype=np.int32)
label_position_ = np.zeros(shape=[cnt, 2, max_masking], dtype=np.int32)
label_weight_ = np.zeros(shape=[cnt, 2, max_masking], dtype=np.float)

for i in range(cnt):
    input_ids_[i] = input_ids[i]
    input_segments_[i] = input_segments[i]
    input_cols_[i] = input_cols[i]
    input_rows_[i] = input_rows[i]

    label_ids_[i] = label_ids[i]
    label_position_[i] = label_position[i]
    label_weight_[i] = label_weight[i]

np.save("input_ids_statements", input_ids_)
np.save("input_segments_statements", input_segments_)
np.save("input_cols_statements", input_cols_)
np.save("input_rows_statements", input_rows_)

np.save("label_ids_statements", label_ids_)
np.save("label_position_statements", label_position_)
np.save("label_weight_statements", label_weight_)

