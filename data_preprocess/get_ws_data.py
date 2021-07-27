from HTML_Utils import *
import os
import json
import os

import tokenization
import numpy as np

import Chuncker

from HTML_Processor import process_document
import Table_Holder

import Name_Tagging
import Ranking_ids
from transformers import AutoTokenizer

from Table_Holder import detect_num_word, detect_simple_num_word, get_space_of_num, get_space_num_lists, trim_number, \
    get_ranks_of_table_row_wise

"""
KorQuAD 내의 Table에 대한 데이터를 추출하고 테이블에 대한 기계독해 학습 데이터 생성을 위한 코드입니다.

[CLS] Question Tokens [SEP] Table Tokens

"""

def RepresentsInt(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


name_tagger = Name_Tagging.Name_tagger()
table_holder = Table_Holder.Holder()
chuncker = Chuncker.Chuncker()

max_length = 512

#KorQuAD 학습 json 파일들이 있는 폴더 위치
path_dir = 'F:\\korquad2_data'

sequence_has_ans = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
segments_has_ans = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
positions_has_ans = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
ranks_has_ans = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
names_has_ans = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
cols_has_ans = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
rows_has_ans = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
mask_has_ans = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
numeric_space = np.zeros(shape=[80000 * 2, 10, max_length], dtype=np.int32)
numeric_mask = np.zeros(shape=[80000 * 2, 10, max_length], dtype=np.int32)
numeric_value = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
head_mask = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
answer_span = np.zeros(shape=[80000 * 2, 2], dtype=np.int32)

sequence_has_ans_z = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
segments_has_ans_z = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
positions_has_ans_z = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
ranks_has_ans_z = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
names_has_ans_z = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
cols_has_ans_z = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
rows_has_ans_z = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)
mask_has_ans_z = np.zeros(shape=[80000 * 2, max_length], dtype=np.int32)

rank_2d = np.zeros(shape=[100000, 50, 15], dtype=np.int16)


file_list = os.listdir(path_dir)
file_list.sort()
file_list.pop(-1)
file_list.pop(-1)

questions = []
for file_name in file_list[0:1]:
    print(file_name, '...')
    in_path = path_dir + '\\' + file_name
    data = json.load(open(in_path, 'r', encoding='utf-8'))

    for article in data['data']:
        for qas in article['qas']:
            question = qas['question']
            questions.append(str(question))

q_idx = np.array(range(len(questions)), dtype=np.int32)
np.random.shuffle(q_idx)

print(file_list)
data_num = 0

"""
모델을 위한 tokenizer와 vocab 생성 코드입니다.
사용하시는 모델의 tokenizer와 vocab에 맞게 수정하시면 됩니다.
"""
vocab = tokenization.load_vocab(vocab_file='vocab.txt')
tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

tokenizer_ = AutoTokenizer.from_pretrained("klue/roberta-base")
tokenizer_.add_tokens('[STA]')
tokenizer_.add_tokens('[END]')
tokenizer_.add_tokens('[table]')
tokenizer_.add_tokens('[/table]')
tokenizer_.add_tokens('[list]')
tokenizer_.add_tokens('[/list]')
tokenizer_.add_tokens('[h3]')
tokenizer_.add_tokens('[td]')

#print(tokenizer_.tokenize('나는[STA] 그와 밥[END] 먹었다'))
#input()
count = 0
false_count = 0
false_count2 = 0

cor = 0
wrong_case = 0

for file_name in file_list:

    print(file_name, 'processing....', data_num)

    in_path = path_dir + '\\' + file_name
    data = json.load(open(in_path, 'r', encoding='utf-8'))

    for article in data['data']:
        doc = str(article['context'])
        doc = doc.replace('\t', ' ')
        doc = doc.replace('\a', ' ')

        print(count, false_count, false_count2, file_name)

        for qas in article['qas']:
            error_code = -1

            answer = qas['answer']
            answer_start = answer['answer_start']
            answer_text = answer['text']
            question = qas['question']

            chuncker.get_feautre(query=question)

            if len(answer_text) > 40:
                continue

            query_tokens = []
            query_tokens.append('[CLS]')
            q_tokens = tokenizer_.tokenize(question.lower())
            for tk in q_tokens:
                query_tokens.append(tk)
            query_tokens.append('[SEP]')
            ######
            # 정답에 ans 토큰을 임베딩하기 위한 코드
            ######

            ans1 = ''
            ans2 = ''
            if doc[answer_start - 1] == ' ':
                ans1 = ' [STA] '
            else:
                ans1 = ' [STA]'

            if doc[answer_start + len(answer_text)] == ' ':
                ans2 = ' [END] '
            else:
                ans2 = ' [END]'

            doc_ = doc[0: answer_start] + ans1 + answer_text + ans2 + doc[answer_start + len(answer_text): -1]
            doc_ = str(doc_)
            #
            #####

            paragraphs = doc_.split('<h2>')
            sequences = []

            tables = []

            for paragraph in paragraphs:
                paragraph_, table_list = pre_process_document(paragraph, answer_setting=False,
                                                              a_token1='',
                                                              a_token2='')

                for table_text in table_list:
                    tables.append(table_text)

            chuncker.get_feautre(query=question)

            ch_scores = []
            selected = -1

            check_table_case = False
            for i, table_text in enumerate(tables):
                if table_text.find('[STA]') != -1 and table_text.find('table') != -1:
                    check_table_case = True
                    selected = i
                    ch_scores.append(-9999)
                else:
                    ch_scores.append(chuncker.get_chunk_score(table_text))
            ch_scores = np.array(ch_scores, dtype=np.float32)

            #print('num of tables:', len(tables))

            if check_table_case is False:
                continue
            if len(tables) == 0:
                continue
            #print('!')

            table_text = tables[selected]
            table_text = table_text.replace('<th', '<td')
            table_text = table_text.replace('</th', '#HEADDATA#</td')

            table_text = table_text.replace(' <td>', '<td>')
            table_text = table_text.replace(' <td>', '<td>')
            table_text = table_text.replace('\n<td>', '<td>')
            table_text = table_text.replace('</td> ', '</td>')
            table_text = table_text.replace('</td> ', '</td>')
            table_text = table_text.replace('\n<td>', '<td>')
            table_text = table_text.replace('[STA]<td>', '<td>[STA] ')
            table_text = table_text.replace('</td>[END]', ' [END]</td>')
            table_text = table_text.replace('</td>', '  </td>')
            table_text = table_text.replace('<td>', '<td> ')
            table_text = table_text.replace('[STA]', '[STA] ')
            table_text = table_text.replace('[END]', ' [END]')

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

            #table_data_ = table_data.copy()
            #rankings = Ranking_ids.numberToRanking(table_data_, table_head)

            #print(table_data)
            #input()

            for j in range(length):
                for i in range(len(table_data)):
                    col_ids[i, j] = j
                    row_ids[i, j] = i
                    #rank_ids[i, j] = rankings[i][j]

            idx = 0
            tokens_ = []
            rows_ = []
            cols_ = []
            spaces_ = []
            head_masks_ = []
            #name_tags_ = []
            #ranks_ = []
            positions_ = []
            num_values_ = []

            ranks_data = get_ranks_of_table_row_wise(table_data)

            """
            for _data in table_data:
                print(_data)
            print('-------------')
            for rank_data in ranks_data:
                print(rank_data)

            print(table_text.replace('\n', ''))
            input()
            """

            for i in range(len(table_data)):
                for j in range(len(table_data[i])):
                    if table_data[i][j] is not None:
                        is_head = 0
                        if table_data[i][j].find('#HEADDATA#') != -1:
                            is_head = 1
                            table_data[i][j] = table_data[i][j].replace('#HEADDATA#', '')

                        tokens = tokenizer_.tokenize(table_data[i][j])
                        #name_tag = name_tagger.get_name_tag(table_data[i][j])

                        is_num, number_value = detect_num_word(table_data[i][j])
                        num_text = '0'
                        if is_num is True:
                            data_text, num_text = trim_number(table_data[i][j])
                            table_data[i][j] = data_text

                        for k, tk in enumerate(tokens):
                            tokens_.append(tk)
                            rows_.append(i + 1)
                            cols_.append(j)
                            positions_.append(k)
                            head_masks_.append(is_head)

                            if is_num is True:
                                if detect_simple_num_word(tk) is True:
                                    space_lists = get_space_num_lists(num_text)
                                    spaces_.append(space_lists)
                                    num_values_.append(int(num_text))
                                else:
                                    spaces_.append(-1)
                                    num_values_.append(-1)
                            else:
                                spaces_.append(-1)
                                num_values_.append(-1)

                            if k >= 40:
                                break

                        if len(tokens) > 40 and str(table_data[i][j]).find('[END]') != -1:
                            tokens_.append('[END]')
                            rows_.append(i + 1)
                            cols_.append(j)
                            positions_.append(0)
                            spaces_.append(-1)
                            head_masks_.append(0)
                            num_values_.append(-1)

            start_idx = -1
            end_idx = -1

            tokens = []
            rows = []
            cols = []
            #ranks = []
            segments = []
            #name_tags = []
            positions = []
            spaces = []
            head_masks = []
            num_values = []

            for j, tk in enumerate(query_tokens):
                tokens.append(tk)
                rows.append(0)
                cols.append(0)
                #ranks.append(0)
                segments.append(0)
                #name_tags.append(0)
                positions.append(j)
                spaces.append(-1)
                head_masks.append(0)
                num_values.append(0)

            for j, tk in enumerate(tokens_):
                if tk == '[STA]':
                    start_idx = len(tokens)
                elif tk == '[END]':
                    end_idx = len(tokens) - 1
                else:
                    tokens.append(tk)
                    rows.append(rows_[j] + 1)
                    cols.append(cols_[j] + 1)
                    segments.append(1)
                    #ranks.append(ranks_[j])
                    #name_tags.append(name_tags_[j])
                    positions.append(positions_[j])
                    spaces.append(spaces_[j])
                    head_masks.append(head_masks_[j])

            ids = tokenization.convert_tokens_to_ids(vocab=vocab, tokens=tokens)

            if end_idx > max_length or start_idx > max_length:
                false_count += 1
                continue

            if start_idx == -1 or end_idx == -1:
                false_count2 += 1
                continue

            length = len(ids)
            if length > max_length:
                length = max_length

            for j in range(length):
                sequence_has_ans[count, j] = ids[j]
                segments_has_ans[count, j] = segments[j]
                positions_has_ans[count, j] = positions[j]
                cols_has_ans[count, j] = cols[j]
                rows_has_ans[count, j] = rows[j]
                mask_has_ans[count, j] = 1
                head_mask[count, j] = head_masks[j]
                if spaces[j] != -1:
                    for k in range(10):
                        numeric_space[count, k, j] = spaces[j][k]
                        if spaces[j][k] != 0:
                            numeric_mask[count, k, j] = 1
                #ranks_has_ans[count, j] = ranks[j]
                #names_has_ans[count, j] = name_tags[j]
            answer_span[count, 0] = start_idx
            answer_span[count, 1] = end_idx

            row_len = len(table_data)
            if row_len > 50:
                row_len = 50

            col_len = len(table_data[0])
            if col_len > 15:
                col_len = 15

            for i in range(row_len):
                for j in range(col_len):
                    rank_2d[count, i, j] = ranks_data[i, j]

            ##########
            count += 1
            checked = True

sequence_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
segments_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
positions_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
mask_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
cols_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
rows_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
ranks_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
names_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
numeric_space_ = np.zeros(shape=[count, 10, max_length], dtype=np.int32)
numeric_mask_ = np.zeros(shape=[count, 10, max_length], dtype=np.int32)
head_mask_ = np.zeros(shape=[count, max_length], dtype=np.int32)

answer_span_ = np.zeros(shape=[count, 2], dtype=np.int32)

rank_2d_ = np.zeros(shape=[count, 50, 15], dtype=np.int16)

for i in range(count):
    sequence_has_ans_[i] = sequence_has_ans[i]
    segments_has_ans_[i] = segments_has_ans[i]
    positions_has_ans_[i] = positions_has_ans[i]
    mask_has_ans_[i] = mask_has_ans[i]
    rows_has_ans_[i] = rows_has_ans[i]
    cols_has_ans_[i] = cols_has_ans[i]
    ranks_has_ans_[i] = ranks_has_ans[i]
    names_has_ans_[i] = names_has_ans[i]
    numeric_space_[i] = numeric_space[i]
    numeric_mask_[i] = numeric_mask[i]
    head_mask_[i] = head_mask[i]
    answer_span_[i] = answer_span[i]

    rank_2d_[i] = rank_2d[i]

np.save('sequence_table', sequence_has_ans_)
np.save('segments_table', segments_has_ans_)
np.save('positions_table', positions_has_ans_)
np.save('mask_table', mask_has_ans_)
np.save('rows_table', rows_has_ans_)
np.save('cols_table', cols_has_ans_)
np.save('ranks_table', ranks_has_ans_)
np.save('names_table', names_has_ans_)
np.save('numeric_space', numeric_space_)
np.save('numeric_mask', numeric_mask_)
np.save('head_mask', head_mask)
np.save('answer_span_table', answer_span_)

np.save('rank_2d', rank_2d)

