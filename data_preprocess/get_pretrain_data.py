from HTML_Utils import *
import os
import json
import os

import tokenization
import numpy as np

import Chuncker

from HTML_Processor import process_document
import Table_Holder
from random import random as rand

import Name_Tagging
import Ranking_ids


def RepresentsInt(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


"""
위키피디아 문서 내에서 테이블 데이터를 추출하고, 해당 테이블과 관련이 있는 텍스트오 함께 사전학습 데이터로 생성하는 코드입니다.

[CLS] 관련 텍스트 시퀀스 [SEP] 테이블 시퀀스

마스킹 정책은 BERT github을 참조하여 설정하였습니다.
"""

name_tagger = Name_Tagging.Name_tagger()

pattern = '<[^>]*>'

table_holder = Table_Holder.Holder()
chuncker = Chuncker.Chuncker()

#생성되는 입력의 최대 길이
max_length = 450

total_size = 20 * 80000

sequence_has_ans = np.zeros(shape=[total_size, max_length], dtype=np.int32)
segments_has_ans = np.zeros(shape=[total_size, max_length], dtype=np.int32)
positions_has_ans = np.zeros(shape=[total_size, max_length], dtype=np.int32)
ranks_has_ans = np.zeros(shape=[total_size, max_length], dtype=np.int32)
cols_has_ans = np.zeros(shape=[total_size, max_length], dtype=np.int32)
rows_has_ans = np.zeros(shape=[total_size, max_length], dtype=np.int32)
mask_has_ans = np.zeros(shape=[total_size, max_length], dtype=np.int32)
names_has_ans = np.zeros(shape=[total_size, max_length], dtype=np.int32)

label_ids = np.zeros(shape=[total_size, 20], dtype=np.int32)
label_position = np.zeros(shape=[total_size, 20], dtype=np.int32)
label_weight = np.zeros(shape=[total_size, 20], dtype=np.float)

data_num = 0

#사용하고자 하는 모델의 Vocab 및 Tokenizer 생성
vocab = tokenization.load_vocab(vocab_file='html_mecab_vocab_128000.txt')
tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

count = 0
false_count = 0
false_count2 = 0

cor = 0
wrong_case = 0

#전처리된 wikipedia 파일을 이용합니다.
html_data_file = open('wiki_html_processed', 'r', encoding='utf-8')
documents = html_data_file.read().split('[split]')

for d, document in enumerate(documents):
    #html text 자리
    doc = document

    #print(doc)
    #input()

    print(count, d, '/', len(documents), false_count, false_count2)

    paragraphs = doc.split('<h2>')
    sequences = []

    checked = False

    loaded_texts = []

    #print(paragraphs)

    for paragraph in paragraphs:
        texts = paragraph.split('<p>')
        if len(texts) > 1:
            for k in range(1, len(texts)):
                text = texts[k].split('</p>')[0]
                if len(text) > 10:
                    loaded_texts.append(text)

        if paragraph.find('table') == -1:
            continue

        try:
            title = paragraph.split('</h')[0]
        except:
            title = ''

        sub_paragraphs = paragraph.split('<h3>')

        for sub_paragraph in sub_paragraphs:
            if sub_paragraph.find('위키 백과') != -1 or sub_paragraph.find('위키백과') != -1:
                continue
            if sub_paragraph.find('이 문서는') != -1 or sub_paragraph.find('내용을 추가해 주세요') != -1:
                continue
            if sub_paragraph.find('파일 역사</h') != -1 or sub_paragraph.find('위키미디어') != -1:
                continue
            if sub_paragraph.find('<ul>') != -1 or sub_paragraph.find('위키미디어') != -1:
                continue

            try:
                title2 = title + ' ' + sub_paragraph.split('</h')[0]
            except:
                title2 = title

            title2 = re.sub(pattern=pattern, repl='', string=title2)

            if checked is True:
                break

            paragraph_, table_list = pre_process_document(sub_paragraph, answer_setting=False, a_token1='',
                                                          a_token2='')
            sub_paragraph = process_document(sub_paragraph)

            #print('text:', sub_paragraph.replace('\n', ''))
            #print('title:', title2.replace('\n', ''))

            add_text = ''

            if len(table_list) > 0:
                texts = sub_paragraph.split('[p]')
                if len(texts) > 1:
                    for k in range(1, len(texts)):
                        add_text += texts[k].split('[/p]')[0] + ' '

            for table_text in table_list:
                if table_text.find('영구 차단되었습니다') != -1:
                    continue

                if len(table_text.split('<td>')) < 5:
                    continue

                table_text = table_text.replace('<th', '<td')
                table_text = table_text.replace('</th', '</td')
                print('-------------------------')
                if len(add_text) < 5:
                    if len(loaded_texts) > 0:
                        add_text = loaded_texts[-1]
                    else:
                        add_text = title2

                #print(sub_paragraph.replace('\n', ''))
                #print(table_text.replace('\n', ''))
                #print(add_text.replace('\n', ''))
                #input()

                label_tokens = []
                label_positions = []

                #텍스트 자리
                query_tokens = tokenizer.tokenize(add_text)

                #query masking...
                query_tokens.insert(0, '[CLS]')
                query_tokens.append('[SEP]')

                table_text, child_texts = overlap_table_process(table_text=table_text)
                table_text = head_process(table_text=table_text)

                table_holder.get_table_text(table_text=table_text)
                table_data = table_holder.table_data
                lengths = []

                #print(table_text.replace('\n', ''))
                #print(table_data)

                for data in table_data:
                    lengths.append(len(data))
                if len(lengths) <= 0:
                    break

                for data in table_data:
                    lengths.append(len(data))
                if len(lengths) <= 0:
                    break

                length = max(lengths)

                table_data_ = table_data.copy()
                try:
                    rankings = Ranking_ids.numberToRanking(table_data_, None)
                except:
                    rankings = np.zeros(shape=[len(table_data), length], dtype=np.int32)

                rank_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)
                col_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)
                row_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)

                for j in range(length):
                    for i in range(len(table_data)):
                        col_ids[i, j] = j
                        row_ids[i, j] = i
                        rank_ids[i, j] = rankings[i][j]

                idx = 0
                tokens_ = []
                rows_ = []
                cols_ = []

                name_tags_ = []
                ranks_ = []
                positions_ = []

                for i in range(len(table_data)):
                    for j in range(len(table_data[i])):
                        if table_data[i][j] is not None:
                            tokens = tokenizer.tokenize(table_data[i][j])
                            #name_tag = name_tagger.get_name_tag(table_data[i][j])

                            for k, tk in enumerate(tokens):
                                tokens_.append(tk)
                                rows_.append(i + 1)
                                cols_.append(j)
                                ranks_.append(rank_ids[i][j])
                                name_tags_.append(0)
                                positions_.append(k)

                                if k >= 25:
                                    break

                            if len(tokens) > 25 and str(table_data[i][j]).find('[/answer]') != -1:
                                tokens_.append('[/answer]')
                                rows_.append(i)
                                cols_.append(j)
                                ranks_.append(rank_ids[i][j])
                                positions_.append(0)

                start_idx = -1
                end_idx = -1

                tokens = []
                rows = []
                cols = []
                ranks = []
                segments = []
                positions = []
                name_tags = []

                for j, tk in enumerate(query_tokens):
                    tokens.append(tk)
                    rows.append(0)
                    cols.append(0)
                    ranks.append(0)
                    segments.append(0)
                    name_tags.append(0)
                    positions.append(j)

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
                        name_tags.append(name_tags_[j])
                        positions.append(positions_[j])

                num_masking = int(len(tokens) * 0.15 * rand())
                if num_masking > 20:
                    num_masking = 20

                #masking tokens
                for j in range(num_masking):
                    mask_idx = int((len(tokens) - 1) * rand())
                    if tokens[mask_idx] == '[CLS]' or tokens[mask_idx] == '[SEP]':
                        continue

                    label_tokens.append(tokens[mask_idx])
                    label_positions.append(mask_idx)
                    tokens[mask_idx] = '[MASK]'

                ids = tokenization.convert_tokens_to_ids(vocab=vocab, tokens=tokens)
                labels_ids = tokenization.convert_tokens_to_ids(vocab=vocab, tokens=label_tokens)

                length = len(ids)
                if length > max_length:
                    length = max_length

                for j in range(length):
                    sequence_has_ans[count, j] = ids[j]
                    segments_has_ans[count, j] = segments[j]
                    positions_has_ans[count, j] = positions[j]
                    cols_has_ans[count, j] = cols[j]
                    rows_has_ans[count, j] = rows[j]
                    ranks_has_ans[count, j] = ranks[j]
                    mask_has_ans[count, j] = 1

                for j in range(len(labels_ids)):
                    label_ids[count, j] = labels_ids[j]
                    label_position[count, j] = label_positions[j]
                    label_weight[count, j] = 1.0

                count += 1

sequence_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
segments_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
positions_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
mask_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
cols_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
rows_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)
ranks_has_ans_ = np.zeros(shape=[count, max_length], dtype=np.int32)

label_ids_ = np.zeros(shape=[80000 * 4, 20], dtype=np.int32)
label_position_ = np.zeros(shape=[80000 * 4, 20], dtype=np.int32)
label_weight_ = np.zeros(shape=[80000 * 4, 20], dtype=np.float)


for i in range(count):
    sequence_has_ans_[i] = sequence_has_ans[i]
    segments_has_ans_[i] = segments_has_ans[i]
    positions_has_ans_[i] = positions_has_ans[i]
    mask_has_ans_[i] = mask_has_ans[i]
    rows_has_ans_[i] = rows_has_ans[i]
    cols_has_ans_[i] = cols_has_ans[i]
    ranks_has_ans_[i] = ranks_has_ans[i]

    label_ids_[i] = label_ids[i]
    label_position_[i] = label_position[i]
    label_weight_[i] = label_weight[i]

np.save('label_table_pre2', sequence_has_ans_)
np.save('sequence_table_pre2', sequence_has_ans_)
np.save('segments_table_pre2', segments_has_ans_)
np.save('mask_table_pre2', mask_has_ans_)
np.save('rows_table_pre2', rows_has_ans_)
np.save('cols_table_pre2', cols_has_ans_)
np.save('ranks_table_pre2', ranks_has_ans_)
np.save('label_ids2', label_ids_)
np.save('label_position2', label_position_)
np.save('label_weight2', label_weight_)




