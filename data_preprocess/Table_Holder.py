import Chuncker

import numpy as np

from itertools import product
from bs4 import BeautifulSoup


def argsort(seq):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
    #non-lambda version by Tony Veijalainen
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]


def table_to_2d(table_tag):
    rowspans = []  # track pending rowspans
    rows = table_tag.find_all('tr')

    # first scan, see how many columns we need
    colcount = 0
    for r, row in enumerate(rows):
        cells = row.find_all(['td', 'th'], recursive=False)
        # count columns (including spanned).
        # add active rowspans from preceding rows
        # we *ignore* the colspan value on the last cell, to prevent
        # creating 'phantom' columns with no actual cells, only extended
        # colspans. This is achieved by hardcoding the last cell width as 1.
        # a colspan of 0 means “fill until the end” but can really only apply
        # to the last cell; ignore it elsewhere.
        colcount = max(
            colcount,
            sum(int(c.get('colspan', 1)) or 1 for c in cells[:-1]) + len(cells[-1:]) + len(rowspans))
        # update rowspan bookkeeping; 0 is a span to the bottom.
        rowspans += [int(c.get('rowspan', 1)) or len(rows) - r for c in cells]
        rowspans = [s - 1 for s in rowspans if s > 1]

    # it doesn't matter if there are still rowspan numbers 'active'; no extra
    # rows to show in the table means the larger than 1 rowspan numbers in the
    # last table row are ignored.

    # build an empty matrix for all possible cells
    table = [[None] * colcount for row in rows]

    # fill matrix from row data
    rowspans = {}  # track pending rowspans, column number mapping to count
    for row, row_elem in enumerate(rows):
        span_offset = 0  # how many columns are skipped due to row and colspans
        for col, cell in enumerate(row_elem.find_all(['td', 'th'], recursive=False)):
            # adjust for preceding row and colspans
            col += span_offset
            while rowspans.get(col, 0):
                span_offset += 1
                col += 1

            # fill table data
            rowspan = rowspans[col] = int(cell.get('rowspan', 1)) or len(rows) - row
            colspan = int(cell.get('colspan', 1)) or colcount - col
            # next column is offset by the colspan
            span_offset += colspan - 1
            value = cell.get_text()
            for drow, dcol in product(range(rowspan), range(colspan)):
                try:
                    table[row + drow][col + dcol] = value
                    rowspans[col + dcol] = rowspan
                except IndexError:
                    # rowspan or colspan outside the confines of the table
                    pass

        # update rowspan bookkeeping
        rowspans = {c: s - 1 for c, s in rowspans.items() if s > 1}

    return table


class Holder:
    def __init__(self):
        None

    def get_table_text(self, table_text):
        table2 = BeautifulSoup(table_text, 'html.parser')

        try:
            table_data_list = table_to_2d(table_tag=table2)

            tr_lines = table_text.split('<tr>')
            tr_lines.pop(0)

            table_heads = []
            table_data = []

            if len(tr_lines) == len(table_data_list):
                for i in range(len(tr_lines)):
                    if tr_lines[i].find('<th') != -1:
                        table_heads.append(table_data_list[i])
                    else:
                        table_data.append(table_data_list[i])
            # print(len(table_heads), len(table_data))
            if len(table_heads) > 0:
                self.table_head = table_heads[-1]
            else:
                self.table_head = table_data[0]
            self.table_data = table_data

            transposed_table = True
            for line in tr_lines:
                if line.find('<th') == -1:
                    transposed_table = False

            if transposed_table is True:
                self.table_head = []
                self.table_data = []
                for table_data in table_data_list:
                    head_word = table_data.pop(0)
                    self.table_head.append(head_word)
                    self.table_data.append(table_data)

        except:
            self.table_head = []
            self.table_data = []
            return

    def get_data_line(self, question):
        chuncker = Chuncker.Chuncker()
        chuncker.get_feautre(question)

        scores = []

        for data in self.table_data:
            string = ''
            for word in data:
                if word is not None:
                    string += word + ' '

            scores.append(chuncker.get_chunk_score(string))

        if len(scores) == 0:
            return 0

        return np.array(scores, dtype=np.float32).argmax()


def detect_num_word(word):
    if len(word.strip()) == 0 or word == ' ':
        return False, None

    word = word.replace(',', '')

    zero_ord = ord('0')
    nine_ord = ord('9')

    start_idx = 0
    end_idx = 0

    for i in range(len(word)):
        start_idx = i
        if zero_ord <= ord(word[i]) <= nine_ord:
            break

    for i in list(reversed(range(len(word)))):
        end_idx = i
        if zero_ord <= ord(word[i]) <= nine_ord:
            break

    for i in range(start_idx, end_idx):
        if not ((zero_ord <= ord(word[i]) <= nine_ord) or word[i] == '.'):
            return False, None

    if end_idx < start_idx:
        return False, None

    if 1 + (end_idx - start_idx) / len(word) < 0.3:
        return False, None

    try:
        float(word[start_idx: end_idx + 1])
    except:
        return False, None

    return True, word[start_idx: end_idx + 1]


def detect_simple_num_word(word):
    word = word.replace(',', '')

    zero_ord = ord('0')
    nine_ord = ord('9')

    is_num = True

    for i in range(len(word)):
        if not (zero_ord <= ord(word[i]) <= nine_ord) and word[i] == '#':
            is_num = False

    return is_num


def get_space_of_num(num):
    num = int(num)

    space = 0
    while True:
        if -1.0 < num < 1.0:
            break
        space += 1
        num = num / 10

    return space


def get_space_num_lists(num):
    space_lists = []
    num = float(num)
    num = int(num * 100)

    space_lists.append((num % 10))
    space_lists.append(int((num % 100) / 10))
    space_lists.append(int((num % 1000) / 100))
    space_lists.append(int((num % 10000) / 1000))
    space_lists.append(int((num % 100000) / 10000))
    space_lists.append(int((num % 1000000) / 100000))
    space_lists.append(int((num % 10000000) / 1000000))
    space_lists.append(int((num % 100000000) / 10000000))
    space_lists.append(int((num % 1000000000) / 100000000))
    space_lists.append(int((num % 10000000000) / 1000000000))

    return space_lists


def trim_number(word):
    result = ''
    num_result = ''

    zero_ord = ord('0')
    nine_ord = ord('9')

    for i in range(len(word)):
        if not (zero_ord <= ord(word[i]) <= nine_ord):
            result += word[i]
        else:
            num_result += word[i]

    return '0 ' + result, num_result


def get_ranks_of_table_row_wise(table_data):
    num_row = len(table_data)
    num_col = len(table_data[0])

    result = np.zeros(shape=[num_row, num_col], dtype=np.int32)

    for i in range(num_col):
        num_list = []

        for j in range(num_row):
            if table_data[j][i] is not None:
                is_num, number_value = detect_num_word(table_data[j][i])
            else:
                is_num = False
                number_value = 0

            if is_num is True:
                num_list.append(float(number_value))
            else:
                num_list.append(99999)

        sorted_idxs = argsort(num_list)
        #print(num_list, sorted_idxs)
        for j in range(num_row):
            if num_list[sorted_idxs[j]] == 99999:
                result[sorted_idxs[j], i] = -1
            else:
                result[sorted_idxs[j], i] = j

    return result


def get_ranks_of_table_column_wise(table_data):
    num_row = len(table_data)
    num_col = len(table_data[0])

    result = []

    for i in range(num_row):
        num_list = []

        for j in range(num_col):
            is_num, number_value = detect_num_word(table_data[i][j])

            if is_num is True:
                num_list.append(int(number_value))
            else:
                num_list.append(-99999)

        sorted_idxs = argsort(num_list)
        result.append(sorted_idxs)

    return result





