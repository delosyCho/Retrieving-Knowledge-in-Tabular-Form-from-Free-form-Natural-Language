def badchim_checker(word):    #아스키(ASCII) 코드 공식에 따라 입력된 단어의 마지막 글자 받침 유무를 판단해서 뒤에 붙는 조사를 리턴하는 함수
    last = word[-1]     #입력된 word의 마지막 글자를 선택해서
    criteria = (ord(last) - 44032) % 28     #아스키(ASCII) 코드 공식에 따라 계산 (계산법은 다음 포스팅을 참고하였습니다 : http://gpgstudy.com/forum/viewtopic.php?p=45059#p45059)
    if criteria == 0:       #나머지가 0이면 받침이 없는 것
        return True
    else:                   #나머지가 0이 아니면 받침 있는 것
        return False


count = 0

triple_file = open('triples_for_medical_knowledge', 'r', encoding='utf-8')

lines = triple_file.read().split('\n')

dic_files = open('medical_relation.txt', 'r', encoding='utf-8')
dic_lines = dic_files.read().split('\n')

statements_file = open('text_statements_for_medical', 'w', encoding='utf-8')
statements_file2 = open('tripe_statements_for_medical', 'w', encoding='utf-8')

dic_tks = []
for dic_line in dic_lines:
    tk = dic_line.split('\t')
    dic_tks.append(tk)

for line in lines:
    if len(line) == 0:
        continue

    #head = ''
    tail = ''

    tk = line.split('[split]')

    if badchim_checker(tk[2]) is True:
        tail = '다.'
    else:
        tail = '이다.'

    relation = ''
    statement = ''

    check = False
    for dic_tk in dic_tks:
        if tk[1] == dic_tk[0]:
            relation = dic_tk[1]
            statement = dic_tk[2]
            check = True

    if check is False:
        continue

    statement = statement + tail

    statements_file.write(tk[0] + '[split]' + statement + '[split]' + tk[2] + '\n')
    statements_file2.write(tk[0] + '[split]' + relation + '[split]' + tk[2] + '\n')

    count += 1

statements_file.close()
statements_file2.close()

print(count)