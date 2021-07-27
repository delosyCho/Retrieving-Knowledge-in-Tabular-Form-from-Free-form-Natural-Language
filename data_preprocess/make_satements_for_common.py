def badchim_checker(word):    #아스키(ASCII) 코드 공식에 따라 입력된 단어의 마지막 글자 받침 유무를 판단해서 뒤에 붙는 조사를 리턴하는 함수
    last = word[-1]     #입력된 word의 마지막 글자를 선택해서
    criteria = (ord(last) - 44032) % 28     #아스키(ASCII) 코드 공식에 따라 계산 (계산법은 다음 포스팅을 참고하였습니다 : http://gpgstudy.com/forum/viewtopic.php?p=45059#p45059)
    if criteria == 0:       #나머지가 0이면 받침이 없는 것
        return True
    else:                   #나머지가 0이 아니면 받침 있는 것
        return False


count = 0

triple_file = open('triples_for_koelectra', 'r', encoding='utf-8')
lines = triple_file.read().split('\n')

statements_file = open('text_statements_for_common', 'w', encoding='utf-8')

for line in lines:
    if len(line) == 0:
        continue

    head = ''
    tail = ''

    tk = line.split('[split]')

    if badchim_checker(tk[1]) is True:
        head = '는'
    else:
        head = '은'

    tk = line.split('[split]')

    if badchim_checker(tk[2]) is True:
        tail = '다.'
    else:
        tail = '이다.'

    statement = tk[0] + '의 ' + tk[1] + head + ' [MASK]' + tail

    statements_file.write(tk[0] + '[split]' + statement + '[split]' + tk[2] + '\n')
    count += 1

statements_file.close()
print(count)