
def DictY():

    Path = "words.txt"
    dic = dict()
    with open(Path) as f:
        content = f.readlines()
        length = len(content)
        for i in range(length):
            content[i] = content[i].split()
        for i in range(len(content)):
            dic[content[i][0]] = content[i][-1]

    return dic

