
def Dict():
    Path = "words.txt"
    Dictionary = dict()
    Representation = 1
    with open(Path) as f:
        contents = f.readlines()
        len_of_contents = len(contents)
        for i in range(len_of_contents):
            contents[i] = contents[i].split()[-1]
        for j in contents:
            for k in j:
                if k in Dictionary:
                    None
                else:
                    Dictionary[k] = Representation
                    Representation += 1
    Dictionary[' '] = 79
    Dictionary[None] = 80
    return Dictionary
   

        
