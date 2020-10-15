import os
from typing import List, Dict

def dataLoader(path: str, keyWords: List[str]) -> Dict:
    
    """
    Extract all the sentence containing specific key words
    
    ---------
    - Input -
    ---------
        path : str
            the path of the directory containing all the data
        keyWord : List[str]
            all the keyword correspoding to smoking status

    ----------
    - Output -                          # each list element represent a data
    ----------
        data{                               
            "name" : list[str]              # file name of the data    
            "sent" : list[str]              # multiple sentences in a data may be concatenated into a single string, may have ""
            "tokens" : list[list[str]]      # first dimension represent each data, second dimension represent tokens in sentece
            "label" : list[int]             # 0:current, 1:not_smoke, 2:past, 3:unknown, 4:test_data
        }

    """

    data = {}
    data["name"] = []
    data["sent"] = []
    data["label"] = []
    data["tokens"] = []
    idx2label = {0: "CURRENT", 1: "NON", 2: "PAST", 3: "UNKNOWN", 4: "TEST_DATA"}

    # find all text file under the path directory
    if os.path.isdir(path): 
        files = os.listdir(path)    # the path is a directory
    else:
        files = path                # the path is a file
    for file in files:
        if file.endswith(".txt"):

            # get file label
            label = "UNKNOWN"
            for idx in range(4):
                if file.find(idx2label[idx]) != -1:
                    label = idx2label[idx]
                    break
            
            # open file and read each line and add to "data" if it contains key words
            f = open(path+file, 'r')
            sent = ""
            tokens = []
            for line in f.readlines():
                line = line.split("\n")[0]
                lower_line = line.lower()
                for keyWord in keyWords:
                    if lower_line.find(keyWord) != -1:
                        sent = sent + line
                        tokens = tokens + line.split()
                        break

            # assing attribute for each file
            data["name"].append(file)
            data["sent"].append(sent)
            data["label"].append(label)
            data["tokens"].append(tokens)
            
            f.close()
    return data

if __name__ == '__main__':
    
    path = "../data/train/"
    keyWords = ["smoke", "smoker", "smokes", "smoking", "smoked", "tobacco"]
    
    data = dataLoader(path, keyWords)
    print(data["sent"])
    
