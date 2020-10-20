For our project, we implement two kinds of model:
    1. Bert based model
    2. If-else decision tree model

Get Started:
    To run our code, 
        
        1. first you can use script/rm_whitespace.sh to remove the whitespaces of the file name
        
        2. secondly, "cd classifier/bert/model/" change to the working directory
        
        3. third, modified the key word and parameter in train.py and test.py respectively
        
        4. "python3 train.py"

1. Bert based 
    a language model based method to first extract those sentences with keywords and embed with pre-trained bert tokenizer. We then use a layer of MLP to classify which class the sentence belongs to.

2. If-else decision tree model
