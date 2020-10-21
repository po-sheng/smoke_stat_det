For our project, we are using python3 to implement two kinds of model:
    
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

    below shows our hyperparameters after doing grid search on out model
    
    params = {                                                  
            "bert_model": "bert-base-cased",
            "train_batch_size": 5,
            "eval_batch_size": 1,
            "max_seq_length": 128,
            "num_mlp_layers": 1,
            "num_mlp_hiddens": 20,
            "epochs": 6,
            "learning_rate": 1e-5,
            "use_cuda": True}

2. If-else decision tree model
