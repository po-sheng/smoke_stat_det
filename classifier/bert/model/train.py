import os
import numpy as np
import torch
import time
from typing import List, Dict
from data_preprocess import dataLoader
from bert_sklearn import BertClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def paramAssign(model, params: Dict):
    """
    Assign each setting of dictionary value to correspodence model parameter which is "params"'s key 

    -----------
    -- Input --
    -----------
        model:
            the model need to be assign parameters
        
        params: Dict
            a dictionary contains all the parameters and their values

    """
    for param, value in params.items():
        setattr(model, param, value)

def train(grid_search: bool, params: Dict, grid_params: Dict, x: List[str], y: List[int]):
    
    """
    Train a Bert embeddings with some fully-connected layers behind

    -----------
    -- Input --
    -----------
        grid_search: boolean
            whether we are using grid search or not
        
        params: Dict
            contains all the parameter and their corresponding value for model
        
        grid_params: Dict
            some variables need to be adjust during grid search
        
        x: List[str]
            the input data, a list of string
        
        y: List[int]
            the input data label, a list of int 
    ------------
    -- Output --
    ------------
        model
            only available when not in grid search train

    """
    if grid_search:

        # create model
        model = BertClassifier()
        paramAssign(model, params)

        # set grid search
        clf = GridSearchCV(
                model,
                grid_params,
                scoring = 'accuracy',
                verbose = 5,
                cv = 3,
                return_train_score = True)

        # fine tuning on our data
        clf.fit(x, y)

        # statistics
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        
        # open file for ouput message
        with open("../outcome/result_2.txt", "a") as f:
            
            # empty the file
            print("\n", file=f)

            # print result
            for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std*2, params), file=f)

            # best scores
            print("\nBest score:", clf.best_score_, "with params:", clf.best_params_, file=f)

        return model

    else:
        
        # create model
        model = BertClassifier()
        paramAssign(model, params)

        # fine tune model
        model.fit(x, y)

        return model 

if __name__ == '__main__':
    
    # set start time
    start = time.time()

    # training options
    grid_search = False

    # env variable
    path = "../../../data/train/"                                     # data path 
    saveFile = "../modelFile/model.bin"                         # model file path
    keyWords = ["smoke", "smoker", "smokes", "smoking", "smoked", "tobacco"]
    
    params = {                                                  # the parameters needed for normal model
            "bert_model": "bert-base-cased",
            "train_batch_size": 5,
            "eval_batch_size": 1,
            "max_seq_length": 128,
            "num_mlp_layers": 1,
            "num_mlp_hiddens": 20,
            "epochs": 6,
            "learning_rate": 1e-5,
            "use_cuda": True}
    grid_params = {                                             # the parameters will be change during grid search 
            "num_mlp_layers": [1, 2, 3],
            "num_mlp_hiddens": [15, 20, 30],
            "train_batch_size": [3, 4, 5],
            "epochs": [6, 9, 12],
            "learning_rate": [1e-4, 1e-5, 1e-6]}
    grid_params_const = {                                       # the parameter will not be changed during grid search
            "bert_model": "bert-base-cased",
            "eval_batch_size": 1,
            "use_cuda": True,
            "max_seq_length": 128,
            "validation_fraction": 0.2}

    # data pre-process 
    data = dataLoader(path, keyWords)
    
    # remove data with empty string
    for idx in range(len(data["label"])-1, -1, -1):
        if data["sent"][idx] == "":
            for key in data.keys():
                del data[key][idx]
    print("Total number of training data:", len(data["sent"]))
    
    # set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    
    # training with the grid search
    model = train(grid_search, grid_params_const if grid_search else params, grid_params, data["sent"], data["label"])

    # test and evaluation
    if not grid_search:
        
        # save model
        model.save(saveFile)

    # set end time
    end = time.time()
    print("Total elapse time:", end-start, "s")
