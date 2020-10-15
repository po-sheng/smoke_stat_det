from bert_sklearn import load_model
from data_preprocess import dataLoader

path = "../../../data/test/"                       # data path   
modelFile = "../modelFile/model.bin"        # model file path
keyWords = ["smoke", "smoker", "smokes", "smoking", "smoked", "tobacco"]

label2idx = {"CURRENT": 0, "NON": 1, "PAST": 2, "UNKNOWN": 3}
pred_idx = []
pred_label = []

# data pre-process
data = dataLoader(path, keyWords)
print("Total number of training data:", len(data["sent"]))

# load model
model = load_model(modelFile)

# do each data singly
for idx in range(len(data["sent"])):
    if data["sent"][idx] == "":
        # which predict as unknown type
        pred_label.append("UNKNOWN")
        pred_idx.append(3)

    else: 
        # make prediction
        y_pred = model.predict([data["sent"][idx]])

        # make probability prediction
        y_prob = model.predict_proba([data["sent"][idx]])

        pred_label.append(y_pred[0])

        # convert label to index
        pred_idx.append(label2idx[y_pred[0]])

print(pred_idx)
print(pred_label)
print(data["name"])
