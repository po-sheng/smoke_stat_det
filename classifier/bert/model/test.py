from bert_sklearn import load_model
from data_preprocess import dataLoader
import re

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

# re-order the data
ans = []
zipped = list(zip(pred_idx, pred_label, data["name"]))
for i in range(len(zipped)):
    for idx in range(len(zipped)):
        if int(re.split(r"[._]", zipped[idx][2])[1]) == i+1:
            ans.append(zipped[idx])

# save and print output
print(ans)
f = open("../outcome/prediction.txt", "w")
for i in range(len(ans)):
    f.write(ans[i][1]+"\n")
f.close
