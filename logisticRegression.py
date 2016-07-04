import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

trainFile = "cnnData2.csv"
testFile = "usTweets.csv"

trainData = pd.read_csv(trainFile, header=0, delimiter="\t")
testData = pd.read_csv(testFile, header=0, names=['tweet_id', 'created_at', 'tweet_text'], delimiter=",")

train_words = []
test_words = []

def filtertext(w):
    # if w is not None or w is not 'nan':
    if isinstance(w, str):
        words = re.sub("[^a-zA-Z]", " ", w)
        words = re.sub(" +", ' ', words)
        return words
    else:
        return None


print("Filtering")
for art in trainData["body"]:
    tw = filtertext(art)
    if tw is not None:
        train_words.append(tw)
    else:
        train_words.append("")

for t in testData["tweet_text"]:
    tw = filtertext(t)
    if tw is not None:
        test_words.append(tw)
    else:
        test_words.append("")

print("tfidf_vectorizer")
tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=0, ngram_range=(1, 3), max_features=3000000)
tfidf_Train_matrix = tfidf_vectorizer.fit_transform(train_words)
tfidf_Test_matrix = tfidf_vectorizer.transform(test_words)

print("LogisticRegression")
LogisticRegression()
logReg = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=10000, penalty="l2", class_weight="balanced")  # n_jobs=-1

print("Training")
logReg.fit(tfidf_Train_matrix, trainData["category"])

print("Predicting")
pred = logReg.predict_proba(tfidf_Test_matrix)  # [:, 1]

print("Writing results to file")
output = pd.DataFrame(data=pred, index=testData["tweet_id"], columns=logReg.classes_)
output.to_csv("predicted_results3000000.csv", index=True, index_label="tweet_id", quoting=3, escapechar="/")
