import csv
import operator
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

newsFile = "cnnData2.csv"
tweetFile = "usTweets.csv"
predFile = 'predicted_results3000000.csv'
features = 3000000


def filtertext(w):
    # if w is not None or w is not 'nan':
    if isinstance(w, str):
        words = re.sub("[^a-zA-Z]", " ", w)
        words = re.sub(" +", ' ', words)
        return words
    else:
        return None


def analyzeLogRegNews():
    csvdata = pd.read_csv(newsFile, header=0, delimiter="\t")

    foldval = 10
    train_words = []

    print("Filtering")
    for art in csvdata["body"]:
        tw = filtertext(art)
        if tw is not None:
            train_words.append(tw)
        else:
            train_words.append("")

    kf = KFold(len(train_words), n_folds=foldval, shuffle=True)

    print("tfidf_vectorizer")
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=0, ngram_range=(1, 3), max_features=features)
    tfidf_train_matrix = tfidf_vectorizer.fit_transform(train_words)

    print("LogisticRegression")
    LogisticRegression()
    logreg = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=10000, penalty="l2", class_weight="balanced")  # n_jobs=-1
    logreg.fit(tfidf_train_matrix, csvdata["category"])

    classes = {}
    c = 0
    for cla in logreg.classes_:
        classes[cla] = c
        c += 1

    microF1 = macroF1 = avg_accuracy = avg_precision = 0
    for train_ind, test_ind in kf:
        test_words = [train_words[ti] for ti in test_ind]

        print("Predicting")
        tfidf_test_matrix = tfidf_vectorizer.transform(test_words)
        pred = logreg.predict_proba(tfidf_test_matrix)

        c = 0
        y_pred = []
        y_true = []
        for rlist in pred:
            index = test_ind[c]
            y_true.append(classes[csvdata["category"][index]])
            y_pred.append(max(enumerate(rlist), key=operator.itemgetter(1))[0])
            c += 1

        if len(y_pred) != 0 and len(y_true) != 0 and len(y_pred) == len(y_true):
            acc = accuracy_score(y_true, y_pred)
            avg_accuracy += acc

            pre = precision_score(y_true, y_pred, average='micro')
            avg_precision += pre

            maf1 = f1_score(y_true, y_pred, average='macro')
            macroF1 += maf1

            mif1 = f1_score(y_true, y_pred, average='micro')
            microF1 += mif1

            print("Accuracy: {0:.5f}".format(acc) + " Precision: {0:.5f}".format(pre) +
                  "\t Macro F1: {0:.5f}".format(maf1) + "\tMicro F1: {0:.5f}".format(mif1))

    if avg_accuracy != 0:
        print("Final average Accuracy: {0:.5f}".format(avg_accuracy/foldval))
    if avg_precision != 0:
        print("Final average Precision: {0:.5f}".format(avg_precision / foldval))
    if microF1 != 0:
        print("Final average Micro F1: {0:.5f}".format(microF1 / foldval))
    if macroF1 != 0:
        print("Final average Macro F1: {0:.5f}".format(macroF1 / foldval))
    print()


if __name__ == "__main__":
    print("Loading Twitter data")
    tweetData = {}
    with open(tweetFile) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)
        for row in csvreader:
            try:
                if len(row) > 0 and row[0].isdigit():
                    tweetData[row[0]] = row[1:]
            except IndexError:
                continue

    print("Loading Predicted Tweets classifiers")
    predtweets = {}
    with open(predFile) as csvfile:
        csvreader2 = csv.reader(csvfile, delimiter=',')
        predheaders = []
        for row in csvreader2:
            if len(predheaders) == 0:
                predheaders = row[1:]
            elif len(row) > 0 and row[0].isdigit():
                data = {}
                c = 0
                for r in row[1:]:
                    try:
                        r = float(r)
                    except ValueError:
                        r = 0
                    data[predheaders[c]] = r
                    c += 1
                predtweets[row[0]] = sorted(data.items(), key=operator.itemgetter(1), reverse=True)

    print("Loading News Data")
    # news headers: id	headline	link	body	category
    newsDataByCategory = {}
    with open(newsFile) as csvfile:
        csvreader3 = csv.reader(csvfile, delimiter="\t")
        next(csvreader3, None)
        for row in csvreader3:
            if len(row) == 5:
                if row[4] not in newsDataByCategory:
                    newsDataByCategory[row[4]] = []
                newsDataByCategory[row[4]].append(row[:4])

    inp = ""
    while inp.lower() != "exit" and inp.lower() != "4":
        inp = input("\nEnter 1 to view Tweets with Predicted Categories\n"
                    "Enter 2 to analyze Logistic Regression on News Articles\n"
                    "Enter 3 to analyze Logistic Regression on Tweets\n"
                    "Enter 'export' to export Tweet body and top predicted categories'\n"
                    "Enter 4 or 'exit' to exit:\n"
                    "Input: ")

        if inp == "1":
            tweetid = list(predtweets.keys())
            c = 0
            while True:
                print("Type 'next' for next page, 'prev' for previous page, 'exit' for menu screen")
                print("Type number to view articles of same category")
                displaycat = []
                for i in range(0, 10):
                    tid = tweetid[c]
                    outstr = str(i) + ":\t" + tid + "\t" + tweetData[tid][1] + "\n\tCat:"
                    highrat = predtweets[tid][0][0]
                    for x in range(0, 6):
                        outstr += " " + predtweets[tid][x][0] + " {0:.5f}".format(predtweets[tid][x][1]) + ", "
                    print(outstr + "\n")
                    displaycat.append((outstr, highrat))
                    c += 1

                inpval1 = input(":")

                if inpval1.isdigit() and 0 <= int(inpval1) <= 9:
                    print()
                    print(displaycat[int(inpval1)][0])
                    cat = displaycat[int(inpval1)][1]
                    print("News articles about " + cat)
                    for x in range(0, 10):
                        if len(newsDataByCategory[cat]) <= x:
                            print("No more news articles")
                            break
                        print("Headline: " + newsDataByCategory[cat][x][1] + "\n\tLink: " + newsDataByCategory[cat][x][2])
                    print()
                    input("Press any key to continue: ")
                elif inpval1.lower() == 'exit':
                    break
                elif inpval1.lower() == 'next':
                    continue
                elif inpval1.lower() == 'prev':
                    if c < 20:
                        c = 0
                    else:
                        c -= 20
                else:
                    if c < 10:
                        c = 0
                    else:
                        c -= 10

        elif inp == "2":
            features = input("Enter the number of features: (default is 3000000)\n")
            if not features.isdigit():
                features = 3000000
            else:
                features = int(features)
            analyzeLogRegNews()
        elif inp == "3":
            print("\nAnalysis of Logistic Regression on Tweets")
            print("Second column is the number of tweets that had that category as the highest probability\n")
            print("Category\t\tTop Predicted Count\t\tAmount Analyzed\t\tCorrect Categorization")
            print("------------------------------------------------------------------------------------")
            print("world\t\t\t\t\t11\t\t\t\t\t11\t\t\t\t\t\t1")
            print("us\t\t\t\t\t\t53\t\t\t\t\t53\t\t\t\t\t\t5")
            print("travel\t\t\t\t\t11\t\t\t\t\t11\t\t\t\t\t\t4")
            print("tennnis\t\t\t\t\t1\t\t\t\t\t1\t\t\t\t\t\t1")
            print("sport\t\t\t\t\t35\t\t\t\t\t35\t\t\t\t\t\t11")
            print("politics\t\t\t\t6939\t\t\t\t100\t\t\t\t\t\t53")
            print("opinions\t\t\t\t12\t\t\t\t\t12\t\t\t\t\t\t0")
            print("motorsport\t\t\t\t1\t\t\t\t\t1\t\t\t\t\t\t0")
            print("middleeast\t\t\t\t4\t\t\t\t\t4\t\t\t\t\t\t1")
            print("luxury\t\t\t\t\t10\t\t\t\t\t10\t\t\t\t\t\t0")
            print("living\t\t\t\t\t97\t\t\t\t\t78\t\t\t\t\t\t7")
            print("health\t\t\t\t\t49\t\t\t\t\t49\t\t\t\t\t\t10")
            print("golf\t\t\t\t\t6\t\t\t\t\t6\t\t\t\t\t\t6")
            print("football\t\t\t\t38\t\t\t\t\t38\t\t\t\t\t\t22")
            print("foodanddrink\t\t\t4\t\t\t\t\t4\t\t\t\t\t\t4")
            print("fashion\t\t\t\t\t11\t\t\t\t\t11\t\t\t\t\t\t9")
            print("europe\t\t\t\t\t15\t\t\t\t\t15\t\t\t\t\t\t2")
            print("entertainment\t\t\t42619\t\t\t\t100\t\t\t\t\t\t62")
            print("architecture\t\t\t2\t\t\t\t\t2\t\t\t\t\t\t2")
            print("africa\t\t\t\t\t13\t\t\t\t\t13\t\t\t\t\t\t6")
            print("cnn-info\t\t\t\t2\t\t\t\t\t2\t\t\t\t\t\t0")
            print("business\t\t\t\t3\t\t\t\t\t3\t\t\t\t\t\t1")
            print("aviation\t\t\t\t11\t\t\t\t\t11\t\t\t\t\t\t9")
            print("auto\t\t\t\t\t5\t\t\t\t\t5\t\t\t\t\t\t2")
            print("asia\t\t\t\t\t14\t\t\t\t\t14\t\t\t\t\t\t7")
            print("arts\t\t\t\t\t12\t\t\t\t\t12\t\t\t\t\t\t7")
            print("americas\t\t\t\t22\t\t\t\t\t22\t\t\t\t\t\t1")
            print("------------------------------------------------------------------------------------")
            print("Total:\t\t\t\t\t50000\t\t\t\t623\t\t\t\t\t\t233\n")
            print("Number of tweets: 50000")
            print("Number of tweets manually examined: 623")
            print("Number of tweets correctly categorized: 233\n")
        elif inp == "export":
            tweetid = list(predtweets.keys())
            c = 0
            with open('pred_with_msg.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for tid in tweetid:
                    if tid in tweetData and len(tweetData[tid]) >= 2:
                        outlist = [tid, tweetData[tid][1]]
                        for x in range(0, 5):
                            outlist.append(predtweets[tid][x][0] + " {0:.5f}".format(predtweets[tid][x][1]))
                        spamwriter.writerow(outlist)
