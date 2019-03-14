
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import pandas as pd
import pickle

if __name__=='__main__':
    data = pd.read_csv('data/youtube1.csv', encoding='latin-1')
    # print(data.head())

    x = data['video_name']
    y = data['type']
    # create lebel for y
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    Y = encoder.fit_transform(y.astype(str))
    print(encoder.classes_)

    # Handle Input text data
    from sklearn.feature_extraction.text import TfidfVectorizer

    cv = TfidfVectorizer(min_df=1, stop_words='english')
    X = cv.fit_transform(x).toarray()

    with open("vocab.pkl", 'wb') as handle:
        pickle.dump(cv, handle)

    from sklearn.metrics import accuracy_score
    from sklearn.utils import shuffle

    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    lin = LinearSVC(loss='hinge',C=10)
    s=[]
    for i in range(0,250):
        t_x,t_y=shuffle(X,Y)

        X_train, X_test, Y_train, Y_test = train_test_split(t_x, t_y, test_size = 0.25, random_state = 0)
        lin.fit(X_train,Y_train)
        sc=lin.score(X_test, Y_test)
        s.append(sc*100)
        print(i)

        import matplotlib.pyplot as plt

        plt.clf()
        plt.plot(s)
        plt.title("Accuracy curve ")
        plt.xlabel("Accuracy")
        plt.xlabel("Iteration")
        plt.savefig('{}.jpg'.format('Accuracy'))

        file=open("summary.txt",'a')
        file.write("Mean -> {} \nMax -> {} \nMin-> {} total->{}".format(np.mean(s),max(s),min(s),250))