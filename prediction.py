import  pandas as pd
import My_ML_Lib as mml
import pickle


if __name__=='__main__':
    cv = pickle.load(open("vocab.pkl", "rb"))
    best_grid= mml.load_model('model/svm 82.76374442793461.pkl')
    xx = ['''World Series G7: Giants vs. Royals'''.lower()]
    X_data = cv.transform(xx)
    x = best_grid.predict(X_data)
    print(x)
    if x == 0:
        print('adult')
    elif x == 1:
        print('animation')
    elif x == 2:
        print('education')
    elif x == 3:
        print('kid')
    elif x == 4:
        print('sports')