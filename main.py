import numpy as np
import pandas as pd
import  json

from flask import  Flask , jsonify
from flask import request

from pymongo import  MongoClient






from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def getDatabase():

    connectionString = "mongodb+srv://Restro-sentiment:yiZnSt5OYbOTULcU@cluster0.pwozgdd.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(connectionString)

    return  client['Restro-sentiment']

dbName = getDatabase()



def reviewPrediction (sentimentArray):
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSTHTr82eiIkmfDK91v_jaSksgwm46Mbow8XOGb8fEHCUh4mvis9SN8JAX67jkpaawjsIGv8JPOhvEc/pub?output=csv')
    X = df.iloc[ : , 1]
    y = df.iloc[ : , -1 ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)
    v = CountVectorizer()

    x_train_vec = v.fit_transform(X_train)
    x_test_vec = v.transform(X_test)

    regression = LogisticRegression()
    parameter = {'penalty': ['l1', 'l2', 'elasticnet'], 'C': [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 100],  'max_iter': [100, 200, 300, 400]}
    classifier_regressor = GridSearchCV(regression, param_grid=parameter, scoring='accuracy', cv=5)
    classifier_regressor.fit(x_train_vec, y_train)
    y_pred = classifier_regressor.predict(x_test_vec)
    score = accuracy_score(y_pred, y_test)

    rev = sentimentArray
    rev_vec = v.transform(rev)

    predictionArray = classifier_regressor.predict(rev_vec)

    return  predictionArray




app = Flask(__name__)
@app.route('/')
def helloWorld():
    return 'Hello World'


@app.route('/reviewAnalysis/<string:id>')
def reviewAnalysis(id: str):
    reviewItem = []
    collectionName = dbName["review"]
    item_details = collectionName.find({"RestraurentId": id})
    print(item_details)
    for item in item_details:
        reviewItem.append(item)


    reviewList = []

    for item in reviewItem:
        reviewList.append(item['review'])


    predictonArray = reviewPrediction(reviewList)
    predictArrayLength = len(predictonArray)
    satisfiedCase = []


    for i in predictonArray:
        if i == 'Satisfied':
            satisfiedCase.append(i)
    satisfiedCaselenth = len(satisfiedCase)


    satisfactionRatio = int((satisfiedCaselenth/predictArrayLength)*100)

    if len(reviewList)==0:
        satisfactionRatio = 0

    sendingData = { "Satisfaction" : satisfactionRatio}



    return jsonify(sendingData)

if __name__ ==  '__main__':
    app.run(debug=True)




