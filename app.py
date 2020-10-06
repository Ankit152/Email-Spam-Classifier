from flask import  Flask,render_template,url_for,request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

# loading the model from the disk
filename='model.pkl'
clf=pickle.load(open(filename,'rb'))
cv=CountVectorizer()
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        txt=[message]
        txt=cv.transform(txt).toarray()
        mypred=clf.predict(txt)
    
    return render_template('result.html',prediction=mypred)



if __name__=='__main__':
    app.run(debug=True)