from flask import  Flask,render_template,url_for,request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import re
import string

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # loading the model from the disk
    filename='model.pkl'
    cvfile='cv.pkl'
    clf=joblib.load(filename)
    cv=joblib.load(cvfile)
    if request.method=='POST':
        txt=request.form['message']
        clean=re.compile('<.*?>')
        txt=re.sub(clean,'',txt)
        txt=txt.lower()
        txt=re.sub('\[.*?\]','',txt)
        txt=re.sub('[%s]'%re.escape(string.punctuation),'',txt)
        txt=re.sub('\w*\d\w*','',txt)
        txt=re.sub('[''"",,,]','',txt)
        txt=re.sub('\n','',txt)
        txt=[txt]
        txt=cv.transform(txt).toarray()
        mypred=clf.predict(txt)
    
    return render_template('result.html',prediction=mypred)



if __name__=='__main__':
    app.run(debug=True)