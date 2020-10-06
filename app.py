from flask import  Flask,render_template,url_for,request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import re
import string


# removing the html tags
def clean_html(text):
    clean=re.compile('<.*?>')
    cleantext=re.sub(clean,'',text)
    return cleantext    
# first round of cleaning
def clean_text1(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    return text
# second round of cleaning
def clean_text2(text):
    text=re.sub('[''"",,,]','',text)
    text=re.sub('\n','',text)
    return text



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
        txt=request.form['message']
        txt=clean_html(txt)
        txt=clean_text1(txt)
        txt=clean_text2(txt)
        txt=[txt]
        txt=cv.transform(txt)
        mypred=clf.predict(txt)
    
    return render_template('result.html',prediction=mypred[0])



if __name__=='__main__':
    app.run(debug=True)