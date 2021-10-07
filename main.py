from flask import Flask
from flask.globals import request
from flask.templating import render_template  
import librosa
import numpy as np
import pickle

filename=("models/emotionmodel.pkl")
s=pickle.load(open(filename,'rb'))

values = {"fearful": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/59-596556_fear-clipart-fear-emotion-cartoon-face-of-fear-removebg-preview.png?alt=media&token=c451d9cc-5fa7-47e1-bfa6-4be37e358ea8",
          "calm": "https://d29fhpw069ctt2.cloudfront.net/clipart/100203/preview/smiling_face_of_a_child_2_preview_9c89.png",
          "happy": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/565-5650281_happy-boy-clipart-can-do-it-png-transparent-removebg-preview.png?alt=media&token=5964f656-e102-4f85-bb5c-ad4209209e39",
          "sad": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/202-2022552_emotional-clipart-sad-dad-sad-clip-art-removebg.png?alt=media&token=3f1938a7-790e-4923-aea7-7f81ee2807b9",
          "angry": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/clipart466731.png?alt=media&token=8dd82f61-b3ef-46f2-86c7-e1cd61f24ff3"
          }

def extract_feature(file_name, mfcc, chroma):
    X,sample_rate=librosa.load(file_name)
    if chroma:
        stft=np.abs(librosa.stft(X))
        result=np.array([])
    if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
    if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
    return result

app = Flask(__name__)  

@app.route('/')
def index():
    result=False
    return render_template('inputfile.html',result1=result)

@app.route('/',methods=["post"]) 
def home():
        r=[0,0]
        result=True
        audio=request.files.get('shashifile')
        feature=extract_feature(audio, mfcc=True, chroma=True)
        p=s.predict([feature])
        # r="https://d29fhpw069ctt2.cloudfront.net/clipart/100203/preview/smiling_face_of_a_child_2_preview_9c89.png"
        r=[p[0],values[p[0]]]
        return render_template('inputfile.html',result1=result,r1=r)

  
if __name__ =='__main__':  
    app.run(debug = True)  