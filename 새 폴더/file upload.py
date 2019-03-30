#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
from werkzeug import secure_filename
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json



# In[3]:


app=Flask(__name__)
ALLOWED_EXTENSIONS=set(['png'])


# In[4]:


@app.route('/upload')
def load_file():
    return render_template('upload.html')


# In[ ]:

# In[5]:


@app.route('/uploader',methods=['GET','POST'])
def upload_file():
    if request.method=='POST':
        f=request.files['file']
        name=f.filename+str(time.time())
        f.save(secure_filename(name))
        json_file = open("model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")

        
        image=cv2.imread(name,cv2.IMREAD_GRAYSCALE)
        image=image.reshape(-1,28,28,1)
        image=image.astype('float32')/255.0

        pred=loaded_model.predict(image)
        pred=np.argmax(pred,axis=1)

        
        return str(pred[0])


# In[6]:


if __name__=='__main__':
    app.run(debug=True)


# In[ ]:




