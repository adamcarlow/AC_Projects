#!/usr/bin/env python
# coding: utf-8

# In[2]:


from fastai2.vision.all import *
from fastai2.vision.widgets import *


# 
# The Super Hero Classifier!
# 
# Is that a Super Hero, now you can quickly and easily find out. Take a pic of the person, mutant, robot, alien etc., and click 'upload' to classify it. (Important: this only handles limited super heroes from the Marvel Universe. It will not give a sensible answer for any DC characters.

# In[3]:


path = Path()
learn_inf = load_learner(path/'export.pkl', cpu=True)
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()


# In[4]:


def on_click(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'


# In[5]:


btn_upload.observe(on_click, names=['data'])


# In[6]:


display(VBox([widgets.Label('Select your Hero!'), btn_upload, out_pl, lbl_pred]))


# In[1]:


#hide
get_ipython().system('pip install voila')
get_ipython().system('jupyter serverextension enable voila --sys-prefix')


# In[ ]:




