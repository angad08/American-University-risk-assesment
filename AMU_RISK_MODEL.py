#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle as pk
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler as sc
pickled_model = pk.load(open("C:\\Users\\Angat\\Downloads\\AMU_RISK\\model.pkl", "rb"))
df=pd.read_csv("C:\\Users\\Angat\\Downloads\\AMU_RISK\\preprocessed_df.csv")
X=df.drop(["At_Risk","ANON_INSTR_ID","TERM","SEX"],axis=1)
Y=df["At_Risk"]
std=sc().fit(X)
std.transform(X)


# In[2]:


# 'GRD_PTS_PER_UNIT':0-5
# "GPAO":1-4
# "HSGPA":0-4


# In[4]:


#2.0,2.945000,3.7,F
st.title("American University TERM Status assesment")
a=st.number_input("GRADE POINTS OBTAINED PER UNIT",min_value=X["GRD_PTS_PER_UNIT"].min(),max_value=X["GRD_PTS_PER_UNIT"].max())
b=st.number_input("GPAO",min_value=X["GPAO"].min(),max_value=X["GPAO"].max())
c=st.number_input("HSGPA",min_value=X["HSGPA"].min(),max_value=X["HSGPA"].max())
d = st.radio(
    "GENDER",
    ('MALE','FEMALE'))

if d == 'MALE':
    d=1
else:
    d=0
new_array = np.append(std.transform(pd.DataFrame([[a,b,c]])),d).reshape(1,-1)
pickled_model.predict(new_array)
#new_array
st.write("")
if st.button('Check Risk'):
    if pickled_model.predict(new_array)[0]==1:
        txt="<p style='font-family:Courier; text-align:center; font-weight:bold; color:Red; font-size: 30px;'>You are at Risk Of Failure</p>"
        st.write(txt,unsafe_allow_html=True)
    else:
        txt="<p style='font-family:Verdana; text-align:center; font-weight:bold; color:Green; font-size: 30px;'>You are Safe!!</p>"
        st.write(txt,unsafe_allow_html=True)


