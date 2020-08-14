# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 01:31:08 2020

@author: KaleemUllah
"""


import os
import glob
import pandas as pd
frame = pd.DataFrame()
df=pd.DataFrame()
basepath='C:/Users/KaleemUllah/Downloads/raw_data_contralateral_tDCS/raw_data_contralateral_tDCS/anodal'

for subject  in os.listdir(basepath):
    for day in os.listdir(os.path.join(basepath,subject)):
        for trail in os.listdir(os.path.join(basepath,subject,day)):
            for file in os.listdir(os.path.join(basepath,subject,day,trail)):
                if( trail[:-2]=='neurofeedback' and (int(trail[-2:])<=5 or int(trail[-2:])>=11)):
                    print(subject+' '+day+' '+trail+' '+file)
                    df= pd.read_csv(os.path.join(basepath,subject,day,trail,file),sep='\t', header=None)
                    lenth=len(df)
                    df=df.T
                    df['subject']=subject
                    df['day']=day
                    df['trail']=trail
                    df['length']=lenth
                    if (int(trail[-2:])<=5):
                        df['Lable']=0
                    else: 
                        df['Lable']=1
                    cols = list(df.columns)
                    cols = cols[-5:] + cols[:-5]
                    df = df[cols]
                    frame=pd.concat([frame, df])
                  
        
  
#%%
        
frame.to_csv('anodal_EEG.csv')                 