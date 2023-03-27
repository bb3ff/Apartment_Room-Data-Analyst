# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 18:58:36 2023

@author: ASUS
"""

import pandas as pd
import numpy as np
import re
df= pd.read_csv('Apartment_Raw_Data.csv')

df['house_area_int']=df['house_area'].apply(lambda x: float(x.split('m²')[0]))

df['have_living_room']=df['house_plan'].apply(lambda x: 1 if 'l' in x.lower() else 0)

df['have_kitchen_room']=df['house_plan'].apply(lambda x: 1 if 'k' in x.lower() else 0)

df['have_dining_room']=df['house_plan'].apply(lambda x: 1 if 'd' in x.lower() else 0)

df['have_storage_room']=df['house_plan'].apply(lambda x: 1 if 's' in x.lower() else 0)

df['have_common_room']=df['house_plan'].apply(lambda x:1 if 'r' in x.lower() else 0)

gift_money=df['gift_money'].apply(lambda x: x.lower().replace('－', 'None') )
df['have_gift_money']=gift_money.apply(lambda x: 0 if 'none' in x.lower() else 1)

df['have_deposit']=df.deposit.apply(lambda x: 0 if '－' in x.lower() else 1)

management_fee= df.management_fee.apply(lambda x: x.lower().replace('－','-1').replace(',', '.'))


df['management_fee_int']=management_fee.apply(lambda x: float(x.replace('円', '')))

rent_fee= df.rent_fee.apply(lambda x: x.replace('万円', ''))
df['rent_fee_int']=rent_fee.apply(lambda x: float(x)*10)

df['building_age_int']=df.building_age.apply(lambda x: int(x.replace('築', '').replace('年', '')))

def split_transport1(transport):

   transport_list1=[]
   transport_list2=[]
   transport_list3=[]
   re_lists=[]
   
   for i in range(len(transport)):
       
       re_list=re.finditer('分', transport[i])


       for res in re_list :
           re_lists.append(res.end()) 

      
       trans1=transport[i][:re_lists[0]]
       transport_list1.append(trans1)
       
       if len(re_lists)>1:
           trans2=transport[i][re_lists[0]+1:re_lists[1]]
           transport_list2.append(trans2)
       else:
           transport_list2.append('')
       
       if len(re_lists)>2:
           trans3=transport[i][re_lists[1]+1:re_lists[2]]
           transport_list3.append(trans3)
       else:
           transport_list3.append('')
       re_lists=[]
       
   return transport_list1, transport_list2, transport_list3

df['transport1'], df['transport2'], df['transport3']=split_transport1(df['transport'])

df['have_convinient_transport']=df['transport'].apply(lambda x: 1 if 'Ｒ山手線 ' in x or'都営大江戸線' in x else 0)

df['building_floor_int']=df['floor'].apply(lambda x: int(x.replace('階建て','')))

df.room_floor.value_counts()
df=df.drop(df.loc[df.room_floor=='B1階'].index)

df['room_floor_int']=df['room_floor'].apply(lambda x: int(x.replace('階','')))
df_out=df.drop_duplicates(subset='building_name', keep="last")

df_out.to_csv('Apartment_cleaned_data.csv', index=False, encoding='utf-8-sig')
    