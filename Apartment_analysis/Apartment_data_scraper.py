# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 19:44:03 2023

@author: ASUS
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd

url= "https://www.apamanshop.com/tokyo/104/"

building_type_list=[]
building_name=[]
floor=[]
transportation=[]
building_age=[]
building_materials=[]
address=[]
span_list=[]
a=[]
room_floor=[]
rent_fee=[]
management_fee=[]
deposit=[]
gift_money=[]
house_plan=[]
house_area=[]
def scrape_page(url):
    pages=1
    while pages!=25:
        url=f"https://www.apamanshop.com/tokyo/104/?page={pages}"
        page = requests.get(url)
    
        soup = BeautifulSoup(page.content, 'lxml')
        span_list,a=get_data(soup)
        pages = pages + 1
    
    get_info(span_list, a)




def get_data(soup):
    lists = soup.find_all('article', class_='mod_box_section_bdt')
    for list in lists:
         building_type_list.append (list.find('p', class_="txt_type").text.replace('\n', ''))
         building_name.append(list.find('h2', class_='name').text.replace('\n',''))
         #floor.append(list.find('p', class_='info').span.text)
         span_list.append(list.find('p', class_='info').text.replace('\n','').replace('\r','').split(' /'))
         address.append(list.find('p', class_='address').text.replace('\n', ''))
         transportation.append(list.find('ul', class_='list_info').text.replace('\n', ' '))
         a.append(list.find('tr', class_='tr_mid').text.replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ',' ').strip().split(' '))
    return span_list,a
def get_info(span_list,a):    
    for i in range(len(span_list)):
            
        floor.append(span_list[i][0])
        building_age.append(span_list[i][1])
        building_materials.append(span_list[i][2])
            
           
    for i in range(len(span_list)):
       room_floor.append(a[i][0])
       rent_fee.append(a[i][1])
       management_fee.append(a[i][2])
       deposit.append(a[i][3])
       gift_money.append(a[i][4])
       house_plan.append(a[i][5])
       house_area.append(a[i][6])



def to_csv():
    data={
         'building_name': building_name,
         'building_type': building_type_list,
         'address': address,
         'floor': floor,
         'transport': transportation,
         'building_age': building_age,
         'building_materials': building_materials,
         'room_floor': room_floor,
         'rent_fee': rent_fee,
         'management_fee': management_fee,
         'deposit': deposit,
         'gift_money': gift_money,
         'house_plan': house_plan,
         'house_area': house_area
            
         }
    df= pd.DataFrame(data)
    
    df.to_csv('Apartment_Raw_Data.csv', index=None, encoding='utf-8-sig')


def main():
    scrape_page(url)
    to_csv()

if __name__ == "__main__":
    main()     

    # location = list.find('div', class_="listing-search-item__location").text.replace('\n', '')
     #price = list.find('span', class_="listing-search-item__price").text.replace('\n', '')
     #area = list.find('span', class_="illustrated-features__description").text.replace('\n', '')
        
 
     