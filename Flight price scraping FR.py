#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required library
import pandas as pd
import time
import selenium
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# In[2]:


# loading chrome by automated
dr1=webdriver.Chrome(r'F:\Python\chromedriver.exe')


# In[3]:


# load wensite
url='https://www.yatra.com/'
dr1.get(url)
dr1.maximize_window()


# In[4]:


# save list of elements
name=[]
deprt=[]
arrvt=[]
sorc=[]
dest=[]
durt=[]
stops=[]
price=[]


# In[5]:


# scraping function
def scrap():
    for i in  dr1.find_elements_by_xpath("//div[@class='fs-13 airline-name no-pad col-8']/span[1]"):
        name.append(i.text)
    for i in dr1.find_elements_by_xpath("//div[@class='fs-15 bold time']/div"):
        deprt.append(i.text)
    for i in dr1.find_elements_by_xpath("//div[@class='i-b pdd-0 text-left atime col-5']/p[1]"):
        arrvt.append(i.text)
    for i in dr1.find_elements_by_xpath("//div[@class='i-b col-4 no-wrap text-right dtime col-3']/p"):
        sorc.append(i.text)
    for i in dr1.find_elements_by_xpath("//div[@class='i-b pdd-0 text-left atime col-5']/p[2]"):
        dest.append(i.text)
    for i in dr1.find_elements_by_xpath("//div[@class='stop-cont pl-13']/p"):
        durt.append(i.text)
    for i in dr1.find_elements_by_xpath("//div[@class=' font-lightgrey fs-10 tipsy i-b fs-10']/span"):
        stops.append(i.text)
    for i in dr1.find_elements_by_xpath("//div[@class='i-b tipsy fare-summary-tooltip fs-18']"):
        price.append(i.text)
    
    return    


# In[6]:


# scrap data from first page
dr1.find_element_by_id('BE_flight_flsearch_btn').click()
time.sleep(1)
for _ in range(10):
    time.sleep(1)
    dr1.execute_script("window.scrollBy(0,7000)")
    time.sleep(3)
    scrap()


# In[128]:


#making list of locations
city_lst = ['surat','Mumbai','Delhi','kolkata','Bangalore','Goa','Ahmedabad','Vadodara','Udaipur','Pune','Kochi','Hyderabad','Bagdogra','Jammu']


# In[122]:


for x in city_lst:
    for y in city_lst:
        if x!=y:
        
            source = dr1.find_element_by_xpath("//div[@class='input-holder pb-2 bdr-btm']/input") # given source city input
            time.sleep(1)
            source.send_keys(Keys.CONTROL,'a')
            source.send_keys(Keys.BACKSPACE)
            time.sleep(1)
            source.send_keys(x)
            time.sleep(1)
            
            destination= dr1.find_element_by_xpath("//div[@class='input-holder  bdr-btm pb-2']/input") # give destination city input
            time.sleep(1)
            destination.send_keys(Keys.CLEAR,'a')
            destination.send_keys(Keys.BACKSPACE)
            time.sleep(1)
            destination.send_keys(y)
            time.sleep(1)
            
            try:
                dr1.find_element_by_xpath("/html/body/section[2]/section/div[2]/i").click() # click button to return top of page 
            except NoSuchElementException:
                pass
            try:
                time.sleep(.5)
                srch_btn = dr1.find_element_by_xpath("//button[@class='fs-14 btn-submit cursor-pointer bold']") # click search button fpr 
                wait = WebDriverWait(dr1, 10)
                wait.until(EC.visibility_of(srch_btn))
                srch_btn.click()
            except NoSuchElementException:
                pass
            for _ in range(5):
                time.sleep(2)
                dr1.execute_script("window.scrollBy(0,3000)")
                time.sleep(3)
                scrap()


# In[123]:


print(len(name),len(deprt),len(arrvt),len(sorc),len(dest),len(stops),len(durt),len(price))


# In[129]:


#pd.set_option('display.max_rows',2000)
df=pd.DataFrame({'name':name,'deprt':deprt,'arrvt':arrvt,'sorct':sorc,'dest':dest,'stops':stops,'durat':durt,'price':price})


# In[131]:


df.head()


# In[132]:


# save csv file
df.to_csv('Flightprice.csv')


# In[ ]:




