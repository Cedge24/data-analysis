#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd

import matplotlib as plt
import seaborn as sns
import sqlalchemy as alch

from sqlalchemy import text
from sqlalchemy.sql import func
from sqlalchemy import inspect
from sqlalchemy_utils import database_exists, create_database


# # Cursory Analysis

# In[2]:


df = pd.read_csv('DATA\casestudy.csv')


# In[3]:


print(df.shape)
df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[14]:


df.isna().sum
df.dtypes


# In[7]:


#df['Unnamed: 0'].value_counts() # <--- index
#df['net_revenue'].value_counts()
df['year'].value_counts()


# # SQL Setup

# In[24]:


engine = alch.create_engine('sqlite:///db.sqlite')
if not database_exists(engine.url):
    create_database(engine.url)
    
print(database_exists(engine.url))


# In[25]:


connection = engine.connect()
metadata = alch.MetaData()


# In[26]:


table_name = 'orders'
df.to_sql(
    table_name,
    engine,
    if_exists='replace',
    index=False,
)


# In[32]:


inspector = inspect(engine)
inspector.get_columns('orders')


# # Queries

# In[154]:


# Verifying dataframe -> database 
statement='SELECT * FROM orders'
with engine.connect() as con:

    rs = con.execute(statement)

    for row in rs:
        print(row)


# In[55]:


conn = engine.connect()


# In[72]:


# question 0,0
q0_0 = text("SELECT SUM(net_revenue)"
            " FROM orders"
            " WHERE year=:x")

a0_0 = conn.execute(q0_0, {"x":2015}).fetchall()
print(a0_0)


# In[60]:


# question 0,1
# same as q0,0


# In[ ]:


# question 0,2
# N/A : first year


# In[ ]:


# question 0,3
# N/A : first year


# In[ ]:


# question 0,4
# same as 0,1


# In[ ]:


# question 0,5
# N/A : first year


# In[64]:


# question 0,6
q0_6 = text("SELECT COUNT(*) FROM orders WHERE year=:x")
a0_6 = conn.execute(q0_6, {"x":2015}).fetchall()
print(a0_6)


# In[ ]:


# question 0,7
# N/A : first year


# In[ ]:


# question 0,8
# same as 0,6


# In[ ]:


# question 0,9
# N/A : first year


# In[74]:


# question 1,0
q1_0 = text("SELECT SUM(net_revenue)"
            " FROM orders"
            " WHERE year=:x")

a1_0 = conn.execute(q1_0, {"x":2016}).fetchall()
print(a1_0)


# In[85]:


# question 1,1
q1_1 = text("SELECT SUM(net_revenue) "
            "FROM orders "
            "WHERE year=:x "
            "AND customer_email NOT IN (SELECT customer_email FROM orders WHERE year=:y)")

a1_1 = conn.execute(q1_1,{"x":2016,"y":2015}).fetchall()
print(a1_1)


# In[120]:


# question 1,2
t1 =    ("SELECT SUM(net_revenue) "
        "FROM orders "
        "WHERE year=:x "
        "AND customer_email IN (SELECT customer_email FROM orders WHERE year=:y)")

t2 =   ("SELECT SUM(net_revenue) "
        "FROM orders "
        "WHERE year=:y "
        "AND customer_email IN (SELECT customer_email FROM orders WHERE year=:x)")

d1 = conn.execute(t1,{"x":2016,"y":2015}).fetchall()
d2 = conn.execute(t2,{"x":2016,"y":2015}).fetchall()

print(d1[0])
print(d2[0])

a1_2 = 7485452.5800000075 - 7465117.120000009
print(a1_2)


# In[125]:


# question 1,3 : rev(existing customers)
q1_3a = "SELECT SUM(net_revenue) FROM orders WHERE year=:y"
q1_3b = "SELECT SUM(net_revenue) FROM orders WHERE year=:x"

print(conn.execute(q1_3a,{"x":2016,"y":2015}).fetchall())
print(conn.execute(q1_3b,{"x":2016,"y":2015}).fetchall())

a1_3 = 29036749.18999953 - 25730943.58999988
print(a1_3)


# In[126]:


# question 1,4
q1_4 = "SELECT SUM(net_revenue) FROM orders WHERE year=:x AND customer_email IN (SELECT customer_email FROM orders WHERE year=:y)"

a1_4 = conn.execute(q1_4,{"x":2016,"y":2015}).fetchall()
print(a1_4)


# In[127]:


# question 1,5
# same as 1,2b
a1_5 = 7465117.120000009
print(a1_5)


# In[132]:


# question 1,6
q1_6 = "SELECT COUNT(*) FROM orders WHERE year=:x"

a1_6a = conn.execute(q1_6,{"x":2016,"y":2015}).fetchall()
print(a1_6a)
print(a0_6)
a1_6 = 204646+231294
print(a1_6)


# In[133]:


# question 1,7
# same as 0_6
a1_7 = a0_6
print(a1_7)


# In[150]:


# question 1,8
q1_8 = ("SELECT COUNT(*) FROM orders WHERE year=:x "
       "AND customer_email NOT IN (SELECT customer_email WHERE year=:y)")

a1_8 = conn.execute(q1_8,{"x":2016,"y":2015}).fetchall()
print(a1_8)


# In[152]:


# question 1,9
q1_9 = ("SELECT COUNT(*) FROM orders WHERE year=:x "
        "AND customer_email NOT IN (SELECT customer_email WHERE year=:y)")

a1_9 = conn.execute(q1_9,{"x":2015,"y":2016}).fetchall()
print(a1_9)


# In[137]:


# question 2,0
q2_0 = "SELECT SUM(net_revenue) FROM orders WHERE year=:x"

a2_0 = conn.execute(q2_0,{"x":2017}).fetchall()

print(a2_0)


# In[139]:


# question 2,1
q2_1 = ("SELECT SUM(net_revenue) FROM orders WHERE year=:x "
       "AND customer_email NOT IN (SELECT customer_email FROM orders WHERE year=:y)")

a2_1 = conn.execute(q2_1,{"x":2017,"y":2016}).fetchall()

print(a2_1)


# In[141]:


# question 2,2
t1 =    ("SELECT SUM(net_revenue) "
        "FROM orders "
        "WHERE year=:x "
        "AND customer_email IN (SELECT customer_email FROM orders WHERE year=:y)")

t2 =   ("SELECT SUM(net_revenue) "
        "FROM orders "
        "WHERE year=:y "
        "AND customer_email IN (SELECT customer_email FROM orders WHERE year=:x)")

d1 = conn.execute(t1,{"x":2017,"y":2016}).fetchall()
d2 = conn.execute(t2,{"x":2017,"y":2016}).fetchall()

print(d1[0])
print(d2[0])

a2_2 = 2641259.990000008 - 2620648.6499999906
print(a2_2)


# In[145]:


# question 2,3
q2_3a = "SELECT SUM(net_revenue) FROM orders WHERE year=:y"
q2_3b = "SELECT SUM(net_revenue) FROM orders WHERE year=:x"

print(conn.execute(q2_3a,{"x":2017,"y":2016}).fetchall())
print(conn.execute(q2_3b,{"x":2017,"y":2016}).fetchall())

a2_3 = 25730943.58999988 - 31417495.02999995
print(a2_3)


# In[146]:


# question 2,4
q2_4 = "SELECT SUM(net_revenue) FROM orders WHERE year=:x AND customer_email IN (SELECT customer_email FROM orders WHERE year=:y)"

a2_4 = conn.execute(q2_4,{"x":2017,"y":2016}).fetchall()
print(a2_4)


# In[147]:


# question 2,5
q2_5 = "SELECT SUM(net_revenue) FROM orders WHERE year=:x AND customer_email IN (SELECT customer_email FROM orders WHERE year=:y)"

a2_5 = conn.execute(q2_5,{"x":2016,"y":2017}).fetchall()
print(a2_5)


# In[149]:


# question 2,6 : interpreted the question as total customers in current year cumulative with past years
q2_6 = "SELECT COUNT(*) FROM orders WHERE year=:x"

a2_6a = conn.execute(q2_6,{"x":2016,"y":2015}).fetchall()

print(a2_6a)
print(a1_6)

a2_6 = 204646 + 435940
print(a2_6)


# In[ ]:


# question 2,7
# same as 1,6


# In[151]:


# question 2,8
q2_8 = ("SELECT COUNT(*) FROM orders WHERE year=:x "
       "AND customer_email NOT IN (SELECT customer_email WHERE year=:y)")

a2_8 = conn.execute(q1_8,{"x":2017,"y":2016}).fetchall()
print(a2_8)


# In[153]:


# question 2,9 
q2_9 = ("SELECT COUNT(*) FROM orders WHERE year=:x "
        "AND customer_email NOT IN (SELECT customer_email WHERE year=:y)")

a2_9 = conn.execute(q1_9,{"x":2016,"y":2017}).fetchall()
print(a2_9)

