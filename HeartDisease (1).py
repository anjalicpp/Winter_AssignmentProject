#!/usr/bin/env python
# coding: utf-8

# #Import Libraries

# In[1]:


import pandas as pd


# 2.Importing the dataset

# In[5]:


data = pd.read_csv(r'C:\Users\Anjali\Downloads\archive\heart.csv')


# In[6]:


print(data)


# In[ ]:





# Taking Care of Missing value

# In[7]:


data.isnull().sum()


# #Taking care of duplicate values

# In[8]:


data_dup = data.duplicated().any()


# In[10]:


data_dup

true means some values are duplicate
# In[12]:


data = data.drop_duplicates()


# In[13]:


data_dup = data.duplicated().any()


# In[14]:


data_dup


# false means duplicate value is dropped down

# #Data processing

# In[17]:


cate_val=[]#categorical data columns
cont_val=[]#

for column in data.columns:
    if data[column].nunique()<=10:
      cate_val.append(column)
    else:
        cont_val.append(column)


# In[19]:


cate_val#it is columns with categorical data


# In[20]:


cont_val#it has columns with numerical value


# # 6Encoding Cateegorical Data

# In[21]:


cate_val


# In[22]:


data['cp'].unique()#checking unique values in each category here cp column


# #dummy variable trap check here 

# In[23]:


data['sex'].unique()


# In[24]:


cate_val.remove('sex')
cate_val.remove('target')
data = pd.get_dummies(data,columns=cate_val,drop_first=True)#creating dummies for categorical data


# In[25]:


data.head()


# # Feature Scaling 

# In[ ]:


#check which ml algorithm requires fetaure scaling or not ....why it is required


# In[26]:


data.head()


# In[29]:


pip install scikit-learn


# In[30]:


from sklearn.preprocessing import StandardScaler


# In[32]:


st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])


# In[33]:


data.head()


# In[ ]:


#8.Splitting THe dataset into the training set and test set


# In[34]:


X=data.drop('target',axis=1)


# In[35]:


Y=data['target']


# In[36]:


Y


# In[37]:


from sklearn.model_selection import train_test_split


# In[40]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# In[41]:


X_train


# In[43]:


Y_train


# # Logistic Regression

# In[44]:


data.head()#since target is 0 or 1 which indicates whether it has heart disease or not it is cateogorical problem


# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


log = LogisticRegression()
log.fit(X_train,Y_train)#train model using fit method


# In[47]:


y_pred1= log.predict(X_test)


# In[48]:


from sklearn.metrics import accuracy_score


# In[50]:


accuracy_score(Y_test,y_pred1)


# # 10 SVC

# In[53]:


from sklearn.svm import SVC


# In[54]:


svm = svm.SVC()
#created instance of this svm that is SvC


# In[55]:


svm.fit(X_train,Y_train)


# 

# In[56]:


y_pred2 = svm.predict(X_test)


# In[58]:


accuracy_score(Y_test,y_pred2)


# In[ ]:





# # 11.KNeighbours Classifier
# 

# In[59]:


from sklearn.neighbors import KNeighborsClassifier


# In[64]:


knn = KNeighborsClassifier()#press shift+tab in () n_neighbour =5 default


# In[61]:


knn.fit(X_train,Y_train)


# In[62]:


y_pred3=knn.predict(X_test)


# In[63]:


accuracy_score(Y_test,y_pred3)


# In[66]:


score = []
for k in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=k)#chaning N-neighour value to cherck at whicjh it gives  max accuaray
    knn.fit(X_train,Y_train)
    y_pred=knn.predict(X_test)
    score.append(accuracy_score(Y_test,y_pred))
    


# In[67]:


score#here model has max accuracy value at k=2 that is n_neighbor value=2


# In[71]:


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)
accuracy_score(Y_test,y_pred)



# # Non linear ML Algorithm(preprocessing not required for it)

# In[72]:


data = pd.read_csv(r'C:\Users\Anjali\Downloads\archive\heart.csv')


# In[73]:


data.head()


# In[74]:


data = data.drop_duplicates()


# In[75]:


data.shape


# In[76]:


x=data.drop("target",axis=1)
y=data["target"]


# In[77]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:





# # Decision Tree Classifier

# In[79]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)


# In[80]:


y_pred4= dt.predict(X_test)

accuracy_score(y_test,y_pred4)


# # Random Forest Classifier

# In[81]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[82]:


y_pred5= rf.predict(X_test)
accuracy_score(y_test,y_pred5)


# # Gradient Boosting Classifier

# In[83]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)


# In[84]:


y_pred6 = gbc.predict(X_test)
accuracy_score(y_test,y_pred6)


# In[85]:


final_data = pd.DataFrame({'Models':['LR','SVM','KNN','DT','RF','GB'],
                          'ACC':[accuracy_score(y_test,y_pred1)*100,
                                accuracy_score(y_test,y_pred2)*100,
                                accuracy_score(y_test,y_pred3)*100,
                                accuracy_score(y_test,y_pred4)*100,
                                accuracy_score(y_test,y_pred5)*100,
                                accuracy_score(y_test,y_pred6)*100]})


# In[86]:


final_data


# In[97]:


#pip install seaborn
python -m pip install seaborn




# In[93]:


import seaborn as sns
sns.barplot(final_data['Models'],final_data['ACC'])


# In[ ]:





# In[ ]:





# In[98]:


from tkinter import *
import joblib


# In[ ]:


from tkinter import *
import joblib
import numpy as np
from sklearn import *
def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=int(e8.get())
    p9=int(e9.get())
    p10=float(e10.get())
    p11=int(e11.get())
    p12=int(e12.get())
    p13=int(e13.get())
    model = joblib.load('model_joblib_heart')
    result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p8,p10,p11,p12,p13]])
    
    if result == 0:
        Label(master, text="No Heart Disease").grid(row=31)
    else:
        Label(master, text="Possibility of Heart Disease").grid(row=31)
        
        
master = Tk()
master.title("Heart Disease Prediction System")


label = Label(master, text = "Heart Disease Prediction System"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)


Label(master, text="Enter Your Age").grid(row=1)
Label(master, text="Male Or Female [1/0]").grid(row=2)
Label(master, text="Enter Value of CP").grid(row=3)
Label(master, text="Enter Value of trestbps").grid(row=4)
Label(master, text="Enter Value of chol").grid(row=5)
Label(master, text="Enter Value of fbs").grid(row=6)
Label(master, text="Enter Value of restecg").grid(row=7)
Label(master, text="Enter Value of thalach").grid(row=8)
Label(master, text="Enter Value of exang").grid(row=9)
Label(master, text="Enter Value of oldpeak").grid(row=10)
Label(master, text="Enter Value of slope").grid(row=11)
Label(master, text="Enter Value of ca").grid(row=12)
Label(master, text="Enter Value of thal").grid(row=13)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)



Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()

