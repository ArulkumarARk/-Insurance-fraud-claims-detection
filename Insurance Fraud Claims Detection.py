#!/usr/bin/env python
# coding: utf-8

# # Insurance Fraud Claims Detection¶

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,precision_score,recall_score,roc_auc_score,mean_squared_error,r2_score
from sklearn.metrics import roc_curve


# ### Importing Dataset

# In[155]:


data = pd.read_csv('insurance_claims.csv')


# In[156]:


data.head()


# In[157]:


data.shape


# In[158]:


data.columns


# In[159]:


data.dtypes


# In[160]:


data.isnull().sum()


# In[161]:


data.info()


# In[162]:


data.describe()


# In[163]:


pd.DataFrame(data['fraud_reported'].value_counts())


# ### Preprocessing

# In[164]:


data = data.drop(['_c39','policy_bind_date','incident_date','policy_csl','capital-loss'], axis=1)


# In[165]:


cat_cols = data.select_dtypes('object')
num_cols = data.select_dtypes('number')
cat_cols.columns,num_cols.columns


# In[171]:


cols = ['months_as_customer', 'age', 'policy_number', 'policy_deductable',
        'policy_annual_premium', 'umbrella_limit', 'insured_zip',
        'capital-gains', 'incident_hour_of_the_day',
        'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
        'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
        'auto_year']
plt.figure(figsize = (16, 20))
plotnumber = 1

for i in range(len(cols)):
    if plotnumber <= 18:
        ax = plt.subplot(9, 2, plotnumber)
        sns.kdeplot(x = cols[i], data = data, ax = ax,hue='fraud_reported')
        plt.title(f"\n{cols[i]} Value Counts\n", fontsize = 20)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[173]:


sns.set(style="white")
f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(data.corr(), cmap=cmap, vmax=.3, center=0,annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[176]:


# outliers
plt.figure(figsize = (7,8))
data.boxplot()
plt.xticks(rotation=90)
plt.show()


# In[106]:


def encoder(df):
    
    for col in df.select_dtypes('object'):
        le=LabelEncoder()
        le.fit(df[col])
        df[col]=le.transform(df[col])
    
    return df


# In[131]:


data = encoder(data)


# ### Extract X and y

# In[132]:


X = data.drop(['fraud_reported'],axis=1)
y = data['fraud_reported']


# In[133]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[134]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### My methodology

# In[135]:


classifier = LogisticRegression(solver='lbfgs' ,max_iter=9000)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred


# In[136]:


confusion_matrix(y_test,y_pred)


# In[137]:


accuracy_score(y_test, y_pred)


# In[138]:


print(classification_report(y_test,y_pred,zero_division=0))


# In[139]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred


# In[140]:


accuracy_score(y_test, y_pred)


# In[141]:


confusion_matrix(y_test,y_pred)


# In[142]:


print(classification_report(y_test,y_pred,zero_division=0))


# ### Proposed Methodology

# In[143]:


dt = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth=4, min_samples_leaf= 5)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_pred


# ### Accuracy Score

# In[144]:


accuracy_score(y_test, y_pred)


# In[145]:


print(classification_report(y_test,y_pred,zero_division=0))


# In[146]:


fig = plt.figure(figsize=(25,20))
tree.plot_tree(dt,feature_names=X.columns,class_names=['No','Yes'],filled = True, rounded = True)


# In[147]:


confusion_matrix(y_test,y_pred)


# In[148]:


print(classification_report(y_test,y_pred,zero_division=0))


# In[149]:


feat_importances = pd.Series(dt.feature_importances_, index=X.columns)
feat_importances


# In[150]:


feat_importances.nlargest(8).plot(kind='barh')


# In[152]:


pred_prob  = classifier.predict_proba(X_test)
pred_prob1 = knn.predict_proba(X_test)
pred_prob2 = dt.predict_proba(X_test)
fpr,tbr,threshold    = roc_curve(y_test, pred_prob[:,1],pos_label=0)
fpr1,tbr1,threshold1 = roc_curve(y_test, pred_prob1[:,1],pos_label=0)
fpr2,tbr2,threshold2 = roc_curve(y_test, pred_prob2[:,1],pos_label=0)
plt.plot(fpr, tbr, linestyle='-', color='green', label='LogisticRegression')
plt.plot(fpr1, tbr1, linestyle='-', color='red', label='RandomForest')
plt.plot(fpr2, tbr2, linestyle='-', color='blue', label='DecisionTree')

plt.title('ROC')
plt.legend(loc = 'best')
plt.show()


# ### Make a prediction for a new person

# In[185]:


new_person = pd.DataFrame({'months_as_customer': [12], 'age': [30], 'policy_number': [1000],
                           'policy_state': ['NY'],  'policy_deductable': [500],
                           'policy_annual_premium': [1000], 'umbrella_limit': [500000], 'insured_zip': [12345], 
                           'insured_sex': ['M'], 'insured_education_level': ['Bachelor'], 'insured_occupation': ['Manager'],
                           'insured_hobbies': ['Golf'], 'insured_relationship': ['Husband'], 'capital-gains': [0],
                           'incident_type': ['Single Vehicle Collision'], 
                           'collision_type': ['Front Collision'], 'incident_severity': ['Minor Damage'], 'authorities_contacted': ['Police'],
                           'incident_state': ['NY'], 'incident_city': ['Albany'], 'incident_location': ['Address 123'], 
                           'incident_hour_of_the_day': [10],'number_of_vehicles_involved': [1], 'property_damage': ['NO'],
                           'bodily_injuries': [1], 'witnesses': [2],'police_report_available': ['YES'],
                           'total_claim_amount': [5000], 'injury_claim': [1000],'property_claim': [3000],
                           'vehicle_claim': [1000], 'auto_make': ['Toyota'], 'auto_model': ['Corolla'],'auto_year': [2021]})


# In[186]:


new_person = encoder(new_person)


# In[187]:


prediction = dt.predict(new_person)
print(prediction)


# In[188]:


if prediction == 1:
    print('Yes, you can give insurance to the person.')
else:
    print('No, you cannot give insurance to the person.')


# In[ ]:




