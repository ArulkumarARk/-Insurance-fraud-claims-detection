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

# In[2]:


data = pd.read_csv('insurance_claims.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data.dtypes


# In[7]:


data.isnull().sum()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


pd.DataFrame(data['auto_make'].value_counts())


# In[11]:


pd.DataFrame(data['fraud_reported'].value_counts())


# In[12]:


data.columns


# In[13]:


pd.DataFrame(data['auto_model'].value_counts())


# ### Preprocessing

# In[14]:


data = data.drop(['_c39','policy_bind_date','incident_date','policy_csl','capital-loss'], axis=1)


# In[ ]:





# In[15]:


cat_cols = data.select_dtypes('object')
num_cols = data.select_dtypes('number')
cat_cols.columns,num_cols.columns


# In[16]:


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


# ##### Plot Headmap :
# Headmap to check Correlation ( Correlation explains how one or more variables are related to each other )

# In[17]:


sns.set(style="dark")
f, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(data.corr(), cmap=cmap, vmax=.3, center=0,annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[18]:


# outliers
plt.figure(figsize = (7,8))
data.boxplot()
plt.xticks(rotation=90)
plt.show()


# In[19]:


def encoder(df):
    
    for col in df.select_dtypes('object'):
        le=LabelEncoder()
        le.fit(df[col])
        df[col]=le.transform(df[col])
    
    return df


# In[20]:


data = encoder(data)


# ### Extract X and y

# In[21]:


X = data.drop(['fraud_reported'],axis=1)
y = data['fraud_reported']


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[23]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### My methodology

# #### LogisticRegression

# In[24]:


classifier = LogisticRegression(solver='lbfgs' ,max_iter=9000)
classifier.fit(X_train,y_train)
y_pred_lr = classifier.predict(X_test)
y_pred_lr


# #### Accuracy Score

# In[25]:


asLR = accuracy_score(y_test, y_pred_lr)
asLR


# In[26]:


confusion_matrix(y_test,y_pred_lr)


# In[27]:


print(classification_report(y_test,y_pred_lr,zero_division=0))


# #### KNeighborsClassifier

# In[28]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_pred_knn


# #### Accuracy Score

# In[29]:


asKNN = accuracy_score(y_test, y_pred_knn)
asKNN


# In[30]:


confusion_matrix(y_test,y_pred_knn)


# In[31]:


print(classification_report(y_test,y_pred_knn,zero_division=0))


# #### DecisionTreeClassifier

# In[32]:


dt = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth=4, min_samples_leaf= 5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_pred_dt


# #### Accuracy Score

# In[33]:


asDT = accuracy_score(y_test, y_pred_dt)
asDT


# In[34]:


confusion_matrix(y_test,y_pred_dt)


# In[35]:


print(classification_report(y_test,y_pred_dt,zero_division=0))


# In[36]:


fig = plt.figure(figsize=(25,20))
tree.plot_tree(dt,feature_names=X.columns,class_names=['No','Yes'],filled = True, rounded = True)


# In[37]:


print(classification_report(y_test,y_pred_dt,zero_division=0))


# In[38]:


feat_importances = pd.Series(dt.feature_importances_, index=X.columns)
feat_importances


# In[39]:


feat_importances.nlargest(8).plot(kind='bar')


# #### Comparison 

# ###### MSE

# In[40]:


import sklearn.metrics as metrics


# In[41]:


lr_mse=metrics.mean_squared_error(y_test, y_pred_lr)
print("MSE LR: ",lr_mse)
knn_mse=metrics.mean_squared_error(y_test, y_pred_knn)
print("MSE KNN: ",knn_mse)
dt_mse=metrics.mean_squared_error(y_test, y_pred_dt)
print("MSE DT: ",dt_mse)


# In[42]:


data_mse = {'lr_mse':[0.305],'knn_mse':[0.305],'dt_mse':[0.195]}
def best_model(data_mse):
# Calculating the lowest MSE
    mse_min = min(data_mse.values())
# Storing the lowest MSE in result
    result = [key for key in data_mse if data_mse[key] == mse_min]
    Model_name = []
    if result == ['lr_mse']:
        a = 'LinearRegression'
        Model_name.append(a)
    elif result == ['knn_mse']:
        b = 'KNeighborsRegressor'
        Model_name.append(b)
    elif result == ['dt_mse']:
        c = 'DecisionTreeClassifier'
        Model_name.append(c)
# Printing the result
    print("The best model with the lowest MSE to be selected is", Model_name)

best_model(data_mse)


# #### Plot 

# In[43]:


classifiers = ['KNN', 'Logistic Regression', 'Decision Tree']
scores = [asKNN, asLR,asDT]


# In[44]:


fig, ax = plt.subplots(figsize=(8, 6))

for i in range(len(classifiers)):
    ax.bar(classifiers[i], scores[i])

ax.set_xlabel('Classifier')
ax.set_ylabel('Accuracy Score')
ax.set_title('Accuracy Scores for Insurance Fraud Claims Detection')
plt.show()


# #### ROC

# In[80]:


pred_prob  = classifier.predict_proba(X_test)
pred_prob1 = knn.predict_proba(X_test)
pred_prob2 = dt.predict_proba(X_test)
fpr,tbr,threshold    = roc_curve(y_test, pred_prob[:,1],pos_label=0)
fpr1,tbr1,threshold1 = roc_curve(y_test, pred_prob1[:,1],pos_label=0)
fpr2,tbr2,threshold2 = roc_curve(y_test, pred_prob2[:,1],pos_label=0)
plt.plot(fpr, tbr, linestyle='-', color='green', label='LogisticRegression')
plt.plot(fpr1, tbr1, linestyle='-', color='red', label='KNN')
plt.plot(fpr2, tbr2, linestyle='-', color='blue', label='DecisionTree')

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc = 'best')
plt.show()


# In[114]:


from sklearn.metrics import roc_curve, auc

# predict class probabilities for each classifier
pred_prob_lr = classifier.predict_proba(X_test)
pred_prob_knn = knn.predict_proba(X_test)
pred_prob_dt = dt.predict_proba(X_test)

# calculate false positive rate (fpr), true positive rate (tpr), and thresholds for each classifier
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, pred_prob_lr[:,1], pos_label=1)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, pred_prob_knn[:,1], pos_label=1)
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, pred_prob_dt[:,1], pos_label=1)

# calculate AUC (Area Under the Curve) for each classifier
auc_lr = auc(fpr_lr, tpr_lr)
auc_knn = auc(fpr_knn, tpr_knn)
auc_dt = auc(fpr_dt, tpr_dt)

# plot ROC curves for each classifier
plt.plot(fpr_lr, tpr_lr, color='green', label='Logistic Regression (AUC = %0.2f)' % auc_lr)
plt.plot(fpr_knn, tpr_knn, color='red', label='KNN (AUC = %0.2f)' % auc_knn)
plt.plot(fpr_dt, tpr_dt, color='blue', label='Decision Tree (AUC = %0.2f)' % auc_dt)

# plot the reference line
plt.plot([0, 1], [0, 1], color='black', linestyle='--')

# set axis labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')

# set legend
plt.legend(loc='lower right')

# show the plot
plt.show()


# ### Make a prediction for a new person

# In[81]:


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


# In[82]:


for i in new_person:
    print(i)


# In[83]:


new_person = encoder(new_person)


# In[84]:


prediction = dt.predict(new_person)
print(prediction)


# In[85]:


if prediction == 1:
    print('Yes, you can give insurance to the person.')
else:
    print('No, you cannot give insurance to the person.')


# ##### Best estimator

# In[86]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# In[87]:


clf = DecisionTreeClassifier()


# In[88]:


parameters = {'criterion': ['gini', 'entropy'],
              'max_depth': [1, 2, 3, 4, 5],
              'min_samples_split': [2, 3, 4, 5],
              'min_samples_leaf': [1, 2, 3, 4, 5]}


# In[89]:


grid_search = GridSearchCV(clf, parameters, cv=5)


# In[90]:


grid_search.fit(X_train, y_train)


# In[91]:


print(grid_search.best_params_)


# In[92]:


dt_bst = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth=3, min_samples_leaf=1, min_samples_split= 3)
dt_bst.fit(X_train, y_train)
y_pred_dt_bst = dt_bst.predict(X_test)
y_pred_dt_bst


# #### Accuracy Score

# In[93]:


asDT_bst = accuracy_score(y_test, y_pred_dt_bst)
asDT_bst


# In[94]:


confusion_matrix(y_test,y_pred_dt_bst)


# In[95]:


print(classification_report(y_test,y_pred_dt_bst,zero_division=0))


# In[96]:


pred_prob_dt_bst = dt_bst.predict_proba(X_test)
pred_prob_dt = dt.predict_proba(X_test)
fprdt,tbrdt,thresholddt = roc_curve(y_test, pred_prob_dt[:,1],pos_label=0)
fpr1bst,tbrbst,thresholdbst = roc_curve(y_test, pred_prob_dt_bst[:,1],pos_label=0)

plt.plot(fprdt, tbrdt, linestyle='-', color='blue', label='DecisionTree')
plt.plot(fpr1bst, tbrbst, linestyle='-', color='red', label='DecisionTree with Best estimator')

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc = 'best')
plt.show()


# In[115]:


from sklearn.metrics import roc_curve, auc

# predict class probabilities for decision tree classifiers
pred_prob_dt = dt.predict_proba(X_test)
pred_prob_dt_bst = dt_bst.predict_proba(X_test)

# calculate false positive rate (fpr), true positive rate (tpr), and thresholds for each classifier
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, pred_prob_dt[:,1], pos_label=1)
fpr_dt_bst, tpr_dt_bst, thresholds_dt_bst = roc_curve(y_test, pred_prob_dt_bst[:,1], pos_label=1)

# calculate AUC (Area Under the Curve) for each classifier
auc_dt = auc(fpr_dt, tpr_dt)
auc_dt_bst = auc(fpr_dt_bst, tpr_dt_bst)

# plot ROC curves for each classifier
plt.plot(fpr_dt, tpr_dt, color='blue', label='Decision Tree (AUC = %0.2f)' % auc_dt)
plt.plot(fpr_dt_bst, tpr_dt_bst, color='red', label='Decision Tree with Best Estimator (AUC = %0.2f)' % auc_dt_bst)

# plot the reference line
plt.plot([0, 1], [0, 1], color='black', linestyle='--')

# set axis labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')

# set legend
plt.legend(loc='lower right')

# show the plot
plt.show()


# In[97]:


pred_prob_dt1 = dt.predict_proba(X_test)
fprdt1,tbrdt1,thresholddt1 = roc_curve(y_test, pred_prob_dt1[:,1],pos_label=0)
plt.plot(fprdt1, tbrdt1, linestyle='-', color='blue', label='DecisionTree')

pred_prob_dt = dt.predict_proba(X_test)
fprdt,tbrdt,thresholddt = roc_curve(y_test, pred_prob_dt[:,1],pos_label=0)
plt.plot(fprdt, tbrdt, linestyle='-', color='blue', label='DecisionTree')

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc = 'best')
plt.show()


# In[116]:


from sklearn.metrics import roc_curve, auc

# predict class probabilities for decision tree classifier
pred_prob_dt = dt.predict_proba(X_test)

# calculate false positive rate (fpr), true positive rate (tpr), and thresholds for decision tree classifier
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, pred_prob_dt[:,1], pos_label=0)

# calculate AUC (Area Under the Curve) for decision tree classifier
auc_dt = auc(fpr_dt, tpr_dt)

# plot ROC curve for decision tree classifier
plt.plot(fpr_dt, tpr_dt, color='blue', label='Decision Tree (AUC = %0.2f)' % auc_dt)

# plot the reference line
plt.plot([0, 1], [0, 1], color='black', linestyle='--')

# set axis labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')

# set legend
plt.legend(loc='lower right')

# show the plot
plt.show()


# In[98]:


plt_bst_dt=classification_report(y_test,y_pred_dt_bst,zero_division=0)
precision = []
recall = []
for line in plt_bst_dt.split('\n')[2:4]:
    p, r, _, _ = map(float, line.split()[1:])
    precision.append(p)
    recall.append(r)

# Create scatter plot
plt.scatter(precision, recall)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.show()


# In[99]:


plt_dt=classification_report(y_test,y_pred_dt,zero_division=0)
precision = []
recall = []
for line in plt_dt.split('\n')[2:4]:
    p, r, _, _ = map(float, line.split()[1:])
    precision.append(p)
    recall.append(r)

# Create scatter plot
plt.scatter(precision, recall)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.show()


# ##### Best Feature columns :

# In[100]:


dt_bst_f = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth=4, min_samples_leaf= 5)


# In[101]:


bst_x = X[['incident_severity','insured_hobbies']]


# In[102]:


bst_x


# In[103]:


X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(bst_x,y,test_size=0.2, random_state=42)


# In[104]:


sc = StandardScaler()
X_train_b = sc.fit_transform(X_train_b)
X_test_b = sc.transform(X_test_b)


# In[105]:


dt_bst_f.fit(X_train_b, y_train_b)
y_pred_dt_bst_bt = dt_bst_f.predict(X_test_b)
y_pred_dt_bst_bt


# ##### Accuracy Score

# In[106]:


asDT_bst_b = accuracy_score(y_test_b, y_pred_dt_bst)
asDT_bst_b


# In[107]:


confusion_matrix(y_test_b,y_pred_dt_bst)


# In[108]:


print(classification_report(y_test_b,y_pred_dt_bst,zero_division=0))


# In[109]:


sns.set_style("whitegrid")

# Create the bar plot

sns.catplot(x="incident_severity", hue="fraud_reported", col="insured_hobbies",
            data=data, kind="count", height=10, aspect=.7)

# Set the labels for the plot
plt.xlabel("Incident Severity")
plt.ylabel("Count")
plt.suptitle("Fraud Reported by Incident Severity and Insured Hobbies")

# Show the plot
plt.show()


# ##### ROC

# In[110]:


dt_bst_f = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth=3, min_samples_leaf=1, min_samples_split= 3)
dt_bst_f.fit(X_train_b, y_train_b)
y_pred_dt_bst_bt = dt_bst_f.predict(X_test_b)
y_pred_dt_bst_bt


# In[111]:


pred_prob_dt_bst = dt_bst.predict_proba(X_test)
pred_prob_dt = dt.predict_proba(X_test)
pred_prob_dt_bf = dt_bst_f.predict_proba(X_test_b)
fprdt,tbrdt,thresholddt = roc_curve(y_test, pred_prob_dt[:,1],pos_label=0)
fpr1bst,tbrbst,thresholdbst = roc_curve(y_test, pred_prob_dt_bst[:,1],pos_label=0)
fpr1bstf,tbrbstf,thresholdbstf = roc_curve(y_test, pred_prob_dt_bf[:,1],pos_label=0)

plt.plot(fprdt, tbrdt, linestyle='-', color='blue', label='DecisionTree')
plt.plot(fpr1bst, tbrbst, linestyle='-', color='red', label='DecisionTree with Best estimator')
plt.plot(fpr1bstf, tbrbstf, linestyle='-', color='green', label='DecisionTree with Best Feature')

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc = 'best')
plt.show()


# In[117]:


# Import required libraries
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get predicted probabilities for each model
pred_prob_dt = dt.predict_proba(X_test)
pred_prob_dt_bst = dt_bst.predict_proba(X_test)
pred_prob_dt_bf = dt_bst_f.predict_proba(X_test_b)

# Get fpr, tpr, and threshold values for each model
fpr_dt, tpr_dt, threshold_dt = roc_curve(y_test, pred_prob_dt[:,1], pos_label=0)
fpr_dt_bst, tpr_dt_bst, threshold_dt_bst = roc_curve(y_test, pred_prob_dt_bst[:,1], pos_label=0)
fpr_dt_bf, tpr_dt_bf, threshold_dt_bf = roc_curve(y_test_b, pred_prob_dt_bf[:,1], pos_label=0)

# Compute AUC for each model
auc_dt = auc(fpr_dt, tpr_dt)
auc_dt_bst = auc(fpr_dt_bst, tpr_dt_bst)
auc_dt_bf = auc(fpr_dt_bf, tpr_dt_bf)

# Plot ROC curves and show AUC values in legend
plt.plot(fpr_dt, tpr_dt, linestyle='-', color='blue', label=f'Decision Tree (AUC = {auc_dt:.2f})')
plt.plot(fpr_dt_bst, tpr_dt_bst, linestyle='-', color='red', label=f'Decision Tree (Best Estimator) (AUC = {auc_dt_bst:.2f})')
plt.plot(fpr_dt_bf, tpr_dt_bf, linestyle='-', color='green', label=f'Decision Tree (Best Feature) (AUC = {auc_dt_bf:.2f})')

# Add diagonal reference line and axis labels
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

# Add legend and show plot
plt.legend(loc='best')
plt.show()


# In[112]:


plt_dt_f=classification_report(y_test_b,y_pred_dt_bst,zero_division=0)
precision = []
recall = []
for line in plt_dt_f.split('\n')[2:4]:
    p, r, _, _ = map(float, line.split()[1:])
    precision.append(p)
    recall.append(r)

# Create scatter plot
plt.scatter(precision, recall)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.show()


# In[113]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='incident_severity', hue='fraud_reported', data=data)
plt.show()

sns.countplot(x='insured_hobbies', hue='fraud_reported', data=data)
plt.show()


# In[ ]:




