
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import KNN
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import f_classif as ano
from sklearn.ensemble import RandomForestClassifier

#directory setting
os.chdir("F:\Data Analytics\Edwisor Project\Churn Identification Project")
os.getcwd()


# In[2]:


def dtype_order(df):
    fact,num=[],[]
    for i in df.columns:
        if df[i].dtype=='object':
            df[i]=df[i].astype('category')
            df[i]=df[i].cat.codes
            #churn[i]=churn[i].astype('category')
            fact.append(i)
        else:
            num.append(i)
    for i in fact:
        df[i]=df[i].astype('category')
    return num,fact


# In[3]:


churn=pd.read_csv("train_data.csv")
test=pd.read_csv("test_data.csv")


# In[4]:


churn['phone number']=churn['phone number'].str.replace("-","")
churn['phone number']=churn['phone number'].astype('int')
churn['area code']=churn['area code'].astype('object')
churn_numeric,churn_fact=dtype_order(churn)


# In[5]:


#Exploratory Data Analysis
def bar_visual(categ_a,categ_b,heading):
    tab=pd.crosstab(churn[categ_a],churn[categ_b])
    c=tab.iloc[:,0]+tab.iloc[:,1]
    tab.iloc[:,0]=tab.iloc[:,0]/c
    tab.iloc[:,1]=tab.iloc[:,1]/c
    tab=tab.iloc[:,[1,0]]
    t=tab.plot.bar(stacked=True,fontsize=7)
    t.set_ylabel('churning rate')
    t.set_title(heading)
    return t


# In[6]:


bar_visual('state','Churn','state vs churn')
bar_visual('voice mail plan','Churn','voicemail plan vs churn')
bar_visual('international plan','Churn','international plan vs churn')


# In[10]:


sns.boxplot(churn['Churn'],churn['total day charge'])
plt.title("day tariff vs Churn")


# In[8]:


sns.boxplot(churn['Churn'],churn['total eve charge'])
plt.title("evening tariff vs Churn")


# In[9]:


sns.boxplot(churn['Churn'],churn['total night charge'])
plt.title("night tariff vs Churn")


# In[11]:


sns.boxplot('international plan','total intl charge',hue='Churn',data=churn)
plt.title("intl call charges based on intl plan")


# In[20]:


a=churn.loc[churn['voice mail plan']==1]
sns.boxplot('voice mail plan','number vmail messages',hue='Churn',data=a)
plt.title("vmail messages based on vmail plan")


# In[21]:


sns.boxplot(churn['Churn'],churn['number customer service calls'])
plt.title("customer calls vs Churn")


# In[27]:


df_corr=churn.loc[:,churn_numeric]
df_corr=df_corr.corr().abs()
df_corr=pd.DataFrame(np.triu(df_corr,k=1),columns=df_corr.columns,index=df_corr.index)
pl=sns.diverging_palette(10,220,as_cmap=True)
sns.heatmap(df_corr,cmap=pl,square=True,vmin=-0.6)


# In[24]:


# chi_square
def Chi_Sqre_test(df,fact,trgt):
    col_to_consider=[]
    for i in fact:
        if i !=trgt:
            chi2, p, dof, ex=chi2_contingency(pd.crosstab(df[trgt],df[i]))
            if p<0.05:
                col_to_consider.append(i)
    return col_to_consider


# In[25]:


def anova_test(df,num,trgt):
    col_to_consider=[]
    for i in num:
        F, p=ano(np.array(df.loc[:,i]).reshape(-1,1),df[trgt])
        if p <0.05:
            col_to_consider.append(i)
    return np.array(col_to_consider)


# In[26]:


def corr_test(df,num,t=0.85):
    df_corr=df.loc[:,num]
    df_corr=df_corr.corr().abs()
    df_corr=pd.DataFrame(np.triu(df_corr,k=1),columns=df_corr.columns,index=df_corr.index)
    cols_to_consider=[]
    for i in df_corr.columns:
        df_corr[i]=df_corr[i].mask(df_corr[i]<t)
        cols_to_consider=cols_to_consider+(list(df_corr[df_corr[i]>t].index))
    return np.array(cols_to_consider)


# In[28]:


def normalize(df,num):
    for i in num:
        df[i]=(df[i]-np.min(df[i]))/(np.max(df[i])-np.min(df[i]))


# In[22]:


#OUTLIER REMOVAL
for i in churn_numeric[0:15]:
        q25,q75=np.percentile(churn.loc[:,i],[25,75])
        iqr=q75-q25
        max=q75+(1.5*iqr)
        min=q25-(1.5*iqr)
        churn=churn.drop(churn.loc[churn.loc[:,i]>max,:][churn['Churn']==0].index)
        churn=churn.drop(churn.loc[churn.loc[:,i]<min,:][churn['Churn']==0].index)
        churn.loc[churn.loc[:,i]>max,i]=np.nan
        churn.loc[churn.loc[:,i]<min,i]=np.nan
churn.reset_index(inplace=True,drop=True)


# imputation method      Actual value: 28.27 (churn['total day charge'][1000])
# mean                   30.636593192868744
# median                 30.5
# KNN                    29.88962811568968

# In[23]:


churn=pd.DataFrame(KNN(k=3).complete(churn),columns=churn.columns)


# In[29]:


valid_facts=Chi_Sqre_test(churn,churn_fact,'Churn')+['Churn']
valid_cols1=anova_test(churn,churn_numeric,'Churn')
valid_cols2=corr_test(churn,valid_cols1)
valid_nums=list(np.setdiff1d(valid_cols1,valid_cols2))
cols=valid_nums+valid_facts


# In[30]:


churn=churn.loc[:,cols]
churn_model=churn.copy()
for i in valid_facts:
    churn[i]=churn[i].astype('category')
churn_model['Churn']=churn_model['Churn'].astype('category')


# In[31]:


normalize(churn,valid_nums)
normalize(churn_model,valid_nums)


# In[32]:


test_num,test_fact=dtype_order(test)
test=test.loc[:,cols]
normalize(test,valid_nums)


# In[33]:


xtest=test.iloc[:,0:len(cols)-1]
ytest=test['Churn']
xtrain=churn.iloc[:,0:len(cols)-1]
ytrain=pd.DataFrame(churn['Churn'],columns=['Churn'])


# In[34]:


# Performance metrics
def perf(true_val,predict_val):
    cm=(pd.crosstab(true_val,predict_val))
    cm1=np.array(cm)
    tn= cm1[0,0]
    fp=cm1[0,1]
    fn=cm1[1,0]
    tp=cm1[1,1]
    accuracy=(tn+tp)/(tn+fp+fn+tp)
    sensitivity=abs(1-(fn/(fn+tp)))
    print(cm)
    print("Accuracy:",round(accuracy*100,ndigits=2),"%")
    print("sensitivity:",round(sensitivity*100,ndigits=2),"%")


# # Logistic Regression

# In[35]:


#Data preparation for logistic regression
xtrain_logit=churn_model.iloc[:,0:len(cols)-1]
ytrain_logit=churn_model['Churn']
xtest_logit=test.iloc[:,0:len(cols)-1]
ytest_logit=test['Churn']
xtest_logit['state']=xtest_logit['state'].astype('float64')
xtest_logit['international plan']=xtest_logit['international plan'].astype('float64')
xtest_logit['voice mail plan']=xtest_logit['voice mail plan'].astype('float64')


# In[36]:


#Logistic Regression
logit_churn=sm.Logit(ytrain_logit,xtrain_logit).fit()
logit_predictions=pd.DataFrame(logit_churn.predict(xtest_logit),columns=['probs'])
logit_predictions['value']=1
logit_predictions.loc[logit_predictions.probs<0.5,'value']=0


# In[37]:


#logistic_regression
perf(ytest,logit_predictions['value'])


# # Naive Bayes

# In[38]:


#Naive Bayes
NB_churn=GaussianNB().fit(xtrain,ytrain)
NB_predictions=NB_churn.predict(xtest)


# In[39]:


#Naive Bayes
perf(ytest,NB_predictions)


# # Decision Tree

# In[40]:


#Decision Tree
DT_churn=tree.DecisionTreeClassifier(criterion='entropy').fit(xtrain,ytrain)
DT_predictions=DT_churn.predict(xtest)


# In[41]:


#Decision Tree
perf(ytest,DT_predictions)


# # Random Forest

# In[42]:


#Random Forest
RF_churn=RandomForestClassifier().fit(xtrain,ytrain)
RF_predictions=RF_churn.predict(xtest)


# In[43]:


#Random Forest
perf(ytest,RF_predictions)


# # resolving class imbalance

# In[44]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# In[45]:


under=RandomUnderSampler(sampling_strategy=0.25, return_indices=False, random_state=None, replacement=False)
xunder,yunder=under.fit_resample(xtrain,ytrain)


# In[46]:


over=RandomOverSampler(sampling_strategy='minority', return_indices=False, random_state=None, ratio=None)
xover,yover=over.fit_resample(xunder,yunder)


# In[47]:


RF_churn_bl=RandomForestClassifier().fit(xover,yover)
RF_predictions_bl=RF_churn_bl.predict(xtest)
perf(ytest,RF_predictions_bl)

