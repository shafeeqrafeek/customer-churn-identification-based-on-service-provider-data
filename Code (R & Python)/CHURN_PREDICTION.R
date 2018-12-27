library(DMwR)
library(dplyr)
library(ggplot2)
library(C50)
library(ROSE)
library(e1071) 
library(caret) 
library(rpart.plot)
library(rpart)
library(rattle)
library(randomForest)
library(rpart)
library(rattle)

rm(list=ls())
setwd("F:/Data Analytics/Edwisor Project/Churn Identification Project")
churn=read.csv("train_data.csv")
test=read.csv("test_data.csv")

#EXPLORATORY DATA ANALYSIS

bar_visual = function(cat_a,cat_b,xmark,ymark,heading){
  tab =table(cat_a,cat_b)
  c=tab[,1]+tab[,2]
  tab[,1]=tab[,1]/c
  tab[,2]=tab[,2]/c
  tab=tab[,c(2,1)]
  return(barplot(t(tab),main=heading,col = c(5,6),xlab=xmark,ylab=ymark))
}

#state
bar_visual(churn$state,churn$Churn,"state","churn scale","Churning percentages in states")
legend("topright",legend=unique(churn$Churn),fill=c(6,5))

#international plan
bar_visual(churn$international.plan,churn$Churn,"international plan","churn scale",
           "International subscription Vs Churning percentages")
legend("topright",legend=unique(churn$Churn),fill=c(6,5))

#vmail plan
bar_visual(churn$voice.mail.plan,churn$Churn,"voicemail plan","churn scale",
           "voicemail subscription Vs Churning percentages")
legend("topright",legend=unique(churn$Churn),fill=c(6,5))

#vmail messages
a=dplyr::filter(churn,voice.mail.plan==' yes')
ggplot(a,aes(x=Churn,y=number.vmail.messages))+theme_bw()+geom_boxplot()+
  labs(title="voice mail messages vs churn")

#day
ggplot(churn,aes(x=Churn,y=total.day.charge))+theme_bw()+geom_boxplot()+
  labs(y="total daytime tariff",title="Total Daytime Tariff Vs churn")

#eve
ggplot(churn,aes(x=Churn,y=total.eve.charge))+theme_bw()+geom_boxplot()+
  labs(y="total evening tariff",title="Total evening Tariff Vs churn")

#night
ggplot(churn,aes(x=Churn,y=total.night.charge))+theme_bw()+geom_boxplot()+
  labs(y="total night tariff",title="Total night Tariff Vs churn")

#international
ggplot(churn,aes(x=Churn,y=total.intl.charge))+theme_bw()+geom_boxplot()+
  facet_wrap(~international.plan)+
  labs(title="international charges as per international plan",
       y="total intl calls  tariff")

#international calls
ggplot(churn,aes(x=Churn,y=total.intl.minutes))+theme_bw()+geom_boxplot()+facet_wrap(~international.plan)

#customer service calls
boxplot(churn$number.customer.service.calls,ylab="customer service calls")
title("Pseudo outliers in customer service calls")
ggplot(churn,aes(x=Churn,y=number.customer.service.calls))+
  theme_bw()+geom_boxplot()+labs(title="customer service calls vs Churn")

dtype_correction=function(datfrme,to_variable)
  {
  facto=c()
  numero=c()
  for (i in colnames(datfrme))
    {
         if (class(datfrme[,i])=='factor')
           {
           facto=append(facto,i)
           datfrme[,i] = factor(datfrme[,i],labels=(1:length(levels(datfrme[,i]))))
           assign(to_variable,datfrme,envir=.GlobalEnv)
           }
         else
           {
           numero=append(numero,i)
           }
  }
  return(list(facto,numero))
  }

outlier_removal=function(df,num,to_variable){
  for (i in num){
    outliers=df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
    out.index=row.names(df[which(df[,i] %in% outliers & df$Churn==1),])
    in.indices=!row.names(df) %in% out.index
    df=df[in.indices,]
    df[which(df[,i] %in% outliers),i]=NaN
    assign(to_variable,df,envir = .GlobalEnv)
  }
}

chi_sqre_test=function(df,fact,trgt){
  col_fact=c()
  for (i in fact){
    x=chisq.test(table(df[,trgt],df[,i]))
    if (x$p.value<0.05){
      col_fact=append(col_fact,i)
    }
  }
  return(col_fact)
}

corr_test=function(df,num,thres=0.85){
  cmat=abs(cor(df[num]))
  cmat[lower.tri(cmat,diag=TRUE)]=NA
  cmat=as.data.frame(cmat)
  col_corr=c()
  for (i in num){
    col_corr=append(col_corr,row.names(cmat[which(cmat[,i]>thres),]))
  }
  return(col_corr)
}

anova_test=function(df,num,trgt){
  col_ano=c()
  for (i in num){
    ftest=aov(df[,i] ~ df[,trgt], data=df)
    qu=summary(ftest)
    if (qu[[1]][1,5]<0.05){
      col_ano=append(col_ano,i)
    }
  }
  return(col_ano)
}

normalize=function(df,num){
  for (i in num){
    df[,i]=(df[,i]-min(df[,i]))/(max(df[,i])-min(df[,i]))
  }
  return(df)
}

churn$phone.number=as.numeric(gsub("-","",churn$phone.number))
churn$area.code=as.factor(churn$area.code)
vars=dtype_correction(churn,'churn')
outlier_removal(churn,vars[[2]][1:15],'churn')
churn = knnImputation(churn, k = 3)
cols_fact=chi_sqre_test(churn,vars[[1]],'Churn')
cols_num1=corr_test(churn,vars[[2]])
cols_num2=anova_test(churn,vars[[2]],'Churn')
cols_num=setdiff(cols_num2,cols_num1)
valid_cols=append(cols_num,cols_fact)

churn=churn[,valid_cols]
churn=normalize(churn,cols_num)

#TEST data preparation
test_vars=dtype_correction(test,'test')
test=test[,valid_cols]
test=normalize(test,cols_num)

#RESOLVING CLASS IMBALANCE
#over=ovun.sample(Churn~.,data = churn,method='over',N=5230)$data
#under=ovun.sample(Churn~.,data = churn,method='under',N=966)$data
both=ovun.sample(Churn~.,data = churn,method='both')$data

#LOGISTIC REGRESSION
logistic_model=glm(Churn~.,data=churn,family='binomial')
logistic_Predictions = predict(logistic_model, newdata = test, type = "response")
logistic_Predictions = as.factor(ifelse(logistic_Predictions > 0.5, 2, 1))
logistic_conf_matrix=confusionMatrix(logistic_Predictions,test$Churn,positive = '2')
logistic_conf_matrix

#NAIVE BAYES
NB_model=naiveBayes(Churn~.,data = churn)
NB_predictions=predict(NB_model,test[,1:10],type='class')
NB_conf=confusionMatrix(NB_predictions,test$Churn,positive = '2')
NB_conf

#DECISION TREE
C50_model = C5.0(Churn ~., churn, trials = 100, rules = FALSE)
C50_Predictions = predict(C50_model, test[,-11], type = "class")
DT_conf=confusionMatrix(C50_Predictions,test$Churn,positive = '2')
DT_conf
#DECISON TREE VISUALISATION
churn_tree=rpart(Churn~.,data=churn,method='class')
fancyRpartPlot(churn_tree,cex=0.4)

#RANDOM FOREST
RF_model = randomForest(Churn~.,data=churn,importance=TRUE)
RF_Predictions = predict(RF_model, test[,-11])
RF_conf=confusionMatrix(RF_Predictions,test$Churn,positive = '2')
RF_conf

#RESOLVING CLASS IMBALANCE AND FREEZING FINAL MODEL
C50_model_both = C5.0(Churn ~., both, trials = 100, rules = TRUE)
C50_Predictions_both = predict(C50_model_both, test[,-11], type = "class")
ConfMatrix_c50_both = confusionMatrix(C50_Predictions_both,test$Churn,positive='2')
ConfMatrix_c50_both