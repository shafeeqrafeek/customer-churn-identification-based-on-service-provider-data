# Predicting customer churning possibilities based on service provider data

_Shafeeq Ahmed_

## Problem Statement

_The objective of this project is to predict whether a particular customer would discontinue his/her subscription with the service provider or not based on the statistical data given by the network service provider. Prediction and prevention of customer churn brings a huge additional revenue and moreover, retaining existing customer is significantly economical compared to spending resources to acquire more customers. We use the telecom customer data set in this project to predict the possibility of customers churning out._


## Exploratory Data Analysis

### Important Visualisations

_Here we highlight some of the predictor variables and the positive/negative influence they inflict upon the customer decision to churn/stay._

_First of all, we start with the geographic location of the customers and we are interested in knowing how their location plays a role in satisfying their network needs. The below figure shows the relative percentage of customers that churned out in all the 50 states and in the national capital of USA._

![01](./Customer_churn_identification/Plots/state_vs_churn.png)

_Since our target variable has a class imbalance problem, the data plotted as it is would not be so clear. Hence the churning out of customers is expressed as a relative percentage of the total number of customers from the respective state. As it is evident from the above bar chart, both the states of California and New Jersey have the highest percentage (26% approx.) of customers churning out. In layman&#39;s terms, it can be said that if a particular customer is from California or New Jersey, there is a pretty good chance that he will not be happy with the service provided. This can be because of various reasons like poor network coverage, inadequate customer service centres in customer locality etc.,         _

_Simple barplots depicting the satisfactory level of customers who have opted for international plan and Voice mail plan are given below:_

![02](./Customer_churn_identification/Plots/intplan_vs_churn.png)
![03](./Customer_churn_identification/Plots/vmail_vs_churn.png)

_Again, the bars are scaled with relative percentages. We can infer from the figures that the percentage of the customers who opted for voice mail plan and churned out is relatively lower that the customers who got out without experiencing voice mail service. On the contrary, A significant fraction of customers who opted for international plan churned out of the network (more than 40%). Simply speaking, if a customer is found to be on the verge of churning out, the service provider can consider providing voice mail service to him/her for a trail period free of cost. Also, the company should review the facilities provided as part of international plan as it turns out to be infamous._

Having analysed the categorical variables, we shall now turn our attention towards the tariff details which take up the form of continuous predictor variables. We have plotted the respective boxplots for the tariff paid by the customers for the calls they made during different timings of a day (daytime, evening and night) and the plot for daytime is given below as an example:

![04](./Customer_churn_identification/Plots/day_to_churn.png)

_We could see that the quartile ranges are significantly lifted above in case of day tariff although the difference is barely visible for tariffs corresponding to evening and night timings. As will be demonstrated in correlation analysis, the duration in mins spent during a call is directly proportional to the tariff (charged on a minutely basis) and so people who spend a lot of time over the phone are more likely to churn out because of the heavy tariff imposed on them._

_The variation of international call tariff with respect to international plan also reveals some information about the churning trend. we find that the boxplot distributions of international call tariff for customers who didn&#39;t opt for international plan is almost the same for churning/non-churning cases but once the international plan comes in as a factor, we find that the customers churned out as the tariff went up as shown below:_

![05](./Customer_churn_identification/Plots/int_vs_churn.png)

_This again highlights the point that whatever offers that are being provided under the international plan are not very efficient in keeping the customers satisfied._

![06](./Customer_churn_identification/Plots/vmail_mess_vs_churn.png)

_The above figure illustrates the distribution of voice mail messages sent by customers who have subscribed for voice mail plan. These boxplots emphasises the simple fact that those who have sent more number of voice mail messages would be charged more and hence it indirectly influences the churning decision of the customer._

_Before proceeding to the final analysis, it is necessary to point out a natural intuition that we get while looking at the relation between customer care calls and churning. If a customer calls the customer care numerous times, then it is safe to assume that he/she has grievances that needs to be addressed. Hence a high number (or abnormal number) of customer care calls indicate that the customer is reporting a lot of issues and hence becoming increasingly uncomfortable with the service._

![06](./Customer_churn_identification/Plots/cust_boxplot.png)
![07](./Customer_churn_identification/Plots/cust_calls_vs_churn.png)

_Having said this, if we look at the figures above, a simple boxplot of the total number of customer care calls made by the customer suggests that there are lot of outliers in the distribution. But the moment, we facet the plot using churn, the outliers are significantly reduced. This indicates that the outliers are indeed not outliers but valid data points providing information about the churning decision. Taking this into account along with the results of the ANOVA test (described in later chapters),_ **outlier removal is not done for this particular predictor variable.**


## Data Pre-Processing

### Outlier detection and removal

_Outliers are data points that differ significantly from the overall spread of the data. The presence of outliers in the variables degrade the quality of learning of the model as it distorts the weightage that is assigned to the predictor variables. Removal of outliers from the dataset is mandatory before using it for further analysis. We have already seen individual boxplots for predictor variables in the previous chapters which also indicates the presence of outliers_

_Observations that contain outliers can either be removed or imputed with suitable values. Our Dataset has a_ **peculiar target class imbalance** _which means that the number of cases available in our dataset where the customer actually churned out is extremely lower than the cases where the customer did not churn. Therefore, we cannot afford to lose more data which describes the minority class by removing outliers. Hence, we have opted for a combination of both._

### Missing value Imputation:

_Our Dataset now contains missing values which were induced as part of outlier analysis. Missing values impede the performance of the classification model that we are going to build on our dataset. Moreover, the various feature selection tests that follow data pre-processing cannot function properly if there are missing values in the dataset. Hence the missing values must imputed and there are several ways to do that. We can use the mean, median values of the predictor variables to fill up NAN values or we can use a more dynamic distance approach._

_For our dataset, we have used KNN (k Nearest Neighbours) imputation method which is a distance based method to find k – number of closest observations to the particular observation which contains missing values and uses the average of those k-values to impute the missing values._

_We have also tested the accuracy of all the three methods to confirm that KNN gives best results. An artificial NAN is induced at the 1000__th_ _value of &#39;total data charge&#39; variable and the imputed values are compared with the real value. KNN produces a value that is closest to the real value and hence it is used_

### Feature Selection

_The data acquired for resolving a particular problem statement may not always be fully relevant to the case in point. Since there may be multiple sources from where data is extracted, there is a good chance for irrelevant features to find their way in to the dataset. If a predictor variable that has no useful information to predict the outcome of a response variable is included in training set of our model, it can cause severe performance degradation of the model._

_Moreover, if two or more predictor variables included in the dataset contains the same information or in technical terms, highly correlated, then the redundant information can also impact the performance of our model. Hence it is a usual practice to subject the features to statistical tests like correlation analysis, chi-square and ANOVA tests to determine the level of contribution they make to the prediction of the response variable. We have carried out the following tests to determine the validity and relevance of the predictor variables._



#### **Correlation Analysis:**

_The following correlation plot illustrates the pairwise correlation between numerical variables of the dataset. The lower triangle and diagonal of the correlation matrix fed in to this heatmap are masked to avoid redundancy._

![08](./Customer_churn_identification/Plots/heat_map.png)

_The value of correlation coefficient ranges between +1 to -1, +1 being highly positively correlated and -1 being highly negatively correlated. As we can see from the above heatmap, number of minutes spent on the phone is strongly correlated with the respective charges which is obvious because the customers are charged on a minutely basis. In fact, there is a constant amount per minute fixed by the network company and the total charge is an exact multiple of this base amount._

_Therefore, as per the results of correlation test, we will_ **remove the attributes representing total minutes spent on the phone during day, evening and night and on international calls** _as charges incurred has more direct relation with customer churn._



#### **Chi-Square test:**

_Chi-square statistics examines the independence of two categorical vectors. By calculating the chi-squared statistic between a feature and the target vector, we obtain a measurement of the independence between the two. If the target is independent of the feature variable, then it is irrelevant for our purposes because it contains no useful information for our model. On the other hand, if the two variables are highly dependent, it is very likely that the predictor is useful for our model._

_From the Chi-square test, we have found out the_ **area code of the customer is irrelevant for our model** _as it contains no useful information and so it is removed._

#### **ANOVA test:**

_Analysis of Variance (ANOVA) test gives us the F-statistic value which is a measure of independence between a numerical attribute and the target vector. F-value scores examine if, when we group the numerical attribute by the classes of the target variable, the means for each group are significantly different. In other words, if there is high variance between those groups, then we can conclude that the feature has some pattern which the target classes follow._

_The results of the ANOVA test were significantly useful in simplifying the dataset as it removes a bunch of unnecessary variables as below._

1. _Phone number of the customer_
2. _Number of calls made by the customer during various times of the day (since the only relevant information is the duration of the calls or the charge incurred)_
3. _Customer account length_

**Important Note:** _our assumption regarding the validity of the customer service calls attribute is further supported by ANOVA test. When the variable was subjected to outlier analysis and then fed to ANOVA, the test rejected the variable. But when the variable was fed as it is to the test, it considered the variable to contain useful information for the target._

### Feature Scaling

_Our dataset contains numerical values that are measured in different scales. This makes the range of each variable disproportionate to each other. For example, the call tariffs may be measured in terms of U.S dollars ranging from 0 to say 100 dollars whereas the customer service calls are just discrete numbers having a much lower range. Considering them with their different unit scales as they are, will result in significant imbalance while building our model. The model will be inclined towards the variable that has higher range of values._

_Therefore it is necessary to apply corrective measures to the variable scales so that they fall under a universal scale. For our dataset, we have normalised the variables to convert their range so that their values fall within a common limit (between 0 and 1).

### Model Selection

_As we have already defined in our problem statement, the main goal of this project is to predict whether the customer would churn out or not and hence, our problem statement falls under the category &quot;classification&quot;. We can choose a variety of machine learning algorithms for building a binary classification model. For this project, the following 4 algorithms are chosen and their performance is evaluated on the same train and test data to select the best one of them._

1. _Logistic Regression_
2. _Naïve Bayes_
3. _Decision Tree_
4. _Random Forest_

#### Performance Metrics

_We have developed four models so far to predict the possibility of a customer churning out. In order to select the best suitable model, we have to use some sort of performance metrics which can be used as a touchstone to compare the performance of different models. In case of classification problems, the performance level is generally determined in terms of combinations of any of the following measures_

1. **Accuracy** _– percentage of the total observations correctly classified by the model_
2. **True positive rate** _– fraction of total positive cases correctly classified as positive by the model_
3. **True Negative rate** _– fraction of  total negative cases correctly classified as negative by the model_
4. **False Positive rate** _- fraction of  total negative cases misclassified as positive by the model_
5. **False negative rate** _- fraction of  total positive cases misclassified as negative by the model_

_For any classification models, a decent level of accuracy is expected and that is a direct measure of the model performance. But in our case, in addition to accuracy, we also have to make sure that no customer who is likely to churn out is missed by our model. Therefore, we have to try and reduce_ **False Negative Rate** _(FNR) as far as possible (customer churning out is the positive case of our model although in practice, this event is not positive). An alternative parameter to FNR is_ **sensitivity** _and is a measure of how sensitive our model is, towards predicting the positive class. Sensitivity is calculated as follows:_

_                                                 Sensitivity (%) = (1 – FNR) \* 100_

_Hence, the primary focus of the model building stage is to build models with high accuracy and high sensitivity (or low false negative rate). We also have to keep in mind, the_ **severe class imbalance problem** _that our dataset is suffering from. The total number of observations available for the customers who did not quit is enormous compared to the ones where the customers did churn out. A simple table containing the respective proportions illustrate this fact._

_This indicates that more than 84% of the observations relate to the customers not churning out (represented by numeric code 1) and a mere 15.6% of the data provides information about unsatisfied customers who churned out. If we were to blindly classify all the customers as belonging to the non-churning category in the training dataset, without considering any information, we would still be right about 84.4% of the time due to the class imbalance. This is called the_ **no information rate** _and we will have kept this parameter as our baseline while evaluating accuracy of various models. The different ways to overcome class imbalance problem are also discussed later in this chapter.  _

# Addressing the target class imbalance problem

_Before proceeding with model selection, we have to resolve the severe class imbalance problem that is found in the training dataset. As mentioned before, the ratio between observations available for positive and negative classes is hugely disproportionate. This causes the model to be biased and it will be inclined towards to majority class of the target variable. Therefore it is necessary to rectify this imbalance and the methods that can be used for the achieving balance are described below.

1. **Under-sampling the majority class:**

_This method removes random samples from the majority class so that the count is reduced and made equal to the minority class. This method is prone to information loss as a lot of samples containing useful information are disregarded._

2. **Over-sampling the minority class:**

_This method duplicates the observations belong to the minority class and inserts them randomly into the dataset so that the ratio of observations become balanced._

3. **Combination of under and over sampling:**

_This method includes a combination of both i.e., random under-sampling of Majority classes and over-sampling of minority classes_

## Conclusion

_From the above detailed portrayal of the performance statistics of individual models, it is evident that_ **a combination of both under and over sampling** _of the training data significantly improves the sensitivity although there is a small decrease in accuracy and the overall trade-off is advantageous. Out of the two models,_ **Decision Tree** _meets the mandatory criteria of achieving an accuracy level of 87.5% which is more than the no information rate of 86.56%. The sensitivity is also significantly improved from 60% to 73.6%. Hence Decision Tree can be selected for predicting the possibility of customer churning for future test cases and the same can be used for deployment._

_Important Note:_ _this is just a brief overview of the project which helps in highlighting the key factors at a superficial level. It is highly recommended that the viewers kindly go through the detailed project report which portrays the performance metrics and the code used in a very elaborate manner._