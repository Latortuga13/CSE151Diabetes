## Milestone 2:
After analysis of the pairplot we generated for our data, we plan to preprocess our data by dropping patient ID, Height, Weight, State, Smoker and Ecig status and TetanusLast10Tdap. Our reasoning for this is as follows: Patient ID is just a label hospitals use for logistics and should bear no effect on a patient's likelihood of having had diabetes. Height and Weight alone can be deceptive, so BMI should better encapsulate the tendency for obesity (as it accounts for height affects on weight). We also thought that we should remove State as we are more interested in how the prior health conditions and health status affects diabetes, not in simply finding which states have higher incidence of diabetes. We dropped Smoker/Ecig status and TetanusLast10dap because they contained categorical data we didn't know how to meaningfully encode. We also decided to round down Age to the lowest value of the bucket into which patients were placed (as opposed to encoding it).

We will able be processing the HadDiebetes column since it is in string format with 4 possible responses. Yes and No will be 1 and 0. Any rows with the options Yes, but only during pregnancy (female) and No, pre-diabetes or borderline diabetes will be removed from our dataset. This is because these are outlier cases in our dataset and we are plan to predict whether people strictly have or do not have diabetes. Since our dataset is already large and these responses are only a small fraction of the responses we will still have a sufficient dataset to predict diebetes. All other columns, being binary variables (with Age and BMI being the only exceptions) are fairly descript and unneeding of further column description. We also figured we don't need to comment on scales for most of our data due to it being binary variables, though we are considering using minmax-scaling on Age and BMI to keep them all even. To get a picture of the data distributions, we found the mean for all columns (excluding those we drop).


## Milestone 3:
1. To finish the **major preprocessing** step, we first dropped the columns we decided to exclude in Milestone 2. We then encoded categorical features such as "RaceEthnicity", "Sex" and "GeneralHealth" and "AgeCategory" using one-hot, ordinal (for GeneralHealth) and parsing to keep just the lower bound of the age group each patient belonged to. We used minmax scaling to scale both Age and BMI. Our reasoning was that doing so would make it easier for us to compare the weights generated from our model for each feature afterwards. We decided not to use feature expansion since most of our data is already binary (and typically of the format "has a condition or doesn't")

2. We trained our first model using a basic LogisticRegressor on our data to predict whether or not a patient had diabetes. We first decided to produce a prediction without removing the rows that had other values for 'HadDiabetes' than yes or no for the purposes of comparison. Then, we removed those rows from our dataframe and ran the regressor again. Later, we also ran our model again with oversampled training data to account for the imbalance between HadDiabetes occurences in our dataset

3. When evaluating our model, we obtained .86 accuracy with 0.83, 0.86, 0.83 weighted avg precision, recall and f1 score respectively for our test data. When we evaluated the model again on our training data, we obtained identical values for our training data (though the precision for No was slightly better on our training data).

4. Based on the near-negligible differences between our model performance (or predictive error) on our training data and test data, we concluded that our model is likely underfitted (though we acknowledge the sheer size of our data set makes it hard to know the scale with which our data is underfitted). For our next model, we're considering using a SVM to yield a more complex decision boundary, and put us in the ideal complexity range where our model is both fairly accurate and generalizable.

6. Our conclusion of our 1st model is that we're underfitting. To improve this we are considering using SVM for our second model to achieve better complexity (we can control C) and accuracy, since we think our predictive error should still to be too high (based on where we are in the fitting graph). 

## Milestone 4:
1. We trained our second model using SVM on our data. We first ran it with the cleaned data from our previous milestone. However this became problematic as the classification report showed that no matter how we changed the regularization coefficient, the results would be similar across all fields (precision, recall, f1, accuracy). Thus we updated our data to use oversampling which yielded different results amongst different regularization coefficients. However the results were very similar to each other with very minor differences.

2. When evaluating our model with different regularization coefficients. We found that the best coefficient was when c=0.01. The accuracy for this test was 0.73690 and for the weighted average of precision, recall, and f1 score we yielded 0.85697, 0.73690, and 0.77304 respectively.

3. The predictive error between the test and training data were still very close, and thus we yielded that our model is still underfitting. To combat this, we determined that our next model should be a decision tree model as we believe it will help us to determine our data's more important features. We also determined that a decision tree model better suits our dataset due to the large variance of data types in our dataset.

4. After running the SVM with the oversampled data and best performing regularization coefficient, we used SKLearn's confusion matrix in order to get the True Negatives, False Positive, False Negatives, and True Positives. The end results were: True Negatives (TN) = 28996, False Positives (FP) = 10493, False Negatives (FN) = 1635, True Positives (TP) = 4980.

### Conclusion Milestone 4:
In conclusion, our model performed relatively the same as the first model. After observing our new model, the accuracy, precision, and recall were all similar from the results of our first model. The reason we believe this to be the case is due to the same problem of underfitting amongst both models. In order to improve this we could do more continuous testing on regularization coefficients. However we believe this change would only result in very minor differences from our current results as all of our current tests are still similar to each other. One other imrprovement we could make is to increase dimensionality by using the svc kernel. By doing this it potentially will better adjust the svm boundary to our complex dataset rather than our current linear kernel which may lose emphasis on more important traits.

### Introduction Milestone 5:
The modern grocery store is chock full of products that contain unhealthy amounts of refined sugars. Even products promoted as healthy alternatives routinely include big servings of sugar that make them addictive must-buys. Even though there is a genetic component to the onset of diabetes, lifestyle choices can greatly impact the possibility of developing especially type 2 diabetes. In light of this, we decided to build models using patient information data to predict whether a patient has or doesn't have diabetes. These models are important because they provide data-driven insights that empower better decision-making. By having a semi accurate predictor, it is possible to diagnose and help people who may suffer from the condition without knowing whether or not they have it. This knowledge from our project can help patients and people throughout the world who may struggle with a lack of knowledge about their own medical conditions.

### Methods Milestone 5:
##Data Exploration
For our data exploration portion of our project, we first generated a pair plot to visualize the relationship of to help us rule which attributes we plan on dropping
and which to keep within our dataset. We ended up dropping the following attributes: patient ID, Height, weight, state, Smoker and Ecig status
and TetanusLast10Tdap. Additionally, we reviewed "HadDiabetes" target value and came to the conclusion to drop rows containing the options "yes, but only during pregnancy (female)" and "No, pre-diabetes or borderline diabetes"

##Preprocessing
within the preprocessing steps, we dropped the columns we mentioned in the data exploring and encoded the following categorical variables: Racethnicity and Sex (One
hot), GeneralHealth (ordinal), Agecategory (lower value). For scaling, we applied the MinMax scaling to Age and BMI and for Target Variable, we one hot encoded it.
Finally, we do recognize that there is a bit of data imbalance and as such, can be influence our results.

##Model 1
For our first model, our group decided to use a logistic regression. we first trained our model, without the values for HadDiabetes, removed any outliers within the
model, and in order to address the imbalance, we oversampled the underrepresented categories.

##Model 2
For our second model, we swapped over to using a Support Vector Machine.Here, we used our cleaned data (same as model 1) and varied the values of our one parmeter c. Similarly, we oversampled the underrepresented categories in order to address the imbalance

##Model 3
For our last model, we used a Decision Tree Model. Here, we trained the model using the same oversampled dataset used for the SM model and we experimented with
different depths and minimum samples to split a node.

### Discussion Milestone 5:

Our initial data preprocessing strategy was guided by the principle of Occam's razor, which suggests that simpler models are preferable. However, upon reflection, this approach may have been overly simplistic. The dismissal of categorical variables like Smoker/Ecig status might have been premature, which could capture the nuanced relationships within these features.

Our choice of LogisticRegressor and SVM was influenced by their interpretability and established performance in binary classification tasks. However, these models may not have been the most suitable for our dataset's complexity and structure. The underfitting observed suggests that our models lacked the capacity to learn from the data effectively. We recognize that our model selection was overly conservative and did not fully leverage the potential of more complex models that can handle high-dimensional data and complex interactions, such as neural networks.

The class imbalance in our dataset was a significant challenge, and while oversampling was a straightforward solution, it may have led to overfitting. A more delicate approach, such as SMOTE, could have been employed to generate synthetic samples that better represent the minority class.

In light of the limitations and insights gained from our analysis, we propose several directions for future research. We will explore the use of neural network models, which are capable of capturing complex, non-linear relationships and can handle high-dimensional data more effectively. Additionally, we will investigate more advanced techniques for dealing with class imbalance. By addressing these areas, we aim to not only improve our model's predictive performance but also to gain a deeper understanding of the underlying patterns in the data.

### Conclusion Milestone 5:
This study investigated the use of data preprocessing, basic logistic regressors, and SVMs for diabetes prediction, utilizing the Data of Patients (For Medical Field) dataset from kaggle. Through our experimentation we realized the limitations of “simple” machine learning models as our logistic regressor and SMVs produced very similar results. We believe that with some modifications to our regularization coefficients in our logistic regressor or an increase in dimensionality in our SVC kernel we could slightly improve our results but our best chances at a better result is though using a more complex machine learning model such as neural networks. With the use of neural networks we could capture more complex relationships between our features which simpler models cannot. Some limiations we faced in our investigation is that our model is extremely skewed with data on negative results for diabetes. If our model was more evenly distributed we would likely have found better results with our SVMs and logistic regressors. We also faced some limitations with the computing power of our devices. We were unable to run our SVMs with higher dimensionality due to a lack of processing power to run the model in a reasonable amount of time. Future work on this investigation could focus on exploring deep learning models to mitigate our issues with the over negative samples in our dataset, aiming to further enhance our accuracy.

### Collaborative Statement Milestone 5:
We worked well as a group and did most of the project together. We would do a couple of group meetings and one call for each of the models and milestones. The roles of each member and their contributions are listed below.
Seth Chng: Programmer and writer
Wrote sections of the final paper and worked on both models. Collaborated to write the abstract for the project and find the dataset. 

Ken Vandeventer: Programmer and writer
Wrote sections of the final paper and worked on both models. Collaborated to write the abstract for the project and find the dataset.

Matthew Mizumoto: Programmer and writer
Wrote sections of the final paper and worked on both models. Collaborated to write the abstract for the project and find the dataset.

Jack Barkes: Programmer and writer
Wrote sections of the final paper and worked on both models. Collaborated to write the abstract for the project and find the dataset.

Victor Ku: Programmer and writer
Wrote sections of the final paper and worked on the second model. Collaborated to write the abstract for the project and find the dataset.

Lixing Shao: Programmer and writer
Wrote sections of the final paper and worked on the first model. Collaborated to help find the dataset.
