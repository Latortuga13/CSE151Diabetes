## Milestone 3:

1. To finish the **major preprocessing** step, we first dropped the columns we decided to exclude in Milestone 2. We then encoded categorical features such as "RaceEthnicity", "Sex" and "GeneralHealth" and "AgeCategory" using one-hot, ordinal (for GeneralHealth) and parsing to keep just the lower bound of the age group each patient belonged to. We used minmax scaling to scale both Age and BMI. Our reasoning was that doing so would make it easier for us to compare the weights generated from our model for each feature afterwards. We decided not to use feature expansion since most of our data is already binary (and typically of the format "has a condition or doesn't")

2. We trained our first model using a basic LogisticRegressor on our data to predict whether or not a patient had diabetes. We first decided to produce a prediction without removing the rows that had other values for 'HadDiabetes' than yes or no for the purposes of comparison. Then, we removed those rows from our dataframe and ran the regressor again. Later, we also ran our model again with oversampled training data to account for the imbalance between HadDiabetes occurences in our dataset

3. When evaluating our model, we obtained .86 accuracy with 0.83, 0.86, 0.83 weighted avg precision, recall and f1 score respectively for our test data. When we evaluated the model again on our training data, we obtained identical values for our training data (though the precision for No was slightly better on our training data).

4. Based on the near-negligible differences between our model performance (or predictive error) on our training data and test data, we concluded that our model is likely underfitted (though we acknowledge the sheer size of our data set makes it hard to know the scale with which our data is underfitted). For our next model, we're considering using a SVM to yield a more complex decision boundary, and put us in the ideal complexity range where our model is both fairly accurate and generalizable.

6. Our conclusion of our 1st model is that we're underfitting. To improve this we are considering using SVM for our second model to achieve better complexity (we can control C) and accuracy, since we think our predictive error should still to be too high (based on where we are in the fitting graph). 
