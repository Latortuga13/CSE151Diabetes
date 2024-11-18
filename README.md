## Milestone 3:
The first changes we made was to finish the preprocessing of our data. This includes dropping the columns mentioned in Milestone 2, changing text based columns into numerical data, and scaling columns BMI and Age (which we also processed to be the lower bound of a patient's age group). Our justification for scaling BMI and Age was to make it easier to compare the weights generated from our model for each feature afterwards. 

After preprocessing we began to train our model and make adjustments to our model as seemed fit. After our first model test, we realized that there are more results in the 'HasDiabetes' column than just 'yes' and 'no'. This prompted us to adjust our data by removing the data that does not feature a simple 'yes' or 'no' response. After our second model test, the main problem that arose was that our model has a low recall score for 'yes'. We believed this is due to our training data having a low amount of data where 'HasDiabetes' is true, and thus in our next test adjusted our training data to include more entries that do have diabetes. This results in our final test having increased prevision and recall for 'yes' but an overall decrease in accuracy. \

We then used our model to predict using the training data to identify where our model would likely be on the fitting graph based on the difference in predictive error (obtained from classification report) between the training and test data sets.
