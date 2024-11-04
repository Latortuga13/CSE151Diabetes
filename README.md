After analysis of the pairplot we generated for our data, we plan to preprocess our data by dropping patient ID, Height, Weight, State, Smoker and Ecig status and TetanusLast10Tdap. Our reasoning for this is as follows: Patient ID is just a label hospitals use for logistics and should bear no effect on a patient's likelihood of having had diabetes. Height and Weight alone can be deceptive, so BMI should better encapsulate the tendency for obesity (as it accounts for height affects on weight). We also thought that we should remove State as we are more interested in how the prior health conditions and health status affects diabetes, not in simply finding which states have higher incidence of diabetes. We dropped Smoker/Ecig status and TetanusLast10dap because they contained categorical data we didn't know how to meaningfully encode. 

We will able be processing the HadDiebetes column since it is in string format with 4 possible responses. Yes and No will be 1 and 0. Any rows with the options Yes, but only during pregnancy (female) and No, pre-diabetes or borderline diabetes will be removed from our dataset. This is because these are outlier cases in our dataset and we are plan to predict whether people strictly have or do not have diabetes. Since our dataset is already large and these responses are only a small fraction of the responses we will still have a sufficient dataset to predict diebetes.
