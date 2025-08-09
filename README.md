# Welcome to the repository!

## Files description<br>

In the repository you can find all the files that were either used or required in the process of analysis. This includes:<br><br>


[*README.md*](./README.md)

Serves as a general introduction into the project outlining the requirements, manual and other types of useful information.<br><br>

[*credit1.ipynb*](./credit1.ipynb)

Main file containing the first part of the analysis of the Home Credit dataset.<br><br>

[*credit2.ipynb*](./credit2.ipynb)

Main file containing the second part of the analysis of the Home Credit dataset.<br><br>

[*credit3.ipynb*](./credit3.ipynb)

Main file containing the third part of the analysis of the Home Credit dataset.<br><br>

[*requirements.txt*](./requirements.txt)

The file containing a detailed list of all the requirements for the use of code on a local machine.<br><br>

[*support_file.py*](./support_file.py)

The file containing all the functions used in the analysis.<br><br>

[*model.json*](./model.json)

The model used in the best prediction used in deployment

## Manual and Implementation<br>

The project would be beneficial to anyone interested in Data Transformation, Machine Learning and Banking. 

As the programme has been written with efficiency and accuracy in mind, so all the calculations are done using vectorization.<br><br>

## Requirements

The user is advised to have them all downloaded on a local machine, as, without them, the functionality of the programme would be limited.

For the full list of requirements, please consult [*requirements.txt*](./requirements.txt)<br><br>

## Limitations and areas for improvement

I took extensive steps to prevent overfitting and to make sure to keep the analysis as close to real life as possible, in a sense that I did not use the rows from all the 'child datasets' if the ID was mentioned in 'application_test', which has really lowered my training data pool, which lowers my Kaggle score much more than it would have, as, as proven in the project, the data in training and testing sets is different, and there is no doubt about it. Additionally, even though Kaggle does not display it directly, because the validation and the testing model have achieved similar results, we can claim that the final model is able to identify 80-85% of clients with payment issuers. Speaking of things that could have potentially improved the performance, I am certain that the feature engineering part could have been done better, and not because I did too little, but because it is very much possible that some features could have had high synergy with features in other datasets but there is a massive chance that those would not make it through the first-wave selection process
