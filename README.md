
# Admissions-Predictions
This projecct describes the process whether the students are eligible for admission criteria based on the random data created with the help of python 


Here in this code the admission criteria is set to be this :- 

admission_prob = 1 / (1 + np.exp(-(0.1*X1 + 0.2*X2 - 10))) // this line indicates the selection process for the admission of the students

# this is a small piece of code in python as an example to this formula

import numpy as np

# Given values
X1 = 54.88135039
X2 = 59.28802708

# Calculate linear combination
z = 0.1 * X1 + 0.2 * X2 - 10

# Calculate admission probability
admission_prob = 1 / (1 + np.exp(-z))

print("Linear combination (z):", z)
print("Admission probability:", admission_prob)


# output to this code will result to 

Linear combination (z): 7.345740455
Admission probability: 0.999357686


So, for this student with an entrance exam score of 54.88135039 and a percentage of 59.28802708, the calculated probability of admission is approximately 0.999354, or 99.94%.

This high probability indicates that the student is very likely to be admitted according to the logistic regression model.

likewise the auto generated dataset is used to predict with output of actual values and after model implementation values i.e predicted output.

