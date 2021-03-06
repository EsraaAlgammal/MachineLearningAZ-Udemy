#Data Preprocessing

#Importing the dataset
dataset <- file.choose()
dataset <- read.csv(dataset)

dataset = read.csv('50_Startups')
#dataset = dataset[, 2:3]

# Encoding categorical data
dataset$State = factor(dataset$State,
                           levels = c('New York', 'California', 'Florida'),
                           labels = c(1, 2, 3))

# Splitting the data set into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8) #for training set 80%
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# [, 2:3 ] is excluding the country and purchased. Note cols in R starts with 1 not 0
# training_set[, 2:3] = scale(training_set[, 2:3]) 
# test_set[, 2:3]= scale(test_set[, 2:3])

# Fitting the Muliple Linear Regression to the Training set
#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spent + State)
regressor = lm(formula = Profit ~ .,
               data = training_set)

#
#regressor = lm(formula = Profit ~ R.D.Spend,
           #    data = training_set)"""


# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Building the optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
              data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)