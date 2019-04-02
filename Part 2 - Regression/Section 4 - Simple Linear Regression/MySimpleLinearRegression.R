#Data Preprocessing

#Importing the dataset
dataset = read.csv('Salary_Data.csv')
dataset = dataset[, 2:3]

#This needs revisit as I'm afraid I've changed the values accidentally 
# Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))

# Splitting the data set into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 2/3) #for training set 2/3
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# [, 2:3 ] is excluding the country and purchased. Note cols in R starts with 1 not 0
# training_set[, 2:3] = scale(training_set[, 2:3]) 
# test_set[, 2:3]= scale(test_set[, 2:3])

# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,  #means salary is proportional to the YearsExperience
                data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Visualizing the Training set results
# install.packages('ggplot2')

library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience), y = predict(regressor, newdata = training_set),
            colour = 'blue') + 
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of Experience') +
  ylab('Salary')

# Visualizing the Training set results
# install.packages('ggplot2')

library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience), y = predict(regressor, newdata = training_set),
            colour = 'blue') + 
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of Experience') +
  ylab('Salary')















