## Load Imports and data  step ##
library(dplyr) # for Data manipulation
library(glmnet) # This is the important one for RR and L Regression
library(caret) # To split data and preprocessing
library(ggplot2) # To visualise/graphs

data <- read.csv("Life-Expectancy-Data-Updated.csv")  #Updated dataset

## Pre-processing step ##
data$Economy_Status <- as.factor(ifelse(data$Economy_status_Developed == 1, "Developed", "Developing")) #Here we make it a single binary variable

data_cleaned <- data %>% select( -Economy_status_Developed, -Economy_status_Developing) # remove the two columns

response <- data_cleaned$Life_expectancy # Response variable is defined

predictors <- data_cleaned %>% select( -Country, -Life_expectancy) # Predictors are defined

# Check dimensions sanity check
length(response)  # Should be n = 2864
dim(predictors)  # Should be n = 2864 p = 18

## Test and Train split ##
set.seed(117) # So data vals can be recreated

train_index <- createDataPartition(response, p = 0.8, list = FALSE) #80/20 split
#train data group
train_data <- predictors[train_index, ]
train_response <- response[train_index]
#test data group
test_data <- predictors[-train_index, ]
test_response <- response[-train_index]

## matrices and scaling ##
# Convert to model matrix for both data sets
x_train <- model.matrix(~ ., data = train_data)[, -1]
x_test <- model.matrix(~ ., data = test_data)[, -1]
y_train <- train_response #also do the response

# Standardise the predictors to make sure all the same scale
preproc <- preProcess(x_train, method = c("center", "scale"))
x_train_std <- predict(preproc, x_train)
x_test_std <- predict(preproc, x_test)

## Fitting both model types ##
# Fit ridge model (alpha = 0)
ridge_model <- cv.glmnet(x_train_std, y_train, alpha = 0, nfolds = 10) #standard kfold cross validation at 10, also done at 5 and 20
best_lambda_ridge <- ridge_model$lambda.min
lambda_1se_ridge <- ridge_model$lambda.1se

# Fit lasso model (alpha = 1)
lasso_model <- cv.glmnet(x_train_std, y_train, alpha = 1, nfolds = 10) #standard kfold cross validation at 10, also done at 5 and 20
best_lambda_lasso <- lasso_model$lambda.min
lambda_1se_lasso <- lasso_model$lambda.1se

cat("Ridge lambda.min:", best_lambda_ridge, "and lambda.1se:", lambda_1se_ridge, "\n")
cat("Lasso lambda.min:", best_lambda_lasso, "and lambda.1se:", lambda_1se_lasso, "\n")

## Add Ordinary Least Squares due to not many p ##
#ensure that the columns are correct for OLS
train_df <- as.data.frame(x_train_std)
colnames(train_df) <- colnames(x_train_std)
# Fit Least Squares model due to not many p
#least_squares_model <- lm(y_train ~ x_train_std)
least_squares_model <- lm(y_train ~., data = train_df)

## Evaluation and root mean squared error ##
ridge_pred <- predict(ridge_model, s = best_lambda_ridge, newx = x_test_std)
lasso_pred <- predict(lasso_model, s = best_lambda_lasso, newx = x_test_std)
#add in OLS comparison
test_df <- as.data.frame(x_test_std)
colnames(test_df) <- colnames(x_train_std)
least_squares_pred <- predict(least_squares_model, newdata = test_df)

# Calculate Root Mean Squared Error (RMSE)
rmse_ridge <- sqrt(mean((test_response - ridge_pred)^2))
rmse_lasso <- sqrt(mean((test_response - lasso_pred)^2))
#add ols comparison
rmse_least_squares <- sqrt(mean((test_response - least_squares_pred)^2))


cat("\n--- COMPARISON OF ALL THREE MODELS ---\n")
cat(sprintf("OLS (Least Squares) RMSE: %.4f\n", rmse_least_squares))
cat(sprintf("Ridge RMSE:               %.4f\n", rmse_ridge))
cat(sprintf("Lasso RMSE:               %.4f\n", rmse_lasso))

# Check OLS coefficients for sanity of model
#least_squares_coefs <- coef(least_squares_model)
#print(least_squares_coefs)

# Check the range of coefficients
#cat("Min Coef:", min(least_squares_coefs), "\n")
#cat("Max Coef:", max(least_squares_coefs), "\n")

## ranking for all countries in dataset ##
x_all <- model.matrix(~ ., data = predictors)[, -1]
x_all_std <- predict( preproc, x_all)

#make the predictions via each RR and Lasso regression
data_cleaned$Pred_Ridge <- predict( ridge_model, s = best_lambda_ridge, newx = x_all_std)
data_cleaned$Pred_Lasso <- predict( lasso_model, s = best_lambda_lasso, newx = x_all_std)

# Collect the country average of the time period the data covers, so 2000- 2015
country_avg_ridge <- data_cleaned %>%
  group_by(Country) %>% #here we group by the column country
  summarise(Avg_Pred_Ridge = mean(Pred_Ridge, na.rm = TRUE)) %>% #take the mean of them
  arrange(desc(Avg_Pred_Ridge))

country_avg_lasso <- data_cleaned %>%
  group_by(Country) %>% #here we greoup by the column country
  summarise(Avg_Pred_Lasso = mean(Pred_Lasso, na.rm = TRUE)) %>% #take the mean of them
  arrange(desc(Avg_Pred_Lasso))

# Results for top 16 (it makes sense due to placement of Denmark as 16 in Lasso)

cat("\n ## Top 16 Life expectancy using ridge regression ## \n") #Must be cat? print doesnt work
print(head(country_avg_ridge, 16))

cat("\n ## Top 16 Life expectancy using lasso regression ## \n")
print(head(country_avg_lasso, 16))

## Plots and some visualisation of the data ##

par( mfrow = c(1, 2), mar = c(4, 4, 5, 2))

plot(ridge_model, xvar = "lambda", label = TRUE, main = "Ridge Coefficients")
plot(lasso_model, xvar = "lambda", label = TRUE, main = "Lasso Coefficients")

par( mfrow = c(1, 1 ))
# Cross validation error vs tuning parameter lambda
ridge_df <- data.frame(
  lambda = ridge_model$lambda,
  error = ridge_model$cvm,
  model = "Ridge"
)

lasso_df <- data.frame(
  lambda = lasso_model$lambda,
  error = lasso_model$cvm,
  model = "Lasso"
)

# Create the plot for Ridge first (Blue)
plot(
  x = ridge_df$lambda,
  y = ridge_df$error,
  log = "x",                  # Log scale for X-axis as it looks wack otherwise
  type = "l",
  col = "blue",
  lwd = 2,
  xlim = c(.1, 10000),
  xlab = "Tuning Parameter (lambda)",
  ylab = "True Prediction Error (MSE)",
  main = "Prediction Error vs. Lambda"
)

#can add Lasso Curve as red on plot
lines(lasso_df$lambda, lasso_df$error, col = "red", lwd = 2)

#Place legend in the plot
legend("bottomright",
       legend = c("Ridge", "Lasso"),
       col = c("blue", "red"),
       lty = c(1, 1, 1),
       lwd = c(1, 2, 2))
