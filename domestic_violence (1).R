plot(ndataset_Sheet1$poverty, ndataset_Sheet1$gii,
     main = "Scatter Plot: gii vs Poverty",
     xlab = "poverty",
     ylab = "gii",
     col = "blue",
     pch = 19)


plot(ndataset_Sheet1$literacy, ndataset_Sheet1$gii,
     main = "Scatter Plot: gii vs literacy",
     xlab = "literacy",
     ylab = "gii",
     col = "blue",
     pch = 19)


plot(ndataset_Sheet1$dindex, ndataset_Sheet1$gii,
     main = "Scatter Plot: dindex vs Poverty",
     xlab = "dindex",
     ylab = "gii",
     col = "blue",
     pch = 19)

plot(ndataset_Sheet1$uhssci, ndataset_Sheet1$gii,
     main = "Scatter Plot: gii vs uhssci",
     xlab = "uhssci",
     ylab = "gii",
     col = "blue",
     pch = 19)

plot(ndataset_Sheet1$gpi, ndataset_Sheet1$gii,
     main = "Scatter Plot: gpi vs Poverty",
     xlab = "gpi",
     ylab = "gii",
     col = "blue",
     pch = 19)

plot(ndataset_Sheet1$chmarriage, ndataset_Sheet1$gii,
     main = "Scatter Plot: gii vs chmarriage",
     xlab = "chmarriage",
     ylab = "gii",
     col = "blue",
     pch = 19)

plot(ndataset_Sheet1$chmarriage, ndataset_Sheet1$poverty,
     main = "Scatter Plot: gii vs chmarriage",
     xlab = "chmarriage",
     ylab = "gii",
     col = "blue",
     pch = 19)


boxplot(ndataset_Sheet1$gii, ndataset_Sheet1$literacy, ndataset_Sheet1$poverty,ndataset_Sheet1$dindex,
        ndataset_Sheet1$uhssci, ndataset_Sheet1$religion, ndataset_Sheet1$gpi, ndataset_Sheet1$chmarriage, ylab = "gii",
        names = c("literacy", "poverty", "dindex","uhssci" , "religion", "gpi", "chmarriage"), 
        main = "Boxplots of Different Cofactors")

#AIC model

model <- lm(ndataset_Sheet1$gii ~ ndataset_Sheet1$religion + ndataset_Sheet1$literacy + ndataset_Sheet1$poverty + ndataset_Sheet1$dindex + ndataset_Sheet1$uhssci + ndataset_Sheet1$gpi + ndataset_Sheet1$chmarriage, data = ndataset_Sheet1)
stepwise_model <- step(model, direction = "both")
summary(stepwise_model)


# new test
# forward

null_model <- lm(gii ~ 1, data = ndataset_Sheet1)

full_model <- lm(gii ~ religion + literacy + poverty + dindex + uhssci + gpi + chmarriage, data = ndataset_Sheet1)
summary(full_model)
forward_pval <- function(data, response, predictors, alpha = 0.05) {
  selected <- c()  
  remaining <- predictors
  while (length(remaining) > 0) {
    pvals <- sapply(remaining, function(predictor) {
      formula <- as.formula(paste(response, "~", paste(c(selected, predictor), collapse = " + ")))
      model <- lm(formula, data = ndataset_Sheet1)
      summary(model)$coefficients[2 + length(selected), 4]  # p-value of the last added predictor
    })
    best_predictor <- names(which.min(pvals))
    if (pvals[best_predictor] < alpha) {
      selected <- c(selected, best_predictor)
      remaining <- setdiff(remaining, best_predictor)
    } else {
      break
    }
  }
  final_formula <- as.formula(paste(response, "~", paste(selected, collapse = " + ")))
  lm(final_formula, data = ndataset_Sheet1)
}

predictors <- c("religion", "literacy", "poverty", "dindex", "uhssci", "gpi", "chmarriage")
model <- forward_pval(data = ndataset_Sheet1, response = "gii", predictors = predictors)
summary(model)

# new backward

# Backward elimination function
backward_pval <- function(data, response, predictors, alpha = 0.05) {
  selected <- predictors
  while (TRUE) {
    formula <- as.formula(paste(response, "~", paste(selected, collapse = " + ")))
    model <- lm(formula, data = ndataset_Sheet1)
    
    pvals <- summary(model)$coefficients[-1, 4] 
    
    max_pval <- max(pvals)
    if (max_pval > alpha) {
      predictor_to_remove <- names(which.max(pvals))
      selected <- setdiff(selected, predictor_to_remove)
    } else {
      break 
    }
  }
  
  final_formula <- as.formula(paste(response, "~", paste(selected, collapse = " + ")))
  lm(final_formula, data = ndataset_Sheet1)
}

response <- "gii"
predictors <- c("religion", "literacy", "poverty", "dindex", "uhssci", "gpi", "chmarriage")

final_model <- backward_pval(data = ndataset_Sheet1, response = response, predictors = predictors)

summary(final_model)

AIC(stepwise_model,final_model)

#multicolinearity
install.packages("car")
library(car)
vif(stepwise_model)

#homoscedasticity
plot(fitted(final_model), residuals(final_model),
     xlab = "Fitted Values",
     ylab = "Residuals",
     main = "Residuals vs Fitted Values")
abline(h = 0, col = "red")

#QQplot normality
qqnorm(residuals(final_model))
qqline(residuals(final_model), col = "red")

#histogram
hist(residuals(final_model), 
     main = "Histogram of Residuals", 
     xlab = "Residuals", 
     col = "lightblue", 
     border = "black")


res <- residuals(final_model)

hist(res, 
     probability = TRUE, 
     main = "Histogram of Residuals with Normal Curve", 
     xlab = "Residuals", 
     col = "lightblue", 
     border = "black")

lines(density(res), col = "red", lwd = 2)
curve(dnorm(x, mean = mean(res), sd = sd(res)), 
      col = "darkblue", lwd = 2, add = TRUE)


# tests for normality

shapiro.test(residuals(final_model))
residuals_model <- residuals(final_model)
ks_test <- ks.test(residuals_model, "pnorm", mean = mean(residuals_model), sd = sd(residuals_model))
print(ks_test)


#prediction

set.seed(123)
sample_index <- sample(1:nrow(ndataset_Sheet1), 0.8 * nrow(ndataset_Sheet1))
train_set <- ndataset_Sheet1[sample_index, ]
test_set <- ndataset_Sheet1[-sample_index, ]

model1 <- lm(gii ~ religion + poverty + uhssci + gpi + chmarriage, data = train_set)

model2 <- lm(gii ~ uhssci + gpi + chmarriage, data = train_set)

predictions_model1 <- predict(model1, newdata = test_set)
predictions_model2 <- predict(model2, newdata = test_set)

actual_values <- test_set$gii

plot(actual_values, type = "l", col = "black", lwd = 2, 
     xlab = "Observation", ylab = "GII", 
     main = "Model Predictions vs. Observed Values")
points(predictions_model1, col = "blue", pch = 16)
points(predictions_model2, col = "red", pch = 1)
legend("topright", legend = c("Observed", "Model 1", "Model 2"),
       col = c("black", "blue", "red"), lty = c(1, NA, NA), 
       pch = c(NA, 16, 1))

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

mae <- function(actual, predicted) {
  mean(abs(actual - predicted))
}

rmse_model1 <- rmse(actual_values, predictions_model1)
rmse_model2 <- rmse(actual_values, predictions_model2)

mae_model1 <- mae(actual_values, predictions_model1)
mae_model2 <- mae(actual_values, predictions_model2)

cat("Model 1 - RMSE:", rmse_model1, "MAE:", mae_model1, "\n")
cat("Model 2 - RMSE:", rmse_model2, "MAE:", mae_model2, "\n")

#test
ndataset_Sheet1$income_group <- cut(ndataset_Sheet1$poverty,
                                    breaks = c(-Inf, 20, 40, Inf),
                                    labels = c("High Income", "Middle Income", "Low Income"))

pairwise_results <- pairwise.wilcox.test(ndataset_Sheet1$gii, 
                                         ndataset_Sheet1$income_group, 
                                         p.adjust.method = "bonferroni")
print(pairwise_results)

boxplot(gii ~ income_group, data = ndataset_Sheet1,
        main = "GII Across Income Groups",
        xlab = "Income Group", ylab = "GII",
        col = c("skyblue", "lightgreen", "pink"))












