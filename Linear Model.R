# Load required libraries
library(caret)  # For cross-validation/pre-processing
library(MASS)   # For stepwise regresson
library(ggplot2)
set.seed(42)

################################################################################
#                            Data Load and Setup                               #
################################################################################

# If the data won't load, fill in source with the global path to your data file
source <- ""

original_df <- read.csv(paste(source, "NCAA_mbb_matchup_metrics_2024.csv", sep=''))

# Create a new column for the score difference (target variable)
original_df$score_diff <- original_df$home_team_score - original_df$away_team_score

# Define metadata and column categories
meta_columns <- c("date", "home_team", "away_team", 
                  "home_team_score", "away_team_score", 
                  "neutral_game", "score_diff")

# Separate ranking-related and non-ranking predictors
rank_predictors     <- grep("_r$", names(original_df), value = TRUE)
non_rank_predictors <- setdiff(
    names(original_df), 
    c(rank_predictors, meta_columns, "winner")
)

# Subset data for rank and non-rank predictors
rank_df     <- original_df[, c(meta_columns, rank_predictors)]
non_rank_df <- original_df[, c(meta_columns, non_rank_predictors)]

# Exclude specific columns not intended for modeling
exclude_columns <- c("date", "home_team", "away_team", 
                     "home_team_score", "away_team_score", 
                     "score_diff")

# Select predictors for non-ranking model
predictor_columns_non_rank <- setdiff(names(non_rank_df), exclude_columns)

# Create a formula for regression with non-ranking predictors
non_rank_formula <- reformulate(
    predictor_columns_non_rank, 
    response = "score_diff"
)


################################################################################
#                          Preprocessing and Split                             #
################################################################################


# Handle missing values and scale predictors using caret
# -> Calculate mean and std.dev for scaling
preprocess_model <- preProcess(non_rank_df[, predictor_columns_non_rank], 
                               method = c("center", "scale"))

# -> Apply the transformations to all the raw metric data
non_rank_df_scaled <- predict(preprocess_model, non_rank_df)

# Split data into training and testing sets for cross-validation
train_index   <- createDataPartition(
    non_rank_df_scaled$score_diff, 
    p = 0.8, 
    list = FALSE
)
train_data    <- non_rank_df_scaled[train_index, ]
test_data     <- non_rank_df_scaled[-train_index,]


################################################################################
#                                Train Model                                   #
################################################################################


# Fit the initial (full) model
initial_model <- lm(non_rank_formula, data = train_data)

# Evaluate model performance on training set
summary(initial_model)

# Backward stepwise selection
backward_model <- stepAIC(initial_model, direction = "backward", trace=0)

print(summary(backward_model))

# RUn model on test set
test_predictions <- predict(backward_model, newdata = test_data)
test_actual      <- test_data$score_diff

# R-squared on test data
r_squared <- cor(test_actual, test_predictions)^2
cat("Test R-squared:", r_squared, "\n")

# predicted vs actual scores (run by hand if sourcing file)
ggplot(data.frame(Predicted = test_predictions, Actual = test_actual), 
       aes(x = Actual, y = Predicted)) +
    geom_point(color="pink") +
    geom_abline(color = "blue") +
    labs(title = "Predicted vs Actual Score Differences",
         x = "Actual Score Difference",
         y = "Predicted Score Difference") +
    theme_minimal()


# Add residual diagnostics
par(mfrow=c(2,2))
plot(backward_model)

# Calculate RMSE
rmse <- sqrt(mean((test_predictions - test_actual)^2))
cat("RMSE:", rmse, "\n")

# Calculate MAE
mae <- mean(abs(test_predictions - test_actual))
cat("MAE:", mae, "\n")

# Prediction intervals
pred_intervals <- predict(backward_model, 
                          newdata = test_data,
                          interval = "prediction",
                          level = 0.95)

print(pred_intervals)
