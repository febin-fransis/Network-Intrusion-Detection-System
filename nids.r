
packages <- c("randomForest", "ggplot2", "gridExtra")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(randomForest)
library(ggplot2)
library(gridExtra)

data <- read.csv("sample_data/network_traffic.csv",
                 stringsAsFactors = FALSE)

if (!"label" %in% names(data)) {
  stop("Dataset must contain a column named 'label'.")
}

labels <- tolower(trimws(as.character(data$label)))
labels[labels %in% c("normal", "benign", "0")] <- "normal"
labels[labels != "normal"] <- "attack"

data$label <- factor(labels, levels = c("normal", "attack"))

features <- data[, names(data) != "label", drop = FALSE]

keep <- vapply(features, function(x) length(unique(x)) > 1,
               logical(1))
features <- features[, keep, drop = FALSE]

for (col in names(features)) {
  if (is.numeric(features[[col]])) {
    missing <- is.na(features[[col]])
    if (any(missing)) {
      features[[col]][missing] <-
        median(features[[col]], na.rm = TRUE)
    }
  } else {
    features[[col]] <- as.factor(features[[col]])
    features[[col]][is.na(features[[col]])] <-
      levels(features[[col]])[1]
  }
}

set.seed(123)

normal_rows <- which(data$label == "normal")
attack_rows <- which(data$label == "attack")

train_rows <- c(
  sample(normal_rows, floor(0.75 * length(normal_rows))),
  sample(attack_rows, floor(0.75 * length(attack_rows)))
)

x_train <- features[train_rows, , drop = FALSE]
x_test <- features[-train_rows, , drop = FALSE]

y_train <- droplevels(data$label[train_rows])
y_test <- factor(data$label[-train_rows],
                 levels = c("normal", "attack"))

cat("\nTraining NIDS model...\n")

model <- randomForest(
  x = x_train,
  y = y_train,
  ntree = 150,
  importance = TRUE
)

pred <- predict(model, newdata = x_test)

cm <- table(Actual = y_test, Predicted = pred)

accuracy <- sum(diag(cm)) / sum(cm)

precision <- if (sum(pred == "attack") > 0) {
  sum(pred == "attack" & y_test == "attack") /
    sum(pred == "attack")
} else {
  0
}

recall <- if (sum(y_test == "attack") > 0) {
  sum(pred == "attack" & y_test == "attack") /
    sum(y_test == "attack")
} else {
  0
}

cat("\n Network Intrusion Detection System RESULTS \n")
cat("Accuracy :", round(accuracy * 100, 2), "%\n")
cat("Precision:", round(precision * 100, 2), "%\n")
cat("Recall   :", round(recall * 100, 2), "%\n")
print(cm)

dir.create("outputs", showWarnings = FALSE)

metrics_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall"),
  Value = c(accuracy, precision, recall)
)

write.csv(metrics_df, "outputs/metrics.csv", row.names = FALSE)
write.csv(as.data.frame(cm),
          "outputs/confusion_matrix.csv", row.names = FALSE)

write.csv(
  data.frame(Actual = y_test, Predicted = pred),
  "outputs/predictions.csv",
  row.names = FALSE
)

traffic_df <- as.data.frame(table(data$label))
names(traffic_df) <- c("Traffic_Type", "Count")

p1 <- ggplot(
  traffic_df,
  aes(x = Traffic_Type, y = Count, fill = Traffic_Type)
) +
  geom_col() +
  labs(title = "Network Traffic Distribution",
       x = "Traffic Type", y = "Number of Records") +
  theme_minimal()

cm_df <- as.data.frame(cm)

p2 <- ggplot(
  cm_df,
  aes(x = Predicted, y = Actual, fill = Freq)
) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  labs(title = "Confusion Matrix",
       x = "Predicted Class", y = "Actual Class") +
  theme_minimal()

imp <- as.data.frame(importance(model))
imp$Feature <- rownames(imp)
imp$Importance <- imp[["MeanDecreaseGini"]]
imp <- imp[order(imp$Importance), , drop = FALSE]

write.csv(imp, "outputs/feature_importance.csv",
          row.names = FALSE)

p3 <- ggplot(
  imp,
  aes(x = Importance, y = reorder(Feature, Importance))
) +
  geom_col() +
  labs(title = "Important Network Features",
       x = "Importance", y = "Feature") +
  theme_minimal()

p4 <- ggplot(
  metrics_df,
  aes(x = Metric, y = Value * 100, fill = Metric)
) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(round(Value * 100, 1), "%")),
            vjust = -0.4) +
  coord_cartesian(ylim = c(0, 110)) +
  labs(title = "Model Performance Comparison",
       x = "Performance Metric", y = "Percentage (%)") +
  theme_minimal()

gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2)

ggsave("outputs/traffic_distribution.png", p1,
       width = 7, height = 5)
ggsave("outputs/confusion_matrix.png", p2,
       width = 7, height = 5)
ggsave("outputs/feature_importance.png", p3,
       width = 8, height = 6)
ggsave("outputs/model_performance.png", p4,
       width = 7, height = 5)

cat("\nAll four graphs generated successfully!\n")
cat("Check the Plots pane and outputs folder.\n")
