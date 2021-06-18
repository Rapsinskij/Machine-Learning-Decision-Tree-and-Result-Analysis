depth_size <- c(2,4,6,8,10,12,14,16)
depth_training_accuracy <- c(77.3,81.0,83.3, 86.3, 89.0, 91.7, 93.3, 94.1)
depth_validation_accuracy <- c(77.5,80.5, 80.1, 79.2, 76.6, 75.0, 74.3, 74.5)
depth_node_count <- c(3, 15, 38, 77, 130, 178, 208,226)

plot(depth_size, depth_node_count, type = "b", pch = 19, col = "red", 
     main = "Plot of Maximum Tree Depth vs Average Node Count",
     xlab = "Maximum Tree Depth", ylab = "Average Node Count")
plot(depth_size, depth_training_accuracy, type = "b", pch = 19, col = "red", 
     main = "Plot of Maximum Tree Depth vs Accuracies",
     xlab = "Maximum Tree Depth", ylab = "Accuracies (%)", ylim = c(74, 95))
lines(depth_size, depth_validation_accuracy, pch = 18, col = "blue", type = "b")
legend("topleft", legend=c("Training Set Accuracy", "Validation Set Accuracy"),
       col=c("red", "blue"), lty = 1:2)

split <- c(2,4,6,8,10,12,14,16)
split_training_accuracy <- c(85.8, 83.1, 82.1, 81.8, 81.4, 80.2, 79.2, 79.0)
split_validation_accuracy <- c(76.9, 77.5, 78.2, 78.3, 79.2, 78.5, 76.6, 76.9)
split_node_count <- c(89, 43, 29, 24, 19, 15, 11, 10)
plot(split, split_node_count, type = "b", pch = 19, col = "red", 
     main = "Plot of Maximum Tree Depth vs Average Node Count",
     xlab = "Maximum Tree Depth", ylab = "Average Node Count")
plot(depth_size, split_training_accuracy, type = "b", pch = 19, col = "red", 
     main = "Plot of Maximum Tree Depth vs Accuracies",
     xlab = "Maximum Tree Depth", ylab = "Accuracies (%)", ylim = c(75, 87))
lines(depth_size, split_validation_accuracy, pch = 18, col = "blue", type = "b")
legend("topright", legend=c("Training Set Accuracy", "Validation Set Accuracy"),
       col=c("red", "blue"), lty = 1:2)
