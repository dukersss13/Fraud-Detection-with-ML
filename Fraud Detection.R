setwd("C:/Users/Duker/Desktop/Fall 2020/CS 614/Final Project/Fraud-Detection-with-ML")
#setwd("~/Desktop/614")
require(ggplot2)
require(RColorBrewer)
require(plyr)
require(dplyr)
require(varhandle)
require(DMwR)
require(cvms)
require(randomForest)
require(outliers)
require(broom)
require(caret)
require(rsample)
require(Metrics)
require(pROC)
require(keras)
require(tensorflow)
#install_keras()
#install_tensorflow()

rm(list=ls())
data = read.csv("data/fraud.csv")

# Data Pre-Processing ####
attach(data)
str(data)

# Renaming destination names. Merchant (M) = 1, Customer (C) = 0.
df = data %>% select(-c(step, nameOrig, oldbalanceDest,
                        newbalanceDest, isFlaggedFraud))

df$nameDest = factor(ifelse(startsWith(unfactor(nameDest), "M"), 1, 0))
df$isFraud = factor(df$isFraud)
df$type = factor(df$type)

# EDA + Plots ####
summary(df)
str(df)

# Distribution Plot of Transaction $
remove.outliers = function(z){
  z.score = scale(z)
  w = which(abs(z.score) <=3)
  return(list(index = w, data = z[w]))
}
d = remove.outliers(df$amount)
amount2 = d$data
df2 = df[d$index,]

ggplot(df2, aes(x = amount2)) +
  geom_histogram(bins = 30, color = "black", fill = "cornflowerblue") +
  labs(x = "Transaction Amount ($)", y = "Count", title = "Distribution of Transaction Amount") +
  theme(plot.title = element_text(size = 13, hjust = 0.5))

# Distribution of Transaction Types
types.count = df %>%
  group_by(type) %>% summarise(counts = n())

type.labels = c("Cash-In", "Cash-Out", "Debit", 
                "Payment", "Transfer")

ggplot(types.count, aes(x = type, y = sort(counts))) +
  geom_bar(stat = "identity", aes(fill = factor(type))) +
  geom_text(aes(label = counts), vjust = -0.3) +
  scale_x_discrete(breaks = c(1:5), 
                   labels = type.labels) +
  scale_fill_brewer(palette = "Set1",
                    labels = type.labels, "Type") +
  labs(x = "Transaction Types", y = "Count", 
       title = "Distribution of Transaction Types") +
  theme(plot.title = element_text(size = 13, hjust = 0.6))

# Train-Test Split ####
set.seed(123)
split = initial_split(df, prop = 0.7)
train = training(split)
test = testing(split)

xtrain = train[,-ncol(train)]
ytrain = train[,ncol(train)]
xtest = test[,-ncol(test)]
ytest = test[,ncol(test)]

# Non-SMOTE Data ####
resample = function(data, n){
  set.seed(42)
  label.0 = data %>% filter(isFraud == 0)
  label.1 = data %>% filter(isFraud == 1)
  subset.label.0 = label.0 %>% sample_n(size = nrow(label.0)*n)
  df.0 = rbind(subset.label.0, label.1)
  order = sample(nrow(df.0))
  df.0 = df.0[order,]
  return(df.0)
}

non.smote.train = resample(train, 0.1)
dim(non.smote.train)
nrow(non.smote.train)
table(non.smote.train$isFraud)
prop.table(table(non.smote.train$isFraud))

# Pre-SMOTE Dist Plot #####
fraudCount = non.smote.train %>%
  group_by(isFraud) %>% summarise(counts = n())

ggplot(fraudCount, aes(x = isFraud, y = counts)) +
  geom_bar(stat = "identity", aes(fill = factor(isFraud))) +
  geom_text(aes(label = counts), vjust = -0.3) +
  scale_x_discrete(breaks = c(0,1), labels = c("No", "Yes")) +
  scale_fill_manual(values = c("brown1", "#56B4E9"),
                    labels = c("No", "Yes"), "Fraud") +
  labs(x = "Fraud (Yes/No)", y = "Count", 
       title = "Pre-SMOTE Fraud vs Non-Fraud Transactions") +
  theme(plot.title = element_text(size = 13, hjust = 0.6))

# SMOTE ####
smote.train = SMOTE(isFraud~., data = non.smote.train, k = 5,
                    perc.over = 3500, perc.under = 120)
prop.table(table(smote.train$isFraud))
table(smote.train$isFraud)
nrow(smote.train)
nrow(non.smote.train)

smote.fraudCount = smote.train %>% 
  group_by(isFraud) %>% summarise(counts = n())

ggplot(smote.fraudCount, aes(x = isFraud, y = counts)) + 
  geom_bar(stat = "identity", aes(fill = factor(isFraud))) + 
  geom_text(aes(label = counts), vjust = -0.3) + 
  scale_x_discrete(breaks = c(0,1), labels = c("No", "Yes")) +
  scale_fill_manual(values = c("brown1", "#56B4E9"), 
                    labels = c("No", "Yes"), "Fraud") +
  labs(x = "Fraud (Yes/No)", y = "Count", 
       title = "Post-SMOTE Fraud vs Non-Fraud Transactions") +
  theme(plot.title = element_text(size = 13, hjust = 0.6))

# Random Forest ####
random.forest = function(train.data){
  rf = randomForest(isFraud~., data = train.data)
  return(rf)
}

validate = function(model, xtest, ytest){
  pred = predict(model, xtest)
  return(list(confusionMatrix = confusionMatrix(pred, ytest),
              ypred = pred))
}

ptm = proc.time()
non.smote.rf = random.forest(non.smote.train)
non.smote.rf.time = (proc.time() - ptm)[3]
non.smote.rf.time 

non.smote.rf
non.smote.rf.results = validate(non.smote.rf, xtest, ytest)
non.smote.rf.cm = non.smote.rf.results$confusionMatrix
non.smote.rf.cm
non.smote.rf.pred = non.smote.rf.results$ypred
non.smote.rf.cMatrix = non.smote.rf.cm$table
non.smote.rf.auc = round(Metrics::auc(ytest, non.smote.rf.pred),3)

plot_confusion_matrix(tidy(non.smote.rf.cMatrix),
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n", 
                      place_x_axis_above = F,
                      add_row_percentages = F,
                      palette = "Purples",
                      tile_border_color = "black", 
                      tile_border_size = 0.2) +
  labs(title = "Pre-SMOTE RF Confusion Matrix",
       subtitle = paste("AUC:", non.smote.rf.auc)) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ptm = proc.time()
smote.rf = random.forest(smote.train)
smote.rf.time = (proc.time() - ptm)[3]
smote.rf.time

smote.rf
smote.rf.results = validate(smote.rf, xtest, ytest)
smote.rf.cm = smote.rf.results$confusionMatrix
smote.rf.pred = non.smote.rf.results$ypred
smote.rf.cMatrix = smote.rf.cm$table
smote.rf.auc = round(Metrics::auc(ytest, smote.rf.pred),3)

plot_confusion_matrix(tidy(smote.rf.cMatrix),
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "n", 
                      place_x_axis_above = F,
                      add_row_percentages = F,
                      tile_border_color = "black", 
                      tile_border_size = 0.2) +
  labs(title = "Post-SMOTE RF Confusion Matrix",
       subtitle = paste("AUC:", smote.rf.auc)) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

smote.rf.labels = c("Type", "Amount", "Old Balance", "New Balance",
                    "Destination/n(Customer/Merchant)")

caret::varImp(smote.rf)
smote.rf.Imp = caret::varImp(smote.rf)

ggplot(data = smote.rf.Imp, aes(x = Overall)) +
  geom_point(aes(x = sort(Overall), y = seq(1,5))) +
  scale_y_continuous(breaks = c(1:5), labels = rev(smote.rf.labels))

# Keras Neural Network ####

# Prepping the data for the Keras NN
NN.data.processing = function(data){
    xtrain = scale(as.matrix(sapply(data[,-ncol(data)], as.numeric)))
    ytrain = to_categorical(data[,ncol(data)],2)
    return(list(xtrain = xtrain, ytrain = ytrain))
}

non.smote.train.nn = NN.data.processing(non.smote.train)
xtrain.non.smote = non.smote.train.nn$xtrain
ytrain.non.smote = non.smote.train.nn$ytrain

smote.train.nn = NN.data.processing(smote.train)
xtrain.smote = smote.train.nn$xtrain
ytrain.smote = smote.train.nn$ytrain

# Setting up the NN
keras.NN = function(xtrain, ytrain, neurons, 
                    epoch, batch_size, validation_split,
                    activation2, activation3, activation4, activation5,
                    loss, optimizer, metrics){
  model = keras_model_sequential()
  model %>%
    layer_dense(units = neurons, input_shape = c(ncol(xtrain))) %>%
    layer_activation_leaky_relu() %>%
    layer_dense(units = neurons*2, activation = "tanh") %>%
    layer_dense(units = neurons*4, activation = activation2) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = neurons, activation = activation3) %>%
    layer_dense(units = neurons*6) %>%
    layer_activation_leaky_relu() %>%
    layer_dense(units = neurons*5, activation = activation4) %>%
    layer_dense(units = neurons*3, activation = activation5)%>%
    layer_dense(units = 2, activation = "softmax")
  
  model %>% compile(loss = loss, optimizer = optimizer,
                       metrics = metrics)
  
  history = model %>% fit(xtrain, ytrain, 
                           epoch = epoch, batch_size = batch_size, 
                          validation_split = validation_split)
  return(model)
}

ptm = proc.time()
non.smote.NN = keras.NN(xtrain.non.smote, ytrain.non.smote, neurons = 30,
                        epoch = 50, batch_size = 32, validation_split = 0.3, 
                        activation2 = "tanh", activation3 = "tanh",
                        activation4 = "tanh", activation5 = "tanh",
                        "binary_crossentropy", "SGD", "accuracy")
non.smote.NN.time = proc.time() - ptm
non.smote.NN.time[3]

ptm = proc.time()
smote.NN = keras.NN(xtrain.smote, ytrain.smote, neurons = 30,
                    epoch = 50, batch_size = 32, validation_split = 0.3, 
                    activation2 = "tanh", activation3 = "tanh", 
                    activation4 = "tanh", activation5 = "tanh",
                    "binary_crossentropy", "SGD", "accuracy")
smote.NN.time = proc.time() - ptm
smote.NN.time[3]

test.nn = NN.data.processing(test)
xtest.nn = test.nn$xtrain

non.smote.NN.pred = non.smote.NN %>% predict_classes(xtest.nn)
non.smote.NN.cMatrix = table(Predicted = non.smote.NN.pred, Actuals = ytest)
non.smote.NN.auc = round(Metrics::auc(ytest, non.smote.NN.pred),3)

plot_confusion_matrix(tidy(non.smote.NN.cMatrix),
                      target_col = "Actuals", 
                      prediction_col = "Predicted",
                      counts_col = "n", 
                      place_x_axis_above = F,
                      add_row_percentages = F,
                      add_counts = F,
                      palette = "Purples",
                      tile_border_color = "black", 
                      tile_border_size = 0.2) +
  labs(title = "Pre-SMOTE NN Confusion Matrix",
       subtitle = paste("AUC:", non.smote.NN.auc)) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

smote.NN.pred = smote.NN %>% predict_classes(xtest.nn)
smote.NN.cMatrix = table(Predicted = smote.NN.pred, Actuals = ytest)
smote.NN.auc = round(Metrics::auc(ytest, smote.NN.pred),3)

plot_confusion_matrix(tidy(smote.NN.cMatrix),
                      target_col = "Actuals", 
                      prediction_col = "Predicted",
                      counts_col = "n", 
                      place_x_axis_above = F,
                      add_row_percentages = F,
                      tile_border_color = "black", 
                      tile_border_size = 0.2) +
  labs(title = "Post-SMOTE NN Confusion Matrix",
       subtitle = paste("AUC:", smote.NN.auc)) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))




