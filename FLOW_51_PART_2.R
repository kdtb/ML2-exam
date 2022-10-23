

####### Flow number: 51 | Machine Learning for Business Intelligence 2 (CM) WOAI  ######


###### Exercise 1 ######

rm(list = ls())
gc()
library(keras)

keras_model_sequential()

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(5, 5), activation = "relu",
                input_shape = c(200, 200, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(4, 4)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(5, 5), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 20, activation = "softmax") # in case of multiclass

model

###### Exercise 2 ######


###### Exercise 3 ######

library(keras)
complaints_train <- read.csv(file = "C:/Users/kaspe/OneDrive - Aarhus universitet/Skrivebord/BI/2. semester/ML2/Exam 2022/complaints_train.csv")
complaints_test <- read.csv(file = "C:/Users/kaspe/OneDrive - Aarhus universitet/Skrivebord/BI/2. semester/ML2/Exam 2022/complaints_test.csv")


set.seed(202)
options(scipen = 999)

# Split the dataset

train_proportion <- round(nrow(complaints_train)*0.05)

train <- sample(nrow(complaints_train), train_proportion)
complaints_train <- complaints_train[train, ]


# x
x_train <- c(complaints_train[,2]) # take sequences from df and turn into a list/vector
x_test <- c(complaints_test[,2]) # take sequences from df and turn into a list/vector

# y
y_train <- c(complaints_train$product)
y_train <- as.factor(y_train)
y_train <- as.integer(y_train)-1 # levels must start from 0

y_test <- c(complaints_test$product)
y_test <- as.factor(y_test)
y_test <- as.integer(y_test)-1 # levels must start from 0

# one-hot encode labels
y_train <- to_categorical(y_train, num_classes = 5)
y_test <- to_categorical(y_test, num_classes = 5)



library(stringr)

# Mean 
lengths <- lengths(gregexpr("[A-z]\\W+", x_train)) + 1L

mean <- round(mean(lengths)) # mean = 81


# Extracting words

sub <- word(x_train, start = 1L, end = mean)
library(tidyverse)
sub <- replace_na(sub, replace = "hi")
head(sub)


for (i in 1:length(x_train)) {
  if (sub[i] == "hi") {
    sub[i] = x_train[i]
  }
}

x_train <- sub
head(x_train)


sub <- word(x_test, start = 1L, end = mean)
library(tidyverse)
sub <- replace_na(sub, replace = "hi")
head(sub)


for (i in 1:length(x_test)) {
  if (sub[i] == "hi") {
    sub[i] = x_test[i]
  }
}

x_test <- sub
head(x_test)


# Define number of words to consider

num_words <- 1000  # number of words to consider as vocabulary



# turn sequences into word index

tokenizer <- text_tokenizer(num_words = num_words,
                            filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
                            lower = TRUE,
                            split = " ",
                            char_level = FALSE,
                            oov_token = NULL) %>% fit_text_tokenizer(x_train) # turn sequences into word index


sequences.train <- texts_to_sequences(tokenizer, x_train) # turn index into list of integer indices
sequences.test <- texts_to_sequences(tokenizer, x_test) # turn index into list of integer indices
head(sequences.train)

# List padding (turning the sequences into a tensor)


x_train <- pad_sequences(sequences.train, maxlen = mean) # Turns the lists of integers into a 2D integer tensor of shape (samples, maxlen)
x_test <- pad_sequences(sequences.test, maxlen = mean) # Turns the lists of integers into a 2D integer tensor of shape (samples, maxlen)
head(x_train) # not at all as sparse as in one-hot encoding. We see for the dirst complaint, the first 9 embedding dimensions aren't activated.



###### Exercise 4 ######


get_val_scores <- function(history){
  
  min_loss <-  min(history$metrics$val_loss) 
  min_epoch <- which.min(history$metrics$val_loss)
  min_acc <- history$metrics$val_accuracy[which.min(history$metrics$val_loss)]
  min_precision <- history$metrics$val_precision[which.min(history$metrics$val_loss)]
  min_recall <- history$metrics$val_recall[which.min(history$metrics$val_loss)]
  
  cat('Minimum validation loss:  ')
  cat(min_loss)
  cat('\n')
  
  cat('Loss minimized at epoch:  ')
  cat(min_epoch)
  cat('\n')
  
  cat('Validation accuracy:      ')
  cat(min_acc)
  cat('\n')
  
  cat('Validation precision:      ')
  cat(min_precision)
  cat('\n')
  
  cat('Validation recall:      ')
  cat(min_recall)
  cat('\n')
  
  
  return(list(min_loss = min_loss,
              min_epoch = min_epoch,
              min_acc = min_acc,
              min_precision = min_precision,
              min_recall = min_recall))
  
}


# 1st model

embedding_dim <- 50
embedding_dim2 <- round(num_words ^ (1/4))
embedding_dim3 <- 64
embedding_dim4 <- 100


k_clear_session()
set_random_seed(123)

COV1 <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words, 
                  input_length = mean,
                  output_dim = embedding_dim3) %>% 
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_global_max_pooling_1d() %>% # either layer_flatten or globalmaxpooling so that we turn spatial feature maps into vectors
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 5, activation = "softmax")

COV1 %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = list(metric_precision(), metric_recall(),
                 'accuracy')
)

COV1history <- COV1 %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2,
  verbose = 1,
  callbacks = list(
    callback_reduce_lr_on_plateau(patience = 2, factor = 0.1),
    callback_early_stopping(patience = 5,
                            restore_best_weights = TRUE,
                            min_delta = 0.0001))
)

plot(COV1history) 

get_val_scores(COV1history)




###### Exercise 5 ######



k_clear_session()
set_random_seed(123)

COV1 <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words, 
                  input_length = mean,
                  output_dim = embedding_dim3) %>% 
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_gru(units = embedding_dim) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 5, activation = "softmax")

COV1 %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = list(metric_precision(), metric_recall(),
                 'accuracy')
)

COV1history <- COV1 %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2,
  verbose = 1,
  callbacks = list(
    callback_reduce_lr_on_plateau(patience = 2, factor = 0.1),
    callback_early_stopping(patience = 5,
                            restore_best_weights = TRUE,
                            min_delta = 0.0001))
)

plot(COV1history) 

get_val_scores(COV1history)


###### Exercise 6 ######



library(keras)
complaints_train <- read.csv(file = "C:/Users/XXX/OneDrive - Aarhus universitet/Skrivebord/BI/2. semester/ML2/Exam 2022/complaints_train.csv")
complaints_test <- read.csv(file = "C:/Users/XXX/OneDrive - Aarhus universitet/Skrivebord/BI/2. semester/ML2/Exam 2022/complaints_test.csv")


set.seed(202)
options(scipen = 999)

# Split the dataset

train_proportion <- round(nrow(complaints_train)*0.05)

train <- sample(nrow(complaints_train), train_proportion)
complaints_train <- complaints_train[train, ]


# x
x_train <- c(complaints_train[,2]) # take sequences from df and turn into a list/vector
x_test <- c(complaints_test[,2]) # take sequences from df and turn into a list/vector

# y
y_train <- c(complaints_train$product)
y_train <- as.factor(y_train)
y_train <- as.integer(y_train)-1 # levels must start from 0

y_test <- c(complaints_test$product)
y_test <- as.factor(y_test)
y_test <- as.integer(y_test)-1 # levels must start from 0

# one-hot encode labels
y_train <- to_categorical(y_train, num_classes = 5)
y_test <- to_categorical(y_test, num_classes = 5)



library(stringr)

# Mean 
lengths <- lengths(gregexpr("[A-z]\\W+", x_train)) + 1L

mean <- round(mean(lengths)) # mean = 81


# Extracting words

sub <- word(x_train, start = 1L, end = mean)
library(tidyverse)
sub <- replace_na(sub, replace = "hi")
head(sub)


for (i in 1:length(x_train)) {
  if (sub[i] == "hi") {
    sub[i] = x_train[i]
  }
}

x_train <- sub
head(x_train)


sub <- word(x_test, start = 1L, end = mean)
library(tidyverse)
sub <- replace_na(sub, replace = "hi")
head(sub)


for (i in 1:length(x_test)) {
  if (sub[i] == "hi") {
    sub[i] = x_test[i]
  }
}

x_test <- sub
head(x_test)



tokenizer <- text_tokenizer(num_words = num_words,
                            filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
                            lower = TRUE,
                            split = " ",
                            char_level = FALSE,
                            oov_token = NULL) %>% fit_text_tokenizer(x_train) # turn sequences into a word index

# what are the most common words?


tokenizer$word_index %>%
  head()
# most common word is credit


# Not useful here, but it gives word index: 
word_index <- tokenizer$word_index # word index
cat("Found", length(word_index), "unique tokens.\n") # how many unique tokens we have
# 43811 unique tokens!



one_hot_results_train <- texts_to_matrix(tokenizer, x_train, mode = "binary") # This one hot encodes the data without need of embedding and padding
one_hot_results_test <- texts_to_matrix(tokenizer, x_test, mode = "binary") # This one hot encodes the data without need of embedding and padding

# ^^ each column represents one token/word (one of the 1000 most popular words), and 1 row corresponds to 1 tweet. So if 2 of the most common words are used in a tweet, then 2 cells has a "1" value
dim(one_hot_results_train)


model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(num_words)) %>% # we don't even need to set input shape argument
  layer_dense(units = 5, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = list(metric_precision(), metric_recall(),
                 'accuracy')
)

history <- model %>% fit(
  one_hot_results_train, y_train,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(
    callback_reduce_lr_on_plateau(patience = 2, factor = 0.1),
    callback_early_stopping(patience = 5,
                            restore_best_weights = TRUE,
                            min_delta = 0.0001)
  )
)

# Plot
plot(history)
# Validation measures
get_val_scores(history)


###### Exercise 7 ######

# COV1
results1 <- COV1 %>% evaluate(x_test, y_test)

# One-hot model (contender)
results2 <- model %>% evaluate(one_hot_results_test, y_test)




