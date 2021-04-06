library(tensorflow)
library(keras)

tf_config()

# Set up your session

# Add your constants
female <- tf$constant(150, name = "FemaleEmployees")
male <- tf$constant(135, name = "MaleEmployees")
total <- tf$add(female, male)
EmployeeSession <- tf$comp
print(session$run(total))

# Write to file
towrite <- tf$summary$FileWriter('./graphs', EmployeeSession$graph)



mnist <- dataset_mnist()
mnist$train$x <- mnist$train$x/255
mnist$test$x <- mnist$test$x/255

model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(28, 28)) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(0.2) %>% 
  layer_dense(10, activation = "softmax")

summary(model)


model %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

model %>% 
  fit(
    x = mnist$train$x, y = mnist$train$y,
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

predictions <- predict(model, mnist$test$x)

model %>% evaluate(mnist$test$x, mnist$test$y, verbose = 0)
