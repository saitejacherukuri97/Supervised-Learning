# Supervised-Learning
This project is part of CSE 574 - Introduction to Machine Learning

**Project Overview:**

The goal of the project is to perform classification using machine learning. In the 
first part of the project, I have performed Data Pre-processing. Then, I trained the 
model using gradient descent for logistic regression and calculated the accuracy by 
tested the model on the testing set. Later I implemented Neural networks using 
different regularization methods and then calculated accuracy for each of them.

**Dataset:**

To implement machine learning models with Pima Indian Diabetes dataset with 768 
samples, I have split the data samples as Training, Validation and Testing, each 
constituting of 60%, 20% and 20% respectively of the overall data. Training dataset 
has 460 samples, Validation dataset has 154 samples and Testing dataset has 154 
samples.

**Python Editor:**

I have used Jupiter Notebook on Google Collab for implementation and shared.

**Data processing:**

1. Extracted feature values from the data
2. Constructed Correlation matrix
3. Performed Data Normalization and Data Partitioning

**Model Building:**

**Part - 1: Implementing Logistic Regression**

First, I have applied Transpose for Training, Validation and Testing matrices so we 
can perform multiplications.
Then I have initialized the weights as numpy array having the dimension as one and 
filled with 0.01 and bias is initially assigned to zero.
Then taking the pre-processed dataset, I have defined the sigmoid activation function 
and gradient descent. Logistic regression is mainly used for binary classification, 
hence the outputs are always between 0 and 1. The role of sigmoid function is to 
convert linear model to predicted output. 
Here we take a linear model (z) of form Z = W*X + B , where 
W is the weights matrix and B is the bias

![image](https://user-images.githubusercontent.com/42407754/147020496-8ccf4874-e53c-43e7-885c-b01adcc62d49.png)

Later, I have used the Forward-Backward propagation to find gradient descent and cost function and then I declared Logistic regression function using training data, learning rate and Epochs (number of iterations):

**Visualization for Validation dataset**

I have plotted two graphs as below :
1. Cost vs Number of Iterations
2. Accuracy vs Number of Iterations
Accuracy achieved = **77.27 %**

![image](https://user-images.githubusercontent.com/42407754/147020697-ad569d5f-6933-4144-81b5-cdf22389d504.png)

**Visualization for Testing dataset**

I have plotted two graphs as below :
1. Cost vs Number of Iterations
2. Accuracy vs Number of Iterations
Accuracy achieved = **80.51 %**

![image](https://user-images.githubusercontent.com/42407754/147020751-bf327838-792c-48ff-bc2b-e8a0d1de049e.png)

![image](https://user-images.githubusercontent.com/42407754/147020786-b29e6c9f-c111-422e-a9d4-b22b619defe2.png)

**Part - 2: Implementing Neural Networks**

• First we load the libraries Sequential and Dense from Keras.
• Now, we start building the artificial neural network which is a sequential model of 3 
layers.
1. The first layer will have 12 neurons and uses the ReLu activation function.
2. The Second layer will have 15 neurons and used ReLu activation function.
3. Last layer which is the Output layer has only 1 neuron which uses Sigmoid 
function.
• Next, I have used a Regularization technique in the model to reduce model 
overfitting.
  o I have used L1 Regularization for first layer and L2 Regularization for second 
  layer.

**Part – 3: Implement different regularization methods for the Neural Networks
Regularization method : Dropout**

• Dropout is one of the regularization techniques, which is the most used one.
• In this method, it randomly selects some nodes and removes them, so each iteration 
has different set of nodes and hence this results in a different set of outputs.
• Dropout method produces best results when applies on larger datasets.

**Regularization method : L2**

Similar to above model explained in part - 2, I have implemented the model with L2 
regularization.
• Batch size = 1000 and Epochs = 32
• Regularization parameter = 0.01 and Learning rate for SGD Optimizer = 0.01

**Conclusion:**

• We have implemented a logistic regression model developed with gradient descent 
yielded an accuracy of 80.51%.
• Splitting dataset into 60% training and 20% Validation dataset, helped in tuning the 
hyper parameters which improved the accuracy of the model.
• Implemented Neural networks with L1, L2 and Dropout regularization methods
• Neural network with L2 regularization method yielded better results than the Dropout
as we have a smaller network.
