Common activation functions include:

- **ReLU**: Turns negative inputs to 0 and keeps positive inputs as they are.

- **Sigmoid**: Squashes inputs to a range between 0 and 1, like a smooth on/off switch.

- **Tanh**: Squashes inputs to a range between -1 and 1, giving a bit more flexibility.

----------------------------------------------------------------------------------------------------------------------------

Common loss functions include:

**Mean Squared Error (MSE)**: Measures the average squared difference between predictions and actual values. Often used in regression tasks (e.g., predicting house prices).

**Cross-Entropy Loss**: Measures how well the predicted probabilities match the actual labels. Often used in classification tasks (e.g., identifying cats vs. dogs).

**Hinge Loss**: Used in tasks like support vector machines (SVMs) for classification.

----------------------------------------------------------------------------------------------------------------------------

Here are some guidelines for selecting an activation function for the output layer based on the type of prediction problem:

**Regression**: Use Linear Activation Function

**Binary Classification**: Use Sigmoid/Logistic Activation Function

**Multi-class Classification**: Use Softmax

**Multi-label Classification**: Use Sigmoid
