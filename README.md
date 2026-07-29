# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

STEP 1 : Load the customer segmentation dataset and preprocess the data by removing unnecessary columns and handling missing values.

STEP 2 : Encode categorical features and convert the target classes into numerical labels.

STEP 3 : Split the dataset into training and testing sets and normalize the feature values.

STEP 4 : Build a neural network classification model using fully connected layers and ReLU activation functions.

STEP 5 : Train the neural network using the Cross Entropy Loss function and Adam optimizer.

STEP 6 : Evaluate the trained model using accuracy, confusion matrix, classification report, and predict the class of a new sample.



## PROGRAM

### Name: Hari Priya M

### Register Number: 212224240047

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here



    def forward(self, x):
        #Include your code here
        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    #Include your code here

```

### Dataset Information
Include screenshot of the dataset.

### OUTPUT

## Confusion Matrix

Include confusion matrix here

## Classification Report
Include classification report here

### New Sample Data Prediction
Include your sample input and output here

## RESULT
Include your result here
