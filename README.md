# EXP-02 : Developing a Neural Network Classification Model
### NAME : HARI PRIYA M
### REG NO : 212224240047

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="700" height="450" alt="ChatGPT Image Jul 29, 2026, 03_54_05 PM" src="https://github.com/user-attachments/assets/088af943-fd87-416c-bfab-74791452a493" />

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

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)



    def forward(self, x):
      x = torch.relu(self.fc1(x))
      x = torch.relu(self.fc2(x))
      x = self.fc3(x)
      return x

        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()

  for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()

      outputs = model(X_batch)

      loss = criterion(outputs, y_batch)

      loss.backward()

      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information
<img width="800" height="391" alt="image" src="https://github.com/user-attachments/assets/c1c16f79-3486-4045-bb18-4bb5761891b1" />

### OUTPUT

## Confusion Matrix & Classification Report
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/69ca9872-381b-4e8c-82a5-792155c33b4d" />
<br><br>
<img width="600" height="370" alt="image" src="https://github.com/user-attachments/assets/4fb68715-1dea-49b9-991f-28f6dc0a6b18" />



### New Sample Data Prediction
<img width="417" height="108" alt="image" src="https://github.com/user-attachments/assets/4b7133fb-eb42-46a7-abe6-a5ad7547e440" />

## RESULT
Thus, a neural network classification model was successfully developed, trained, and tested using PyTorch. The model was able to classify customers into the appropriate segments based on the given input features.
