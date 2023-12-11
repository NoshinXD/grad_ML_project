import torch
import torch.nn as nn
import torch.optim as optim
from mnist_loader import load_data, preprocessing
import numpy as np
from sklearn.metrics import accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from model import NeuralNetwork

import warnings
warnings.filterwarnings("ignore")

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight) # Glorot
#         m.bias.data.fill_(0.01)

def evaluate_model(model, loader, criterion):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    probabilities = []
    actuals = []
    
    model_loss = 0.0
    with torch.no_grad():  # No need to compute gradients
        for features, labels in loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            model_loss += loss.item() 
            
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            # actuals.extend(labels.cpu().numpy())

            probs = torch.softmax(outputs, dim=1)
            probabilities.extend(probs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
     
    return actuals, predictions, probabilities, model_loss

def runNeuralNetwork(learning_rate=0.001, batch_size=64, num_epochs=5, init_type='xavier'):    
    training_data, validation_data, test_data = load_data()
    train_X, train_y = training_data
    validation_X, validation_y = validation_data
    test_X, test_y = test_data

    # preprocessing
    # train_X = preprocessing(train_X)
    # validation_X = preprocessing(validation_X)
    # test_X = preprocessing(test_X)

    # Hyperparameters
    input_size = train_X.shape[1]  
    hidden_size = 128  
    num_classes = len(np.unique(train_y))   

    learning_rate = learning_rate
    batch_size = batch_size   
    num_epochs = num_epochs

    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    validation_X = torch.tensor(validation_X, dtype=torch.float32)
    validation_y = torch.tensor(validation_y, dtype=torch.long)
    validation_dataset = torch.utils.data.TensorDataset(validation_X, validation_y)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size)

    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.long)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = NeuralNetwork(input_size, hidden_size, num_classes)
    model.init_weights(init_type)
    criterion = nn.CrossEntropyLoss() # Initializes loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_actuals = []
    train_predictions = []
    train_probabilities = []

    val_actuals = []
    val_predictions = []
    val_probabilities = []

    train_losses = [] 
    train_accuracies = [] 
    val_losses = [] 
    val_accuracies = []

    train_rocs = []
    val_rocs = []

    # Training Loop
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        # check_train_loss = 0.0
        
        model.train()
        for i, (features, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            # check_train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad() # prevents mixing up gradients between different batches of data
            loss.backward()
            optimizer.step() # updates weights
                
        train_actuals, train_predictions, train_probabilities, train_loss = evaluate_model(model, train_loader, criterion)
        # print(len(train_actuals), len(train_predictions))
        train_acc = accuracy_score(train_actuals, train_predictions)
        train_accuracies.append(train_acc)
        
        train_losses.append(train_loss) 

        actuals_binarized = label_binarize(train_actuals, classes=range(num_classes))
        train_roc_score = roc_auc_score(actuals_binarized, train_probabilities, multi_class='ovr')
        train_rocs.append(train_roc_score)

        val_loss = 0.0
        val_acc = 0.0

        val_actuals, val_predictions, val_probabilities, val_loss = evaluate_model(model, validation_loader, criterion)
        val_acc = accuracy_score(val_actuals, val_predictions)
        val_accuracies.append(val_acc)
        
        val_losses.append(val_loss) 

        actuals_binarized = label_binarize(val_actuals, classes=range(num_classes))
        val_roc_score = roc_auc_score(actuals_binarized, val_probabilities, multi_class='ovr')
        val_rocs.append(val_roc_score)

        # print(check_train_loss)
        print('Epoch [{}/{}],Loss:{:.4f},Validation Loss:{:.4f},Accuracy:{:.2f},Validation Accuracy:{:.2f}'.format( 
            epoch+1, num_epochs, train_loss, val_loss, train_acc ,val_acc))

    print('mean val accuracy', np.mean(val_accuracies))
    print('mean val roc score', np.mean(val_rocs))
    # train_accuracy, train_roc_score = evaluate_model(model, train_loader)
    # validation_accuracy, validation_roc_score = evaluate_model(model, validation_loader)

    test_actuals, test_predictions, test_probabilities, test_model_loss = evaluate_model(model, test_loader, criterion)
    actuals_binarized = label_binarize(test_actuals, classes=range(num_classes))

    test_acc = accuracy_score(test_actuals, test_predictions)
    test_prec = precision_score(test_actuals, test_predictions, average='macro')
    test_rec = recall_score(test_actuals, test_predictions, average='macro')
    test_f1 = f1_score(test_actuals, test_predictions, average='macro')
    
    test_roc_score = roc_auc_score(actuals_binarized, test_probabilities, multi_class='ovr')

    print('test accuracy', test_acc)
    print('test precison', test_prec)
    print('test recall', test_rec)
    print('test f1', test_f1)
    print('test roc score', test_roc_score)

    return train_losses, train_accuracies, val_losses, val_accuracies

learning_rate = 0.001
batch_size = 64    
num_epochs = 5

# Try different number of hidden units, different weight/bias initializations, different learning rates, and
# discuss whether they affect the training/testing performa
train_losses, train_accuracies, val_losses, val_accuracies = runNeuralNetwork(learning_rate, batch_size, num_epochs)
# print(len(train_losses))
# print(len(train_accuracies))
# print(len(val_losses))
# print(len(val_accuracies))