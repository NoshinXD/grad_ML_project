In this project, we implement six models, Logistic Regression, K Nearest Neighbor (KNN), Decision Tree, (Support Vector Machine) SVM, Random Forest, and Adaboost. To train our model, we use 2 different dataset. The dataset can be found inside the "dataset" folder.

We also implement a Neural Network model using PyTorch which can be found inside "notebooks/NeuralNetwork/NeuralNetwork.py". The analysis is done in "notebooks/NeuralNetwork/NN_analysis.ipynb" We use popular MNIST dataset to train and test our model.

As the bonus task, we implemented an AutoEncoder to detech Fraudulent Handwritten Signatures.

To run the models do the following steps:

1. To run all of the ML Classifiers except Neural Network, run notebooks/{classifierName}.ipynb
i.e., notebooks/LogisticRegression.ipynb, notebooks/KNearestNeighbour.ipynb, notebooks/Decision_tree.ipynb, notebooks/SVM.ipynb, notebooks/RandomForest.ipynb, notebooks/AdaBoost.ipynb
2. To run the neural Neural Network model, run notebooks/NeuralNetwork/NN_analysis.ipynb
3. To run the bonus task, run bonus/notebooks/ae_forged_sign_detection.ipynb
