#Ahnaf Ahmad
#1001835014

import pandas as pd
import numpy as np
import sys
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def euc_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            prediction = self._predict(x)
            predictions.append(prediction)
        return predictions


    def _predict(self, x):
        distances = []
        for x_train in self.X_train:
            distance = euc_distance(x, x_train)
            distances.append(distance)
        indices = np.arange(len(distances))
        indices = sorted(indices, key=lambda i: distances[i])[:self.k]


        nearest_labels = []
        for i in indices:
            label = self.y_train[i]
            nearest_labels.append(label)
        return Counter(nearest_labels).most_common(1)[0][0]

    def predict2(self, X):
        predictions = []
        for x in X:
            prediction = self._predict2(x)
            predictions.append(prediction)
        return predictions
    
    def _predict2(self, x):
        distances = []
        for x_train in self.X_train:
            distance = euc_distance(x, x_train)
            distances.append(distance)
        indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        
        nearest_labels = []
        for i in indices:
            label = self.y_train[i].tolist()
            nearest_labels.append(label)

        return Counter(tuple(x) for x in nearest_labels).most_common(1)[0][0]

class SVM:

    def __init__(self, lr, itterations, lambda_param):
        self.lr = lr
        self.itterations= itterations
        self.lambda_param = lambda_param
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.z = 0
    

        i = 0
        while i < self.itterations:
            self.update()
            i = i + 1

    def update(self):
        y_label = np.ones(self.y.shape)
        y_label[self.y <= 0] = -1

        for i, x_i in enumerate(self.X):
           
            if (y_label[i] * (np.dot(x_i, self.w) - self.z)) >= 1:
                dw = 2 * self.lambda_param * self.w
                db = 0
            else:
                dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_label[i])
                db = y_label[i]

            self.w = self.w - self.lr * dw
            self.z = self.z - self.lr * db
    
    def predict(self, X):
        output = np.dot(X, self.w) - self.z
        predicted_labels = np.sign(output)
        return np.where(predicted_labels <= -1,0,1)


class MLChoice:
    
    def run(self, ml, dataset):
        if ml == 'knn' and dataset== 'banknote':
            data = pd.read_csv('data_banknote_authentication.txt', header=None)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            knn = KNN(5)
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            knn_sklearn = KNeighborsClassifier(n_neighbors=5)
            knn_sklearn.fit(X_train, y_train)
            predictions_sklearn = knn_sklearn.predict(X_test)
            accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)
            
            accuracy = accuracy * 100
            accuracy_sklearn = accuracy_sklearn * 100
            print(f'Accuracy of training (scratch): {accuracy:.2f}%')
            print(f'Accuracy of Scikitlearn: {accuracy_sklearn:.2f}%')

            f = open("output.txt","w+")
            f.write(f'Dataset: Bank Note\n')
            f.write(f'Machine Learning Algorithm chose: KNN\n')
            f.write(f'Accuracy of training (scratch): {accuracy:.2f}%\n')
            f.write(f'Accuracy of Scikitlearn: {accuracy_sklearn:.2f}%')
            f.close()

            return 
        elif ml == 'knn' and dataset== 'sonar':
            data = pd.read_csv('sonar.txt', header=None)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            onehot_encoder = OneHotEncoder(sparse_output=False)
            y = y.reshape(len(y), 1)
            y = onehot_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            sk_knn = KNeighborsClassifier(n_neighbors=3)
            sk_knn.fit(X_train, y_train)
            y_pred = sk_knn.predict(X_test)
            sk_accuracy = accuracy_score(y_test, y_pred)

            knn = KNN(5)
            knn.fit(X_train, y_train)
            predictions = knn.predict2(X_test)
            predictions = np.argmax(predictions, axis=1)
            y_test = np.argmax(y_test, axis=1)
            accuracy = accuracy_score(y_test, predictions)
            
            accuracy = accuracy * 100
            sk_accuracy = sk_accuracy * 100


            print(f'Accuracy of training (scratch): {accuracy:.2f}%')
            print(f'Accuracy of Scikitlearn: {sk_accuracy:.2f}%')
            
            f = open("output.txt","w+")
            f.write(f'Dataset: Sonar\n')
            f.write(f'Machine Learning Algorithm chose: KNN\n')
            f.write(f'Accuracy of training (scratch): {accuracy:.2f}%\n')
            f.write(f'Accuracy of Scikitlearn: {sk_accuracy:.2f}%')
            f.close()

            return
        
        elif ml == 'svm' and dataset== 'banknote':
            df = pd.read_csv('data_banknote_authentication.txt', header=None)

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            svm = SVM(0.001, 1000,0.01)
            svm.fit(X_train, y_train)
            svm_sk = SVC()
            svm_sk.fit(X_train, y_train)

            predictions = svm.predict(X_test)
            predictions_sk = svm_sk.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            accuracy_sk = accuracy_score(y_test, predictions_sk)

            accuracy = accuracy * 100
            accuracy_sk = accuracy_sk * 100

            print(f'Accuracy of training (scratch): {accuracy:.2f}%')
            print(f'Accuracy of Scikitlearn: {accuracy_sk:.2f}%')

            f = open("output.txt","w+")
            f.write(f'Dataset: Bank Note\n')
            f.write(f'Machine Learning Algorithm chose: SVM\n')
            f.write(f'Accuracy of training (scratch): {accuracy:.2f}%\n')
            f.write(f'Accuracy of Scikitlearn: {accuracy_sk:.2f}%')
            f.close()

        elif ml == 'svm' and dataset== 'sonar':
            data = pd.read_csv('sonar.txt', header=None)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values

            le = LabelEncoder()
            y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            svm = SVM(0.001, 1000, 0.01)
            svm.fit(X_train, y_train)
            predictions = svm.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracy = accuracy * 100
            print("Accuracy of training (scratch): " + str(round(accuracy,2)) + "%")

            svm_sk = SVC()
            svm_sk.fit(X_train, y_train)
            predictions_sk = svm_sk.predict(X_test)
            accuracy_sk = accuracy_score(y_test, predictions_sk)
            accuracy_sk = accuracy_sk * 100
            print(f'Accuracy of Scikitlearn: {accuracy_sk:.2f}%')

            f = open("output.txt","w+")
            f.write(f'Dataset: Sonar\n')
            f.write(f'Machine Learning Algorithm chose: SVM\n')
            f.write(f'Accuracy of training (scratch): {accuracy:.2f}%\n')
            f.write(f'Accuracy of Scikitlearn: {accuracy_sk:.2f}%')
            f.close()
        
mlc = MLChoice()

mlc.run(sys.argv[1], sys.argv[2])