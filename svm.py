# Importujemy potrzebne biblioteki
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from prettytable import PrettyTable
import seaborn as sns

# Ładujemy zbiór danych - breast cancer dataset from sklearn
cancer = datasets.load_breast_cancer()

# Wybieramy drugi zbiór danych - wine dataset from sklearn
wine = datasets.load_wine()

# Funkcja do opisu zbioru danych
def describe_data(dataset, name):
    print(f"\nOpis zbioru danych {name}:")
    print("Nazwy cech:", dataset.feature_names)
    print("Nazwy klas:", dataset.target_names)
    print("Równowaga klas:", np.bincount(dataset.target))

# Funkcja do podziału i skalowania danych
def split_and_scale_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Funkcja do trenowania modelu SVM
def train_svm(X_train, y_train, kernel='linear', C=1.0):
    svm_model = svm.SVC(kernel=kernel, C=C)
    svm_model.fit(X_train, y_train)
    return svm_model

# Funkcja do oceny modelu
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Confusion matrix:\n", confusion_mat)
    

def conclude_evaluation(y_test, y_pred, class_names):
    print("Classification Report:")
    print_classification_report(y_test, y_pred, class_names)
    plot_confusion_matrix(y_test, y_pred, class_names)

def plot_class_balance(dataset, name):
    unique, counts = np.unique(dataset.target, return_counts=True)
    plt.bar(dataset.target_names[unique], counts)
    plt.title(f'Równowaga klas w zbiorze {name}')
    plt.xlabel('Klasy')
    plt.ylabel('Liczba próbek')
    plt.show()

def plot_confusion_matrix(y_test, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def print_classification_report(y_test, y_pred, class_names):
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    table = PrettyTable()
    table.field_names = ["", "precision", "recall", "f1-score", "support"]
    for key, values in report.items():
        if key == 'accuracy':
            continue
        row = [key]
        row.extend([round(v, 2) if isinstance(v, float) else v for v in values.values()])
        table.add_row(row)
    print(table)

# Opisujemy zbiory danych
describe_data(cancer, "Breast Cancer")
describe_data(wine, "Wine")

# Podział i skalowanie danych
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = split_and_scale_data(cancer.data, cancer.target)
X_train_wine, X_test_wine, y_train_wine, y_test_wine = split_and_scale_data(wine.data, wine.target)

# Trenowanie i ocena modeli SVM dla różnych wartości parametru C
print("\nTrenowanie i ocena modeli SVM dla różnych wartości parametru C. Kernel: linear")
print("\nBreast Cancer dataset:")
for C in [0.1, 1, 5]:
    print(f"\nKernel: linear")
    print(f"\nParametr C: {C}")
    svm_model_cancer = train_svm(X_train_cancer, y_train_cancer, 'linear', C)
    evaluate_model(svm_model_cancer, X_test_cancer, y_test_cancer)
    conclude_evaluation(y_test_cancer, svm_model_cancer.predict(X_test_cancer), cancer.target_names)

print("\nWine dataset:")
for C in [0.1, 1, 5]:
    print(f"\nKernel: linear")
    print(f"\nParametr C: {C}")
    svm_model_wine = train_svm(X_train_wine, y_train_wine, 'linear', C)
    evaluate_model(svm_model_wine, X_test_wine, y_test_wine)
    conclude_evaluation(y_test_wine, svm_model_wine.predict(X_test_wine), wine.target_names)

# Trenowanie i ocena modeli SVM dla różnych kerneli
print("\nBreast Cancer dataset:")
for kernel in ['linear', 'rbf']:
    C = 0.1
    print(f"\nKernel: {kernel}")
    print(f"\nParametr C: {C}")
    svm_model_cancer = train_svm(X_train_cancer, y_train_cancer, kernel, 0.1)
    evaluate_model(svm_model_cancer, X_test_cancer, y_test_cancer)
    conclude_evaluation(y_test_cancer, svm_model_cancer.predict(X_test_cancer), cancer.target_names)

print("\nWine dataset:")
for kernel in ['linear', 'rbf']:
    C = 0.1
    print(f"\nKernel: {kernel}")
    print(f"\nParametr C: {C}")
    svm_model_wine = train_svm(X_train_wine, y_train_wine, kernel, 0.1)
    evaluate_model(svm_model_wine, X_test_wine, y_test_wine)
    conclude_evaluation(y_test_wine, svm_model_wine.predict(X_test_wine), wine.target_names)

plot_class_balance(cancer, "Breast Cancer")
plot_class_balance(wine, "Wine")
