import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

simplefilter(action='ignore', category=FutureWarning)


url ='diabetes.csv'
data = pd.read_csv(url)

data.drop(['Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'],axis=1, inplace=True)
data.Age.replace(np.nan,34, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60,100]
nombres = ['1','2','3','4','5','6','7']
data.Age=pd.cut(data.Age, rangos, labels=nombres)
data.dropna(axis=0,how='any',inplace=True)

#partir la tabla en dos
data_train = data[:391]
data_test = data[391:]

x=np.array(data_train.drop(['Outcome'], 1))
y=np.array(data_train.Outcome)# 0 sale 1 no sale

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)# 0 sale 1 no sale

# seleccionar modelo regresion logistica
logreg = LogisticRegression(solver='lbfgs', max_iter=7600)

#entreno el modelo
logreg.fit(x_train, y_train)

#metricas

print('*'*50)
print('Regresion Logistica')

#accuracy de entrenamiento train
print(f'accuracy de train de entrenamiento:{logreg.score(x_train, y_train)}')
#accuracy de entrenamiento test
print(f'accuracy de test de entrenamiento:{logreg.score(x_test,y_test)}')
#accuracy de validacion
print(f'accuracy de validacion:{logreg.score(x_test_out,y_test_out)}')


# seleccionar maquina soporte vectorial
svc = SVC(gamma='auto')

#entreno el modelo
svc.fit(x_train, y_train)

#metricas

print('*'*50)
print('Maquina de soporte vectorial')

#accuracy de entrenamiento train
print(f'accuracy de train de entrenamiento:{svc.score(x_train, y_train)}')
#accuracy de entrenamiento test
print(f'accuracy de test de entrenamiento:{svc.score(x_test,y_test)}')
#accuracy de validacion
print(f'accuracy de validacion:{svc.score(x_test_out,y_test_out)}')



# seleccionar modelo arbol de decisiones
arbol = DecisionTreeClassifier()

#entreno el modelo
arbol.fit(x_train, y_train)

#metricas

print('*'*50)
print('arbol de decisiones|')

#accuracy de entrenamiento train
print(f'accuracy de train de entrenamiento:{arbol.score(x_train, y_train)}')
#accuracy de entrenamiento test
print(f'accuracy de test de entrenamiento:{arbol.score(x_test,y_test)}')
#accuracy de validacion
print(f'accuracy de validacion:{arbol.score(x_test_out,y_test_out)}')


# seleccionar modelo Naive Bayes
gnb = GaussianNB()

#entreno el modelo
gnb.fit(x_train, y_train)

#metricas

print('*'*50)
print('Naive Bayes')

#accuracy de entrenamiento train
print(f'accuracy de train de entrenamiento:{gnb.score(x_train, y_train)}')
#accuracy de entrenamiento test
print(f'accuracy de test de entrenamiento:{gnb.score(x_test,y_test)}')
#accuracy de validacion
print(f'accuracy de validacion:{gnb.score(x_test_out,y_test_out)}')

# seleccionar modelo KNN
knn = neighbors.KNeighborsClassifier()

#entreno el modelo
knn.fit(x_train, y_train)

#metricas

print('*'*50)
print('modelo KNN')

#accuracy de entrenamiento train
print(f'accuracy de train de entrenamiento:{knn.score(x_train, y_train)}')
#accuracy de entrenamiento test
print(f'accuracy de test de entrenamiento:{knn.score(x_test,y_test)}')
#accuracy de validacion
print(f'accuracy de validacion:{knn.score(x_test_out,y_test_out)}')


# REGRESIÓN LOGÍSTICA CON VALIDACIÓN CRUZADA

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

for train, test in kfold.split(x, y):
    logreg.fit(x[train], y[train])
    scores_train_train = logreg.score(x[train], y[train])
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = logreg.predict(x_test_out)

print('*'*50)
print('Regresión Logística Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')

# maquina soporte vectorial con validacion cruzada

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train= []
svc = SVC(gamma='auto')

for train, test in kfold.split(x, y):
    svc.fit(x_train, y_train)
    scores_train_train = svc.score(x[train], y[train])
    scores_test_train = svc.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)

print('*'*60)
print('Maquina soporte vectorial Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion_2 = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion_2)
plt.title("Matriz de confución")

precision_2 = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision_2}')

recall_2 = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall_2}')

f1_score_2 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_2}')


#Arbol de decisiones con validacion cruzada

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
arbol = DecisionTreeClassifier()

for train, test in kfold.split(x, y):
    arbol.fit(x_train, y_train)
    scores_train_train = arbol.score(x[train], y[train])
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = arbol.predict(x_test_out)

print('*'*50)
print('arbol de decision Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_3 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_3}')

# modelo naive bayes con Validación cruzada

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
gnb = GaussianNB()

for train, test in kfold.split(x, y):
    gnb.fit(x_train, y_train)
    scores_train_train = gnb.score(x[train], y[train])
    scores_test_train = gnb.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = gnb.predict(x_test_out)

print('*'*50)
print('modelo naive bayes con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {gnb.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_4 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_4}')

# modelo knn con validacion cruzada
kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
knn = neighbors.KNeighborsClassifier()

for train, test in kfold.split(x, y):
    knn.fit(x_train, y_train)
    scores_train_train = knn.score(x[train], y[train])
    scores_test_train = knn.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = knn.predict(x_test_out)

print('*'*50)
print('modelo naive bayes con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {knn.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_5 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_5}')