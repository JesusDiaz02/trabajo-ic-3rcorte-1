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

#