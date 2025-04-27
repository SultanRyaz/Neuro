# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:30:32 2025

@author: AM4
"""

# Попробуем обучить один нейрон на задачу классификации двух классов
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Для 3D визуализации

# Считываем данные 
df = pd.read_csv('data.csv')

# смотрим что в них
print(df.head())

# выделим целевую переменную в отдельную переменную
y = df.iloc[:, 4].values

# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, -1)

# возьмем три признака для обучения
X = df.iloc[:, [0, 1, 2]].values  # Теперь используем три признака

# Визуализация данных в 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], color='red', marker='o')
ax.scatter(X[y==-1, 0], X[y==-1, 1], X[y==-1, 2], color='blue', marker='x')
ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_zlabel('Признак 3')
plt.title('Исходные данные (3 признака)')
plt.show()

# Модифицируем функцию нейрона для работы с тремя признаками
def neuron(w, x):
    # w[0] - смещение (bias), w[1], w[2], w[3] - веса для признаков
    if (w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[0]) >= 0:
        predict = 1
    else: 
        predict = -1
    return predict

# Процедура обучения для трех признаков
w = np.random.random(4)  # Теперь 4 веса: w0 + 3 признака
eta = 0.01  # скорость обучения
w_iter = [] # сохраняем веса для визуализации

for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w, xi)   
    # Обновляем веса для всех трех признаков
    w[1:] += (eta * (target - predict)) * xi
    w[0] += eta * (target - predict)
    if j % 10 == 0:
        w_iter.append(w.copy())  # сохраняем копию текущих весов

# Оценка качества
sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w, xi) 
    sum_err += (target - predict)/2

print("Всего ошибок:", sum_err)

# Визуализация разделяющей плоскости
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Отображаем данные
ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], color='red', marker='o')
ax.scatter(X[y==-1, 0], X[y==-1, 1], X[y==-1, 2], color='blue', marker='x')

# Создаем сетку для плоскости
x1_range = np.linspace(min(X[:,0]), max(X[:,0]), 10)
x2_range = np.linspace(min(X[:,1]), max(X[:,1]), 10)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Уравнение плоскости: w0 + w1*x1 + w2*x2 + w3*x3 = 0 => x3 = -(w0 + w1*x1 + w2*x2)/w3
for i, weights in enumerate(w_iter):
    if weights[3] != 0:  # избегаем деления на ноль
        x3_plane = -(weights[0] + weights[1]*x1_grid + weights[2]*x2_grid) / weights[3]
        plane = ax.plot_surface(x1_grid, x2_grid, x3_plane, alpha=0.5, color='gray')
        plt.pause(0.5)
        plane.remove()  # удаляем плоскость для следующей итерации

# Финальная плоскость
x3_plane = -(w[0] + w[1]*x1_grid + w[2]*x2_grid) / w[3]
ax.plot_surface(x1_grid, x2_grid, x3_plane, alpha=0.5, color='green')
ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_zlabel('Признак 3')
plt.title('Финальная разделяющая плоскость')
plt.show()