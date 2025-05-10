

## !!!!Программа должна запускаться в режиме интерпретатора построчно или блоками кода!!!!!!!
## !!!!!!!!   Чтиайте комментарии и только после этого запускайте код !!!!!!!!!!!!!


## Перед выполнением убедитесь что у вас установлен pytorch
#%%
import torch 
import numpy as np
import pandas as pd

# Основная структура данных pytorch - тензоры
# Основные отличия pytorch-тензоров от numpy массивов:
# 1 Место их хранения можно задать ( память CPU или GPU )
# 2 В отношении тензоров можно задать вычисление и отслеживание градиентов


# Создавать тензоры можно разными способами:
# Пустой тензор
a = torch.empty(5, 3)
print(a) # Пустой тензор обычно содержит "мусор"
#%%

b = torch.Tensor(5, 3)
print(b) # Пустой тензор обычно содержит "мусор"

#%%
# тензор с нулями
a = torch.zeros(5, 3)
print(a) # тензор заполняем нулями
#%%
b = torch.ones(5, 3)
print(b) # тензор заполняем единицами
#%%
# можно задать тип тензора
# pytorch поддерживает следующие основные типы тензоров
# 32-bit с плавающей точкой - torch.float32 или torch.float
# 64-bit с плавающей точкой - torch.float64 или torch.double
# 16-bit с плавающей точкой - torch.float16 или torch.half
# 8-bit целый беззнаковый - torch.uint8 
# 8-bit целый - torch.int8
# 16-bit целый - torch.int16 или torch.short
# 32-bit целый - torch.int32 или torch.int
# 64-bit целый - torch.int64 или torch.long
# Булевский бинарный - torch.bool
# квантизованный целый беззнаковый 8-bit - torch.quint8
# квантизованный целый 8-bit - torch.qint8
# квантизованный целый 32-bit - torch.qint32
# квантизованный целый 4-bit - torch.quint4x2

с = torch.ones(5, 3, dtype=torch.int32)
print(с) # тензор заполняем единицами

# кроме того тензоры, используемые для вычислений на GPU имеют свои типы данных
# torch.cuda.FloatTensor, 
# torch.cuda.DoubleTensor, 
# torch.cuda.HalfTensor, 
# torch.cuda.ByteTensor,
# torch.cuda.ShortTensor, 
# torch.cuda.IntTensor, 
# torch.cuda.LongTensor, 
# torch.cuda.BoolTensor


# Можно преобразовать созданный тензор в другой тип
b = b.to(dtype=torch.int32)
print(b)


# Тензор можно заполнить случайными числами
a = torch.rand(5, 3)
print(a) # распределеными по равномерному закону распределения

b = torch.randn(5, 3)
print(b) # распределеными по нормальному закону распределения

# или можем явно указать нужные значения
a = torch.Tensor([[1,2],[3,4]])
print(a) 

# Наиболее часто используемые методы создания тензоров

#    torch.rand: случайные значения из равномерного распределения
#    torch.randn: случайные значения из нормального распределения
#    torch.eye(n): единичная матрица
#    torch.from_numpy(ndarray): тензор на основе ndarray NumPy-массива
#    torch.ones : тензор с единицами
#    torch.zeros_like(other): тензор с нулями такой же формы, что и other
#    torch.range(start, end, step): 1-D тензор со значениями между start и end с шагом steps


# тензоры можно преобразовать к Numpy массивам
# !!! но нужно не забывать про копирование данных !!!
# если не указать метод copy(), обе переменные будут 
# указывать на одну область памяти и изменения в одной из переменных будут 
# вызывать изменения в другой
d = a.numpy().copy()

# тензоры можно "слайсить" как и Numpy массивы, списки или строки
print(a[1,:].numpy())

# Понять можем ли мы использовать графический ускоритель для вычислений поможет функция
print(torch.cuda.is_available())

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu' 

# device теперь хранит тип устройства на котором можно проводить вычисления
# создавать тензоры можно в памяти видеокарты, если CUDA доступна
a = torch.Tensor([[1,2],[3,4]]).to(device) 

#%%
# Для обучения нейронных сетей некоторые тензоры 
# должны отслеживать градиенты (происходящие с ними изменения)
a = torch.randn(2, 2, requires_grad=False)
a.requires_grad

a.requires_grad=True # Теперь все операци над тензором a будут отслеживаться
print(a)

# выполненные операции хранятся в свойстве grad_fn
print(a.grad_fn) # пока с тензором ничего не делали

# Выполним какую-нибудь операцию с тензором:
a = a + 2
print(a)

# теперь grad_fn, который хранит информацию об операции с тензором
print(a.grad_fn)

a = a * 2
print(a)

print(a.grad_fn)
#%%
# Все это нужно для вычисления градиентов
# посмотрим детально как это происходит
# создадим простую последовательность вычислений

x = torch.zeros(2, 2, requires_grad=True)
y = x + 3
z = y**2
out = z.mean()
print(z)
print(out)

# теперь воспользовавшись правилом дифференцирования сложных функций
# продифференцируем полученную последовательность
# для этого вызовем метод .backward(), который проведет дифференцирование в обратном порядке 
# до изначально заданного тензора x
out.backward()
print(x.grad) # градиенты d(out)/dx

# метод .backward() без аргументов работает только для скаляра (например ошибки нейросети)
# чтобы вызвать его для многомерного тензора, внего в качестве параметра необходимо 
# передать значения градиентов с "предыдущего" блока  
x = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
print(z)
z = x**2
print(z) 
z.backward(torch.ones(2,2))
print(x.grad) # градиенты d(z)/dx = 2*x


###########                                                        ############
###########    Обучение линейного алгоритма на основе нейронов    #############
###########                                                       #############

# Для работы с нейронными сетями pytorch предоставляет широкий набор инструментов:
# слои, функции активации, функционалы потерь, оптимизаторы
import torch.nn as nn

# Попробуем обучить нейроны решать искусственную задачу
# Создадим 2 тензора:  1 входной размером (10, 3) - наша обучающая выборка 
# из 10 примеров с 3 признаками

X = torch.randn(10, 3)

#  2 выходной тензор - значения, которые мы хотим предсказывать нашим алгоритмом
y = torch.randn(10, 2)


# создадим 3 сумматора без функци активации, это называется полносвязный слой (fully connected layer)
# Отсутствие фунций активаци на выходе сумматора эквивалетно наличию  линейной активации
linear = nn.Linear(3, 2)

# при создании веса и смещения инициализируются автоматически
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# выберем вид функции ошибки и оптимизатор
# фунция ошибки показывает как сильно ошибается наш алгоритм в своих прогнозах
lossFn = nn.MSELoss() # MSE - среднеквадратичная ошибка, вычисляется как sqrt(sum(y^2 - yp^2))

# создадим оптимизатор - алгоритм, который корректирует веса наших сумматоров (нейронов)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01) # lr - скорость обучения

# прямой проход (пресказание) выглядит так:
yp = linear(X)

# имея предсказание можно вычислить ошибку
loss = lossFn(yp, y)
print('Ошибка: ', loss.item())

# и сделать обратный проход, который вычислит градиенты (по ним скорректируем веса)
loss.backward()

# градиенты по параметрам
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# далее можем сделать шаг оптимизации, который изменит веса 
# на сколько изменится каждый вес зависит от градиентов и скорости обучения lr
optimizer.step()

# итерационно повторяем шаги
# в цикле (фактически это и есть алгоритм обучения):
for i in range(0,10):
    pred = linear(X)
    loss = lossFn(pred, y)
    print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())
    loss.backward()
    optimizer.step()
    

# Задание 1

print("\nЗадание 1\n")

# 1. Создаем тензор целочисленного типа со случайным значением
x = torch.randint(1, 10, (1,), dtype=torch.int32)
print("Исходный тензор x:", x)

# 2. Преобразуем тензор к типу float32
x = x.to(dtype=torch.float32)

# Устанавливаем requires_grad=True для вычисления градиента
x.requires_grad = True

# 3. Проводим операции с тензором
n = 2  # Нечетный вариант

# Возведение в степень n
x_pow = x ** n
print("x после возведения в степень:", x_pow)

# Умножение на случайное значение от 1 до 10
random_multiplier = torch.rand(1) * 9 + 1  # Случайное значение в диапазоне [1, 10]
x_mult = x_pow * random_multiplier
print("x после умножения на случайное значение:", x_mult)

# Взятие экспоненты
x_exp = torch.exp(x_mult)
print("x после взятия экспоненты:", x_exp)

# 4. Вычисляем производную
x_exp.backward()  # Вычисляем градиент
print("Производная d(x_exp)/dx:", x.grad)

# Задание 2
print("\nЗадание 2\n")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt  # matplotlib для построения графиков
import torch.nn as nn  # Модуль для создания нейронных сетей
import torch.optim as optim  # Модуль для оптимизации

# Считываем данные
df = pd.read_csv('data.csv')

# три столбца - это признаки, четвертый - целевая переменная (то, что мы хотим предсказывать)

# выделим целевую переменную в отдельную переменную
y = df.iloc[:, 4].values

# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, -1)

# возьмем три признака для обучения
X = df.iloc[:, [0, 1, 2]].values  # теперь используем три признака

# Преобразуем данные в тензоры PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)  # Признаки
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Целевые значения (преобразуем в размер [n, 1])

# Определим модель (линейный слой)
class SimpleNeuron(nn.Module):
    def __init__(self):
        super(SimpleNeuron, self).__init__()
        self.linear = nn.Linear(3, 1)  # Линейный слой с 3 входами и 1 выходом

    def forward(self, x):
        return self.linear(x)

# Создаем модель
model = SimpleNeuron()

# Определяем функцию потерь и оптимизатор
criterion = nn.MSELoss()  # Среднеквадратичная ошибка
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Стохастический градиентный спуск

# Список для сохранения весов
weights_history = []

# Обучение модели
num_iterations = 100
for iteration in range(num_iterations):
    # Прямой проход
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Обратный проход и оптимизация
    optimizer.zero_grad()  # Обнуляем градиенты
    loss.backward()  # Вычисляем градиенты
    optimizer.step()  # Обновляем веса

    # Сохраняем веса на каждой 10-й итерации
    if (iteration + 1) % 10 == 0:
        weights_history.append([model.linear.weight.data.numpy().copy(), model.linear.bias.data.numpy().copy()])

    # Выводим данные каждые 10 итераций
    if (iteration + 1) % 10 == 0:
        print(f'Итерация [{iteration + 1}/{num_iterations}], Ошибка: {loss.item():.4f}')
        print(f'Веса: {model.linear.weight.data.numpy()}')
        print(f'Смещение: {model.linear.bias.data.numpy()}\n')

# После обучения визуализируем результаты
# Преобразуем предсказания в numpy для визуализации
with torch.no_grad():  # Отключаем вычисление градиентов
    predicted = model(X_tensor).numpy()

# Визуализация данных и разделяющей гиперплоскости
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Визуализируем данные
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='red', marker='o', label='Iris-setosa')
ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], color='blue', marker='x', label='Iris-versicolor')
ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_zlabel('Признак 3')

# Устанавливаем пределы осей
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
z_min, z_max = X[:, 2].min(), X[:, 2].max()
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_zlim([z_min, z_max])

plt.legend()

# Визуализируем разделяющие гиперплоскости
x1_range = np.linspace(x_min, x_max, 10)
x2_range = np.linspace(y_min, y_max, 10)
x1, x2 = np.meshgrid(x1_range, x2_range)

for i, (weight, bias) in enumerate(weights_history):
    # Уравнение гиперплоскости: w1*x1 + w2*x2 + w3*x3 + b = 0
    # Для визуализации создаем сетку точек по x1 и x2, а x3 вычисляем
    x3 = -(weight[0, 0] * x1 + weight[0, 1] * x2 + bias[0]) / weight[0, 2]

    # Рисуем гиперплоскость
    ax.plot_surface(x1, x2, x3, alpha=0.5, color='gray')
    plt.pause(1)  # Пауза для анимации
    # Добавляем надпись END у последней плоскости
    if i == len(weights_history) - 1:
        ax.text(x1_range[-1] - 0.3, x2_range[-1], x3[-1, -1], 'END', size=14, color='red')

plt.show()
