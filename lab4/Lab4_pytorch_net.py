import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd


# Cоздадим простую нейронную сеть.
# Для этого объявим свой класс 
# наш класс будем наследовать от nn.Module, который включает большую часть необходимого нам функционала

class NNet(nn.Module):
    # для инициализации сети на вход нужно подать размеры (количество нейронов) входного, скрытого и выходного слоев
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет слои и позволяет запускать их одновременно
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.Tanh(),                       # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Tanh()
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred



# Данные как и в прошлых работах загружаем и выделяем в отдельные переменные
# X - признаки
# y - правильные ответы, их кодируем числами
# X и y преобразуем в pytorch тензоры
df = pd.read_csv('data.csv')
X = torch.Tensor(df.iloc[0:100, 0:3].values)
y = df.iloc[0:100, 4].values
y = torch.Tensor(np.where(y == "Iris-setosa", 1, -1).reshape(-1,1))


# Параметры нашей сети определяются данными.
# Размер входного слоя - это количество признаков в задаче, т.е. количество 
# столбцов в X.
inputSize = X.shape[1] # количество признаков задачи 

# Размер (количество нейронов) в скрытом слое задается нами, четких правил как выбрать
# этот параметр нет, это открытая проблема в нейронных сетях.
# Но есть общий принцип - чем сложнее зависимость (разделяющая поверхность), 
# тем больше нейронов должно быть в скрытом слое.
hiddenSizes = 3 #  число нейронов скрытого слоя 

# Количество выходных нейронов равно количеству классов задачи.
# Но для двухклассовой классификации можно задать как один, так и два выходных нейрона.
outputSize = 1


# Создаем экземпляр нашей сети
net = NNet(inputSize,hiddenSizes,outputSize)

# Веса нашей сети содержатся в net.parameters() 
for param in net.parameters():
    print(param)

# Можно вывести их с названиями
for name, param in net.named_parameters():
    print(name, param)


# Посчитаем ошибку нашего не обученного алгоритма
# градиенты нужны только для обучения, тут их можно отключить, 
# это немного ускорит вычисления
with torch.no_grad():
    pred = net.forward(X)

# Так как наша сеть предсказывает числа от -1 до 1, то ее ответы нужно привести 
# к значениям меток
pred = torch.Tensor(np.where(pred >=0, 1, -1).reshape(-1,1))

# Считаем количество ошибочно классифицированных примеров
err = sum(abs(y-pred))/2
print(err) # до обучения сеть работает случайно, как бросание монетки

# Для обучения нам понадобится выбрать функцию вычисления ошибки
lossFn = nn.MSELoss()

# и алгоритм оптимизации весов
# при создании оптимизатора в него передаем настраиваемые параметры сети (веса)
# и скорость обучения
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# В цикле обучения "прогоняем" обучающую выборку
# X - признаки
# y - правильные ответы
# epohs - количество итераций обучения

epohs = 100
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred, y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%10==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >=0, 1, -1).reshape(-1,1))
err = sum(abs(y-pred))/2
print('\nОшибка (количество несовпавших ответов): ')
print(err) # обучение работает, не делает ошибок или делает их достаточно мало

###############################################################################
###############################################################################
# Но что делать, если нужно предсказывать не два, а больше классов?
# При количестве классов больше двух нам нужно по другому кодировать метки классов.
# Теперь на каждый класс у нас должен быть свой нейрон, соответственно 
# переменная, содержащая ответы теперь будет не вектором, а матрицей.

df = pd.read_csv('data_3class.csv')
X = torch.Tensor(df.iloc[:, 0:3].values) # Признаки остаются без изменений
y = df.iloc[:, 4].values                # ответы берем из четвертого столбца как и раньше
# Но теперь классы кодируются иначе.
# Метка класса будет состоять из 3-х бит, где каждому классу соответствует одна единица
# такой подход называтеся one-hot encoding
print(pd.get_dummies(y).iloc[0:150:50,:])

labels = pd.get_dummies(y).columns.values  #  отдельно сохраним названия классов
y = torch.Tensor(pd.get_dummies(y).values) # сохраним значения


# в структуру нашей сети необходимо внести изменения
# гиперболический тангенс в выходном слое теперь нам не подходит,
# т.к. мы ожидаем 0 или 1 на выходе нейронов, то нам подойдет Сигмоида в качестве
# функции активации выходного слоя
class NNet_multiclass(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.Tanh(),                       # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Sigmoid(),
                                    # nn.Softmax(dim=1) # вместо сигмоиды можно использовать softmax
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred

# задаем параметры сети
inputSize = X.shape[1] # количество признаков задачи 
hiddenSizes = 3     #  число нейронов скрытого слоя 
outputSize = y.shape[1] # число нейронов выходного слоя равно числу классов задачи

net = NNet_multiclass(inputSize,hiddenSizes,outputSize)

# В задачах многоклассовой классификации используется ошибка,
# вычисляющая разницу между предсказанной вероятностью появления класса 
# и истинной вероятностью его появления для конкретного примера
lossFn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

epohs = 100
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred, y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%10==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

pred_lbl = labels[pred.max(1).indices] # предсказанные названия классов
true_lbl = labels[y.max(1).indices]    # истинные названия классов

err = sum(pred_lbl!=true_lbl) # все что не совпало, считаем ошибками
print('\nОшибка (количество несовпавших ответов): ')
print(err)                    # ошибок много, но попробуйте увеличить число скрытых нейронов


###############################################################################
###############################################################################
# Теперь посмотрим что изменится в структуре нейронной сети, если нам нужно решить задачу регрессии
# Задача регрессии заключается в предсказании значений одной переменной
# по значениям другой (других).  
# От задачи классификации отличается тем, что выходные значения нейронной сети не 
# ограничиваются значениями меток классов (0 или 1), а могут лежать в любом 
# диапазоне чисел.
# Примерами такой задачи можгут быть предсказание цен на жилье, курсов валют или акций,
# количества выпадающих осадков или потребления электроэнергии.

# Рассмотрим задачу предсказания прочности бетона (измеряется в мегапаскалях)
df = pd.read_csv('concrete_data.csv')

# Известно что прочность бетона зависит от многих факторов - количесва цемента, 
# используемых добавок, 
# Cement - количество цемента в растворе kg/m3
# Blast Furnace Slag - количество шлака в растворе kg/m3 
# Fly Ash - количетво золы в растворе kg/m3
# Water - количетво воды в растворе kg/m3
# Superplasticizer - количетво пластификатора в растворе kg/m3
# Coarse Aggregate - количетво крупного наполнителя в растворе kg/m3
# Fine Aggregate - количетво мелкого наполнителя в растворе kg/m3
# Age - возраст бетона в днях
# Concrete compressive strength -  прочность бетона MPa


X = torch.Tensor(df.iloc[:, [0]].values) # выделяем признаки (независимые переменные)
y = torch.Tensor(df.iloc[:, -1].values)  #  предсказываемая переменная, ее берем из последнего столбца

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(df.iloc[:, [0]].values, df.iloc[:, -1].values, marker='o')

# Чтобы выходные значения сети лежали в произвольном диапазоне,
# выходной нейрон не должен иметь функции активации или 
# фуннкция активации должна иметь область значений от -бесконечность до +бесконечность

class NNet_regression(nn.Module):
    
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, out_size) # просто сумматор
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred

# задаем параметры сети
inputSize = X.shape[1] # количество признаков задачи 
hiddenSizes = 3   #  число нейронов скрытого слоя 
outputSize = 1 # число нейронов выходного слоя

net = NNet_regression(inputSize,hiddenSizes,outputSize)

# В задачах регрессии чаще используется способ вычисления ошибки как разница квадратов
# как усредненная разница квадратов правильного и предсказанного значений (MSE)
# или усредненный модуль разницы значений (MAE)
lossFn = nn.L1Loss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

epohs = 1
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred.squeeze(), y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%1==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

print('\nПредсказания:') # Иногда переобучается, нужно запускать обучение несколько раз
print(pred[0:10])
err = torch.mean(abs(y - pred.T).squeeze()) # MAE - среднее отклонение от правильного ответа
print('\nОшибка (MAE): ')
print(err) # измеряется в MPa


# Построим график
plt.figure()
plt.scatter(df.iloc[:, [0]].values, df.iloc[:, -1].values, marker='o')

with torch.no_grad():
    y1 = net.forward(torch.Tensor([100]))
    y2 = net.forward(torch.Tensor([600]))

plt.plot([100,600], [y1.numpy(),y2.numpy()],'r')


# Задание 1

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Загрузка данных
df = pd.read_csv('dataset_simple.csv')
X = df.iloc[:, 0:2].values  # Признаки: возраст и доход
y = df.iloc[:, 2].values    # Целевая переменная: will_buy

# Визуализация исходных данных
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Не купит', alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Купит', alpha=0.6)
plt.title('Исходные данные')
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.legend()
plt.grid(True)
plt.show()

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Преобразование данных в тензоры PyTorch
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train).reshape(-1, 1)
y_test = torch.Tensor(y_test).reshape(-1, 1)

# Определение архитектуры нейронной сети
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.layers(X)

# Параметры сети
inputSize = X_train.shape[1]
hiddenSizes = 5
outputSize = 1

# Создание экземпляра сети
net = NNet(inputSize, hiddenSizes, outputSize)

# Функция потерь и оптимизатор
lossFn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Обучение сети
epochs = 100
loss_history = []

for i in range(epochs):
    optimizer.zero_grad()
    pred = net(X_train)
    loss = lossFn(pred, y_train)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if i % 10 == 0:
        print(f'Ошибка на {i+1} итерации: {loss.item()}')

# Визуализация процесса обучения
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), loss_history)
plt.title('График обучения')
plt.xlabel('Эпоха')
plt.ylabel('Ошибка')
plt.grid(True)
plt.show()

# Оценка модели на тестовых данных
with torch.no_grad():
    test_pred = net(X_test)
    test_pred_class = (test_pred > 0.5).float()
    accuracy = (test_pred_class == y_test).float().mean()
    print(f'\nТочность модели на тестовых данных: {accuracy.item()*100:.2f}%')

# Визуализация границы принятия решений
def plot_decision_boundary(X, y, model):
    h = 0.02  # шаг сетки
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Прогнозируем для каждой точки сетки
    Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]))
    Z = (Z > 0.5).float().numpy()
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['red', 'blue']))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Не купит', alpha=0.6)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Купит', alpha=0.6)
    plt.title('Граница принятия решений')
    plt.xlabel('Возраст (нормализованный)')
    plt.ylabel('Доход (нормализованный)')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(X_test.numpy(), y_test.numpy().flatten(), net)

# Вывод количества ошибок
errors = (test_pred_class != y_test).sum()
print(f'Количество ошибок: {errors.item()}')































































































# Пасхалка, кто найдет и сможет объяснить, тому +
# X = np.hstack([np.ones((X.shape[0], 1)), df.iloc[:, [0]].values])

# y = df.iloc[:, -1].values

# w = np.linalg.inv(X.T @ X) @ X.T @ y

# predicted = X @ w

# print(predicted)