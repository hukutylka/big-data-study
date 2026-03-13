import pandas as pd
values_df=pd.read_csv("surv.csv",delimiter=',')
print("Массив данных")
print(values_df.info)

import numpy as np
header = np.genfromtxt("surv.csv", delimiter=",", max_rows=1, dtype=str)

# считываем данные без первой строки
data = np.genfromtxt("surv.csv", delimiter=",", skip_header=1)

for i in range(1, data.shape[1]):
    print("Признак:", header[i])

    x = data[:, i]
    

    size = np.size(x)
    min_x = np.min(x)
    max_x = np.max(x)
    sum_x = np.sum(x)
    sum2_x = np.dot(x, x)

    mean_x = sum_x / size

    var_x = np.var(x)
    sdm_x = var_x * size
    std_x = np.sqrt(var_x)
    varcoef_x = std_x / mean_x

    print("Минимум: ", min_x)
    print("Максимум: ", max_x)
    print("Среднее значение: ", mean_x)
    print("Дисперсия: ", var_x)
    print("Среднеквадратичное отклонение: ", std_x)
    print("Коэффициент вариации: ", varcoef_x)
    print("Медиана: ", np.median(x))
    print("Квантили [25, 50, 75]: ", np.percentile(x, [25, 50, 75]))
    print("Размах: ", max_x - min_x)
    print(" ")
    
    
    
#Удаление строк с некорректными значениями
q_cols = [f"Q{i}" for i in range(1, 29)]
values_df = values_df[
    (values_df["instr"].between(1, 3)) &
    (values_df["class"].between(1, 13)) &
    (values_df["attendance"].between(1, 4)) &
    (values_df["difficulty"].between(1, 5))
]
for col in q_cols:
    values_df = values_df[values_df[col].between(1, 5)]

print("Размер таблицы:", values_df.shape)


# 3. Удаление анкет с одинаковыми ответами по всем вопросам

values_df = values_df[values_df[q_cols].nunique(axis=1) > 1]

# результат
print("Размер очищенной таблицы:", values_df.shape)
print(values_df.info)