
import pandas as pd
df = pd.read_csv("surv.csv")

print("Исходный размер:", df.shape)

# 1. удаление пропусков
before = df.shape[0]
df = df.dropna()
after = df.shape[0]
print("Удалено строк с пропусками:", before - after)

# 2. удаление строк с некорректными значениями
q_cols = [f"Q{i}" for i in range(1, 29)]

before = df.shape[0]

df = df[
    (df["instr"].between(1, 3)) &
    (df["class"].between(1, 13)) &
    (df["difficulty"].between(1, 5))
]
for col in q_cols:
    df = df[df[col].between(1, 5)]

after = df.shape[0]
print("Удалено строк с некорректными значениями:", before - after)

# 3. поиск и удаление анкет с одинаковыми ответами по всем вопросам
before = df.shape[0]

# выбираем анкеты, где все ответы одинаковые
same_answer_rows = df[df[q_cols].nunique(axis=1) == 1]

# добавляем столбец со значением этого одинакового ответа
same_answer_rows["same_value"] = same_answer_rows[q_cols[0]]

# считаем, сколько таких анкет для каждого значения ответа
same_answer_counts = same_answer_rows["same_value"].value_counts().sort_index()

print("Количество анкет с одинаковыми ответами по величине ответа:")
print(same_answer_counts)

# удаляем такие анкеты из основного датафрейма
df = df[df[q_cols].nunique(axis=1) > 1]

after = df.shape[0]
print("Удалено анкет с одинаковыми ответами:", before - after)
#4. удаление анкет с нулевым посещением
before = df.shape[0]
df = df[
    (df["attendance"].between(1, 4))]
after = df.shape[0]
print("Удалено анкет с нулевым посещением:", before - after)

print("Итоговый размер:", df.shape)
print(df.head())

df.to_csv("surv-clean.csv", index=False)



dff = pd.read_csv("surv-clean.csv")

# выбираем только оценки (Q1–Q28)
cols = [c for c in dff.columns if "Q" in c]


# функция удаления выбросов
def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return data[~((data < lower) | (data > upper)).any(axis=1)]


clean_dff = remove_outliers(dff[cols])

print("Размер до:", dff.shape)
print("После удаления:", clean_dff.shape)



import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = clean_dff.corr()

plt.figure(figsize=(12,10))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7}   # размер чисел внутри ячеек
)

plt.title("Матрица корреляций")
plt.show()


subjects = df['class'].unique()

for s in subjects[:3]:
    subset = df[df['class'] == s][cols]
    corr_matrix = subset.corr()

    # убираем единицы на диагонали
    corr_values = corr_matrix.values
    n = corr_values.shape[0]

    mean_corr = (corr_values.sum() - n) / (n * n - n)

    print(f"\nПредмет {s}")
    print("Средняя корреляция:", mean_corr)
    
    
    
    
teacher_stats = dff.groupby("instr")[cols].mean()

print("\nОписательная статистика преподавателей:")
print(teacher_stats.describe())

teacher_stats["rating"] = teacher_stats.mean(axis=1)

print("\nРейтинг преподавателей:")
print(teacher_stats.sort_values("rating", ascending=False))



import pandas as pd
import matplotlib.pyplot as plt

subject_stats = df.groupby("class")[cols].mean()

subject_stats["rating"] = subject_stats.mean(axis=1)

print("\nРейтинг предметов:")
print(subject_stats.sort_values("rating", ascending=False))

# график средней оценки предметов
plt.figure(figsize=(10, 6))
subject_stats["rating"].plot(kind="bar")

plt.title("Средняя оценка предметов")
plt.xlabel("Предмет")
plt.ylabel("Средняя оценка")
plt.xticks(rotation=0)

plt.show()
