import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
from sklearn.model_selection import train_test_split


# Загрузка датасета Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)


# one-hot encoding для категориальных признаков
def manipulate_cat_features(data, non_categorial_features, categorial_features):
    dumm = pd.get_dummies(data[categorial_features], columns = categorial_features)
    new_features = list(dumm.columns)
    features = non_categorial_features+new_features
    data = pd.concat([data, dumm], axis=1)
    return data, features

# ВАЖНО - заполнить признаки (категориальные и нет). нефичи должны быть удалены из дф
cat_features = ['Sex', 'Embarked']
non_cat_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
new_data, features = manipulate_cat_features(data, non_cat_features, cat_features)

# Разделяем на обучающую и тестовую выборки (сохраняя DataFrame)
train_df, test_df = train_test_split(new_data, test_size=0.2, random_state=42)

# ВАЖНО!!!
# здесь может быть полезным добавить оверсэмплинг

# оснвной алгоритм - составление ветки по заданному порогу (процент отсекаемого плохого портфеля)
def greedy_exit50proc(data, features, global_target_sum, segment='нет', thresh=0.25, target='Survived', depth=1):
    # дф для записи ветки
    res_df = pd.DataFrame({'Сегмент': [], 'action_list': [], 'Глубина': [], 'Отсекаем': [],
                           'Отсекаем таргет': [], 'Отсекаем нетаргет': [], 'Доля таргета отсекаемых': [],
                           'Всего таргета': [], 'Доля отсекаемых из таргета портфеля': []})
    df = data
    #лист записи действий
    action_list = []
    # считаем просрочки и непросрочки по таргету
    cur_prosr = len(df[df[target] == 1])
    cur_neprosr = len(df[df[target] == 0])

    # глубина ветки (можно добавить выход по глубине)
    depth_it = 0

    # можно включить условие на изменение глубины только по внешним признакам
    # our_segments = segments+segments_industry+segments_regions

    while True:

        # переменные лучших резов для записи в action_list
        flag_find_better = False
        best_threshold = 0
        best_feature = 0
        best_prosr_del_neprosr = cur_prosr / cur_neprosr
        best_prosr = cur_prosr
        best_neprosr = cur_neprosr
        best_znak = '<='

        # условие выхода из цикла - непросрочки == 0 или просрочки >= непросрочки
        if cur_neprosr == 0 or depth_it > depth:
            break

        # перебор всех фичей
        for col_name in features:
            # получение значений признака, перебор
            values = df[col_name]
            thresholds = values.unique()

            steps_len = 20
            if len(thresholds) > steps_len:
                step = (max(values) - min(values)) / steps_len
                thresholds = [min(values) + i * step for i in range(steps_len + 1)]

            for threshold in thresholds:
                # смотрим левую и правую часть
                prosr1 = len(df[(df[col_name] <= threshold) & (df[target] == 1)])
                neprosr1 = len(df[(df[col_name] <= threshold) & (df[target] == 0)])
                prosr2 = len(df[(df[col_name] > threshold) & (df[target] == 1)])
                neprosr2 = len(df[(df[col_name] > threshold) & (df[target] == 0)])
                coef1 = prosr1 / neprosr1 if neprosr1 else prosr1/10
                coef2 = prosr2 / neprosr2 if neprosr2 else prosr2/10
                coef = max(coef1, coef2)
                prosr = prosr1 if abs(coef - coef1) < 0.01 else prosr2
                neprosr = neprosr1 if abs(coef - coef1) < 0.01 else neprosr2
                znak = '<=' if abs(coef - coef1) < 0.01 else '>'
                if coef > best_prosr_del_neprosr and prosr >= thresh * global_target_sum:
                    if not flag_find_better:
                        flag_find_better = True
                    best_threshold = threshold
                    best_prosr_del_neprosr = coef
                    best_feature = col_name
                    best_prosr = prosr
                    best_neprosr = neprosr
                    best_znak = znak
                    """
                    if (best_threshold==0 and best_feature=='2.43'):
                        print(best_prosr, best_neprosr, best_znak)
                    """
                    # print(best_znak, best_threshold, best_feature, best_prosr, best_neprosr)
        if not flag_find_better:
            break
        # доп_условие на увеличение глубины (социнженерия)
        """
        if best_feature not in our_segments and best_feature not in [row[0] for row in action_list]:
            depth+=1
        """
        # увеличение глубины итерации, сохранение резов очередного спуска в дф ветки
        depth_it += 1
        action_list.append([best_feature, best_threshold, best_znak])
        it_list = action_list.copy()
        res_df = pd.concat([res_df, pd.DataFrame({'Сегмент': [segment], 'action_list': [it_list],
                                                  'Глубина': [len(action_list)],
                                                  'Отсекаем': [best_neprosr + best_prosr],
                                                  'Отсекаем таргет': [best_prosr], 'Отсекаем нетаргет': [best_neprosr],
                                                  'Всего таргета': global_target_sum,
                                                  'Доля таргета отсекаемых': [best_prosr / (best_prosr + best_neprosr)],
                                                  'Доля отсекаемых из таргета портфеля': [
                                                      best_prosr / global_target_sum]})], axis=0)
        if best_znak == '<=':
            df = df[df[best_feature] <= best_threshold]
        else:
            df = df[df[best_feature] > best_threshold]
        # exec(f"""df = df[df[best_feature]{best_znak}best_threshold]""")
        # print(len(df))
        cur_prosr = best_prosr
        cur_neprosr = best_neprosr
    return res_df

# критерий приемлемости ветки (риск отсекаемого портфеля)
def branch_criteria(res_df):
    risk = 0.9
    if len(res_df)!=0 and res_df.iloc[-1]['Доля таргета отсекаемых']>=risk:
        return True
    return False

# функция бинарного поиска наибольшего порога по заданному риску (риск задается в функции выше)
# сегменты - это части датафрейма. можно закинуть их в признаки, а можно рассматривать как отдельные части
# например - смотреть заявки только из московского региона
def steps_by_best_threshes(segments_dict, global_target_sum):
    df_results = pd.DataFrame({'Сегмент': [], 'action_list': [], 'Глубина': [], 'Отсекаем': [],
                               'Отсекаем таргет': [], 'Отсекаем нетаргет': [], 'Доля таргета отсекаемых': [],
                               'Всего таргета': [], 'Доля отсекаемых из таргета портфеля': []})

    for segment_name, df in segments_dict.items():
        it_df = df
        it = 0
        while True:
            # подбор наилучшего порога итерации
            const_low = 0.05
            low = const_low
            high = 1
            res_df = pd.DataFrame()
            # ограничение для случая отсутствия нужного порога
            for _ in range(10):
                cur = (low + high) / 2
                res_df_it = greedy_exit50proc(it_df, features, global_target_sum, segment=str(it) + ' ' + segment_name,
                                           thresh=cur, depth=3)
                flag_is_validate = branch_criteria(res_df_it)
                if flag_is_validate:
                    if high - low < 0.01:
                        break
                    res_df = res_df_it
                    low = cur
                else:
                    if high - low < 0.01 and low == const_low:
                        break
                    high = cur
                print(high, low)
            if len(res_df)==0:
                break
            else:
                action_list = res_df.iloc[-1]['action_list']
                # print(action_list)
                df_results = pd.concat([df_results, res_df])
                # s = 'it_df = it_df[~('+'&'.join([f'(it_df["{lst[0]}"]{lst[2]+str(lst[1])})' for lst in action_list])+')]'
                s = 'it_df[~(' + '&'.join([f'(it_df["{lst[0]}"]{lst[2] + str(lst[1])})' for lst in action_list]) + ')]'
                # it_df = it_df[~('&'.join([f'(it_df["{lst[0]}"]{lst[2]+str(lst[1])})' for lst in action_list])]
                it_df = eval(s)
                it += 1
    return df_results

segments_dict = {'all':train_df}
df_results = steps_by_best_threshes(segments_dict, len(train_df[train_df['Survived']==1]))

print(df_results)

# считаем данные для тестовой выборки
def cut_test_df(df_results, test_df):
    it_df = test_df
    for i in range(len(df_results)):
        action_list = df_results.iloc[i]['action_list']
        s = '&'.join([f'(it_df["{lst[0]}"]{lst[2] + str(lst[1])})' for lst in action_list])
        it_df[f'new_surv_{i}'] = eval(s)
    return it_df

test_df = cut_test_df(df_results.drop_duplicates('Сегмент', keep='last'), test_df)
test_df['new_surv'] = test_df.filter(like='new_surv').any(axis=1)
TP = len(test_df[(test_df['Survived']==test_df['new_surv']) & (test_df['Survived']==1)])
TN = len(test_df[(test_df['Survived']==test_df['new_surv']) & (test_df['Survived']==0)])
FN = len(test_df[(test_df['Survived']!=test_df['new_surv']) & (test_df['Survived']==1)])
FP = len(test_df[(test_df['Survived']!=test_df['new_surv']) & (test_df['Survived']==0)])

acc = (TP+TN)/(TP+TN+FP+FN)
prec = TP / (TP+FP)
rec = TP / (TP+FN)
f1_score = 2*(prec*rec)/(prec+rec)

print(f'acc = {acc}, prec = {prec}, rec = {rec}, f1-score = {f1_score}')


