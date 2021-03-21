import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import scipy.stats


def data_analyse():
    data = pd.read_csv('BankChurners.csv')

    data.columns = data.columns.str.replace(' ', '')
    categories = {'Attrition_Flag': {'Attrited Customer': 1, 'Existing Customer': 0},
                  'Gender': {'F': 1, 'M': 0}}

    data = data.replace(categories)
    data_binary = data[['Attrition_Flag', 'Gender']]
    data = data.drop(columns=['Attrition_Flag'])

    cat_cols = []
    num_cols = []

    for column in data.columns:
        if data[column].nunique() < 8:
            cat_cols.append(column)
        else:
            num_cols.append(column)

    cols_final = data.columns

    for column in cols_final:
        if column in cat_cols:
            data[column] = data[column].fillna(value=data[column].value_counts().index[0])
        else:
            data[column] = data[column].fillna(value=data[column].mean())

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data[cat_cols])
    data_cat = enc.transform(data[cat_cols])
    data_cat.columns = enc.get_feature_names(cat_cols)

    data_cat = pd.DataFrame.sparse.from_spmatrix(data_cat)
    data_cat.columns = enc.get_feature_names(cat_cols)
    data = data.drop(cat_cols, axis=1)
    data = pd.concat([data, data_cat], axis=1)

    data[['Attrition_Flag', 'Gender']] = data_binary

    scaler = MinMaxScaler()

    for column in num_cols:
        normalized = scipy.stats.yeojohnson(data[column])[0]
        data[column] = scaler.fit_transform(normalized.reshape(-1, 1))

    f, ax = plt.subplots(figsize=(30, 25))
    mat = data.corr('pearson')
    df = mat['Attrition_Flag']
    df.to_csv('Attrition.csv')
    mat = round(mat, 3)
    mask = np.triu(np.ones_like(mat, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig('histogram.png', dpi=300)
    plt.show()


def data_selection():
    data = pd.read_csv('BankChurners.csv')

    data.columns = data.columns.str.replace(' ', '')
    categories = {'Attrition_Flag': {'Attrited Customer': 1, 'Existing Customer': 0}}

    data = data.replace(categories)
    data_binary = data['Attrition_Flag']
    data = data.drop(columns=['Attrition_Flag'])

    data = data[['Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Total_Revolving_Bal', 'Total_Trans_Amt',
                 'Avg_Utilization_Ratio', 'Months_Inactive_12_mon', 'Total_Relationship_Count']]

    cat_cols = []
    num_cols = []

    for column in data.columns:
        if data[column].nunique() < 8:
            cat_cols.append(column)
        else:
            num_cols.append(column)

    cols_final = data.columns

    for column in cols_final:
        if column in cat_cols:
            data[column] = data[column].fillna(value=data[column].value_counts().index[0])
        else:
            data[column] = data[column].fillna(value=data[column].mean())

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data[cat_cols])
    data_cat = enc.transform(data[cat_cols])
    data_cat.columns = enc.get_feature_names(cat_cols)

    data_cat = pd.DataFrame.sparse.from_spmatrix(data_cat)
    data_cat.columns = enc.get_feature_names(cat_cols)
    data = data.drop(cat_cols, axis=1)
    data = pd.concat([data, data_cat], axis=1)

    data['Attrition_Flag'] = data_binary

    scaler = MinMaxScaler()

    for column in num_cols:
        normalized = scipy.stats.yeojohnson(data[column])[0]
        data[column] = scaler.fit_transform(normalized.reshape(-1, 1))

    data.to_csv('data.csv')
    plt.show()


if __name__ == '__main__':
    data_analyse()
