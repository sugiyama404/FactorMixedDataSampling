from sklearn.linear_model import LassoLarsIC, LinearRegression
import numpy as np
import pandas as pd
# sklearnの標準化モジュールをインポート
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA, PCA
import matplotlib.pyplot as plt

gdp = pd.read_csv('import/GDP_JAPAN.csv',
                  parse_dates=['DATE'], index_col='DATE')
iip = pd.read_csv('import/IIP.csv', parse_dates=['DATE'], index_col='DATE')
ita = pd.read_csv('import/ITA.csv', parse_dates=['DATE'], index_col='DATE')

gdp.index = pd.to_datetime(gdp.index, format='%m/%d/%Y').strftime('%Y-%m-01')
iip.index = pd.to_datetime(iip.index, format='%m/%d/%Y').strftime('%Y-%m-01')
ita.index = pd.to_datetime(ita.index, format='%m/%d/%Y').strftime('%Y-%m-01')

dfX = pd.concat([iip['IIP_YOY'], ita['ITA_YOY']], axis=1)

# データを変換する計算式を生成
sc = StandardScaler()
sc.fit(dfX)

# 実際にデータを変換
z = sc.transform(dfX)

dfX_std = pd.DataFrame(z, columns=dfX.columns)

# 主成分分析
transformer = PCA(n_components=1, random_state=0)
transformer.fit(z)
X_transformed = transformer.transform(z)

'''
# スパース主成分分析
transformer = SparsePCA(n_components=1, random_state=0)
transformer.fit(z)
X_transformed = transformer.transform(z)
'''

# 前処理
dfX_factor = pd.DataFrame(X_transformed, columns=['FACTOR'])
dfX_factor.index = iip.index
dfX_std.index = iip.index
df_std_factor = pd.merge(dfX_factor, dfX_std, left_index=True,
                         right_index=True, how='outer')
dfX_std_factor = pd.merge(df_std_factor, gdp, left_index=True,
                          right_index=True, how='outer')
dfX_std_factor = dfX_std_factor[['GDP_CYOY', 'IIP_YOY', 'ITA_YOY', 'FACTOR']]
dfX_std_factor = dfX_std_factor[dfX_std_factor.index != '2013-01-01']
dfX_std_factor['PERIOD'] = pd.to_datetime(dfX_std_factor.index.to_series()).apply(
    lambda x: 3 if x.month in [1, 4, 7, 10] else (1 if x.month in [2, 5, 8, 11] else 2))

# bridge_factor
bridge_factor = pd.DataFrame(
    columns=['BRIDGE_IIP_YOY', 'BRIDGE_ITA_YOY', 'BRIDGE_FACTOR'])
x = np.array([])
y = np.array([])
z = np.array([])
flag = False
for date, IIP_YOY, ITA_YOY, FACTOR in zip(dfX_std_factor.index, dfX_std_factor['IIP_YOY'], dfX_std_factor['ITA_YOY'], dfX_std_factor['FACTOR']):

    x = np.append(x, IIP_YOY)
    y = np.append(y, ITA_YOY)
    z = np.append(z, FACTOR)
    if flag == False:
        if date == '2013-07-01':
            flag = True

    if flag:
        x3t = x[-1]
        x3tm1 = x[-2]
        x3tm2 = x[-3]
        x3tm3 = x[-4]
        x3tm4 = x[-5]
        xt = (x3t + 2*x3tm1 + 3*x3tm2 + 2*x3tm3 + x3tm4)/3

        y3t = y[-1]
        y3tm1 = y[-2]
        y3tm2 = y[-3]
        y3tm3 = y[-4]
        y3tm4 = y[-5]
        yt = (y3t + 2*y3tm1 + 3*y3tm2 + 2*y3tm3 + y3tm4)/3

        z3t = z[-1]
        z3tm1 = z[-2]
        z3tm2 = z[-3]
        z3tm3 = z[-4]
        z3tm4 = z[-5]
        zt = (z3t + 2*z3tm1 + 3*z3tm2 + 2*z3tm3 + z3tm4)/3
        record = pd.Series([xt, yt, zt],
                           index=bridge_factor.columns, name=date)
        bridge_factor = bridge_factor.append(record)

bridge_factor.index.name = 'DATE'
df_bridge = pd.merge(gdp, bridge_factor,
                     left_index=True, right_index=True, how='outer')
df_bridge = df_bridge.dropna()


# factor_MIDAS
df_factor = pd.DataFrame(columns=[
    'GDP_CYOY', 'IIP_YOY_Q1', 'IIP_YOY_Q2', 'IIP_YOY_Q3', 'ITA_YOY_Q1', 'ITA_YOY_Q2', 'ITA_YOY_Q3', 'FACTOR_Q1', 'FACTOR_Q2', 'FACTOR_Q3'])
for date, GDP_CYOY, IIP_YOY, ITA_YOY, FACTOR, PERIOD in zip(dfX_std_factor.index, dfX_std_factor.GDP_CYOY, dfX_std_factor.IIP_YOY, dfX_std_factor.ITA_YOY, dfX_std_factor.FACTOR, dfX_std_factor.PERIOD):

    if PERIOD == 1:
        q1_iip = IIP_YOY
        q1_ita = ITA_YOY
        q1_factor = FACTOR
    elif PERIOD == 2:
        q2_iip = IIP_YOY
        q2_ita = ITA_YOY
        q2_factor = FACTOR
    else:
        record = pd.Series([GDP_CYOY, q1_iip, q2_iip, IIP_YOY, q1_ita, q2_ita, ITA_YOY, q1_factor, q2_factor, FACTOR],
                           index=df_factor.columns, name=date)
        df_factor = df_factor.append(record)

df_factor.index.name = 'DATE'


# 目的変数のみ削除して変数Xに格納
X_bridge = df_bridge.drop("GDP_CYOY", axis=1)
# 目的変数のみ抽出して変数Yに格納
Y_bridge = df_bridge["GDP_CYOY"]

# 目的変数のみ削除して変数Xに格納
X_factor = df_factor.drop("GDP_CYOY", axis=1)
# 目的変数のみ抽出して変数Yに格納
Y_factor = df_factor["GDP_CYOY"]

model_bridge = LinearRegression()
model_factor = LinearRegression()

model_bridge.fit(X_bridge, Y_bridge)
model_factor.fit(X_factor, Y_factor)

# パラメータ算出
# 回帰係数
reg_bridge_a_0 = model_bridge.coef_[0]
reg_bridge_a_1 = model_bridge.coef_[1]
reg_bridge_a_2 = model_bridge.coef_[2]
reg_factor_a_0 = model_factor.coef_[0]
reg_factor_a_1 = model_factor.coef_[1]
reg_factor_a_2 = model_factor.coef_[2]
reg_factor_a_3 = model_factor.coef_[3]
reg_factor_a_4 = model_factor.coef_[4]
reg_factor_a_5 = model_factor.coef_[5]
reg_factor_a_6 = model_factor.coef_[6]
reg_factor_a_7 = model_factor.coef_[7]
reg_factor_a_8 = model_factor.coef_[8]

# 切片
reg_bridge_b = model_bridge.intercept_
reg_factor_b = model_factor.intercept_

df_bridge['NOWCAST'] = df_bridge.apply(
    lambda x: reg_bridge_b + reg_bridge_a_0*x['BRIDGE_IIP_YOY'] + reg_bridge_a_1*x['BRIDGE_ITA_YOY'] + reg_bridge_a_2*x['BRIDGE_FACTOR'], axis=1)
df_factor['NOWCAST'] = df_factor.apply(lambda x: reg_factor_b + reg_factor_a_0*x['IIP_YOY_Q1'] + reg_factor_a_1*x['IIP_YOY_Q2'] + reg_factor_a_2*x['IIP_YOY_Q3'] + reg_factor_a_3 *
                                       x['ITA_YOY_Q1'] + reg_factor_a_4*x['ITA_YOY_Q2'] + reg_factor_a_5*x['ITA_YOY_Q3'] + reg_factor_a_6*x['FACTOR_Q1'] + reg_factor_a_7*x['FACTOR_Q2'] + reg_factor_a_8*x['FACTOR_Q3'], axis=1)


df_bridge_new = df_bridge.copy()
df_bridge_new = df_bridge_new.drop('BRIDGE_IIP_YOY', axis=1)
df_bridge_new = df_bridge_new.drop('BRIDGE_ITA_YOY', axis=1)
df_bridge_new = df_bridge_new.drop('BRIDGE_FACTOR', axis=1)

df_factor_new = df_factor.copy()
df_factor_new = df_factor_new.drop('IIP_YOY_Q1', axis=1)
df_factor_new = df_factor_new.drop('IIP_YOY_Q2', axis=1)
df_factor_new = df_factor_new.drop('IIP_YOY_Q3', axis=1)
df_factor_new = df_factor_new.drop('ITA_YOY_Q1', axis=1)
df_factor_new = df_factor_new.drop('ITA_YOY_Q2', axis=1)
df_factor_new = df_factor_new.drop('ITA_YOY_Q3', axis=1)
df_factor_new = df_factor_new.drop('FACTOR_Q1', axis=1)
df_factor_new = df_factor_new.drop('FACTOR_Q2', axis=1)
df_factor_new = df_factor_new.drop('FACTOR_Q3', axis=1)


if __name__ == '__main__':
    # print("ok")
    # print(dfX_std[['IIP_YOY']].values)
    # print(dfX_std['ITA_YOY'].values)

    '''
    print(alpha_bic_)
    print(model_bic.coef_)
    '''
    #pd.set_option('display.max_columns', None)
    # print(df_bridge.head(10))
    # print(df_factor.head(10))
    # print(df_bridge.columns)
    # print(df_factor.columns)

    '''
    # 回帰係数
    print('回帰係数')
    print(model_bridge.coef_)
    print(model_factor.coef_)
    # 切片
    print('切片')
    print(model_bridge.intercept_)
    print(model_factor.intercept_)
    # 決定係数
    print('決定係数')
    print(model_bridge.score(X_bridge, Y_bridge))
    print(model_factor.score(X_factor, Y_factor))
    '''

    plt.figure()
    df_bridge_new.plot()
    plt.savefig('export/nowcast_bridge_PCA.png')
    plt.close('all')

    plt.figure()
    df_factor_new.plot()
    plt.savefig('export/nowcast_factor_PCA.png')
    plt.close('all')
