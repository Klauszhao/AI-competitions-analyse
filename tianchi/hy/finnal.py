#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from scipy.stats import kurtosis,mode
import scipy.signal as signal
from sklearn.metrics import f1_score
import time
import warnings
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')


# In[2]:



def pinghua(sig, method):
    if method == 0:
        return sig
    elif method == 1:
        # 中值滤波，kernel_size 表示窗口值，窗口内的值排序，取中间的值，必须是奇数
        return signal.medfilt(volume=sig, kernel_size=13)
    elif method == 2:
        return signal.savgol_filter(sig, 5, 3, 0)
    elif method == 3:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        return np.convolve(sig, kernel, mode='same')   

def pinghua_cols(group):
    cols = ['x','y','v','d','x+y']
    for col in cols:
        sig = group[col]
        group[col] = pinghua(sig,1)
    return group

def rate(df,feat):
    df1 = df.groupby(['shipid']).agg({feat:'diff','time_second':'diff'})
    df1[feat+'rate'] = abs(3.6*df1[feat]/df1['time_second'])
    df1['shipid'] = df['shipid']
    df1.columns = [feat,'time_delta',feat+'_rate','shipid']
    return(df1[['shipid',feat+'_rate']])
    
def basic_digital_features(group):
    data = group
    fea = []
    fea.append(data.mode().values[0])
    fea.append(data.max())
    fea.append(data.min())
    fea.append(data.mean())
    fea.append(data.ptp()) #最大值与最小值的差
    fea.append(data.std())
    fea.append(data.median())
    fea.append(data.kurt())
    fea.append(data.skew())
    fea.append(np.mean(np.abs(data - data.mean()))) 
    fea.append(fea[1] / fea[0])
    fea.append(fea[2] / fea[0])
    fea.append(fea[3] / fea[0])
    fea.append(fea[6] / fea[0])
    return fea

# v,d专用的
def basic2_digital_features(group):
    data = group
    fea = []
    fea.append(data.mode().values[0])
    fea.append(data.max())
    fea.append(data.min())
    fea.append(data.mean())
    fea.append(data.ptp())
    fea.append(data.std())
    fea.append(data.median())
    fea.append(data.kurt())
    fea.append(data.skew())
    fea.append(np.mean(np.abs(data - data.mean()))) 
    return fea

## 区间数
def feaquan(group):
    data = group
    fea = []
    fea.append(data.quantile(.01))
    fea.append(data.quantile(.05))
    fea.append(data.quantile(.25))
    fea.append(data.quantile(.75))
    fea.append(data.quantile(.95))
    fea.append(data.quantile(.99))
    fea.append(data.quantile(.75)-data.quantile(.25))
    return fea

# 组合变量的统计数据
def basic3_digital_features(group):
    data = group
    fea = []
    fea.append(data.mode().values[0])
    fea.append(data.max())
    fea.append(data.min())
    fea.append(data.mean())
    fea.append(data.ptp())
    fea.append(data.std())
    fea.append(data.median())
    return fea


# In[3]:


def filter_by_dis(df,ind_diff_num,ind_dist_diff_num):

    df_new = df[['shipid','x', 'y', 'v','time_second']].copy()
    df_new.sort_values(['shipid', 'time_second'], ascending=[False, True],inplace = True)
 
    df_diff = df_new[['x', 'y','time_second']].diff().abs()
    df_diff.columns = ['x_diff', 'y_diff','t_diff']
    df_new = pd.merge(df_new,df_diff, left_index=True, right_index=True,how='left')

    group_ship = df_new.groupby(['shipid'])
    df_ex = group_ship['t_diff'].head(1)
    ex_list = list(df_ex.index)
    df_new = df_new.drop(index = ex_list,axis = 1)
    
    df_new = df_new.dropna()  # we
    
    df_new['ind_dist'] = np.sqrt(df_new['x_diff']**2  + df_new['y_diff']**2)
    df_new['ind_dist'] = df_new['ind_dist'].round(0)    
    
    df_new['ind_dist_v_t'] = df_new['t_diff']*df_new['v']

    # 坐标是有误差范围的，
    df_new_error = df_new[(df_new['ind_dist_v_t'] == 0) & (df_new['ind_dist'] > ind_diff_num)]
    
    df_new = df_new[df_new['ind_dist_v_t'] > 0]
    
    df_new['ind_dist_diff'] = df_new['ind_dist']/df_new['ind_dist_v_t']
    df_new['ind_dist_diff_10'] = df_new['ind_dist_diff']*10
    df_new['ind_dist_diff_10'] = df_new['ind_dist_diff_10'].round(0)
    
    df_new['ind_dist_diff_100'] = df_new['ind_dist_diff']*100
    df_new['ind_dist_diff_100'] = df_new['ind_dist_diff_100'].round(0)
    
    df_new['ind_dist_diff_1000'] = df_new['ind_dist_diff']*1000
    df_new['ind_dist_diff_1000'] = df_new['ind_dist_diff_1000'].round(0)
    
    df_new = df_new[df_new['ind_dist_diff_10'] < ind_dist_diff_num]
    
    print("filter_by_dis over, shape = ",df_new.shape)
    return df_new


# In[4]:


train_path = 'tcdata/hy_round2_train_20200225'
test_path = 'tcdata/hy_round2_testB_20200312'

#train_path = '/Users/zn/Public/work/data/海洋大赛/hy_round1_train_20200102'
#test_path = '/Users/zn/Public/work/data/海洋大赛/hy_round1_testB_20200221'

def getData(filepath,files,str_type):
    ret = []
    for file in files:
        df = pd.read_csv(f'{filepath}/{file}')
        ret.append(df)
    
    df_new = pd.concat(ret)
    if str_type == 'train':
        df_new.columns = ['shipid','x','y','v','d','time','type']
        return df_new
    
    df_new.columns = ['shipid','x','y','v','d','time']
    return df_new


train_files = os.listdir(train_path)

test_files = os.listdir(test_path)

train = getData(train_path,train_files,'train')
test = getData(test_path,test_files,'test')
print("- "*6+ "train.shape="+str(train.shape)+" -"*6)
print("- "*6+ "test.shape="+str(test.shape)+" -"*6)


# In[6]:


feat_test = pd.DataFrame()
feat_test['shipid'] = test.groupby('shipid')['shipid'].head(1)
print(feat_test.shape)


# In[11]:


#data = data[data['shipid'] == 1]

data = pd.concat([train,test],ignore_index=True)

data['x'] = data['x'].replace(0,24.325)
data['y'] = data['y'].replace(0,117.117)
tmp = data[(data['x']>23) & (data['y'] < 115)]
data = data.drop(tmp.index,axis = 0)
data = data[data['v']<13]
print(data.shape)

data['time'] = data['time'].apply(lambda x: '2019' + str(x))
data['time'] = pd.to_datetime(data['time'], format='%Y%m%d %H:%M:%S')

data['time_second'] = data['time'].apply(lambda x:time.mktime(x.timetuple())).astype(int)

ind_diff_num = 40
ind_dist_diff_num = 40
df_new = filter_by_dis(data,ind_diff_num,ind_dist_diff_num)

data =  pd.merge(df_new,data[['d', 'time', 'type']], left_index=True, right_index=True,how='left')


data.sort_values(['shipid','time_second'], ascending=[True,True],inplace = True)
data = data.reset_index(drop=True)

data['x+y'] = np.sqrt(data['x']**2+data['y']**2)

group_shipid = data.groupby('shipid')

# 平滑数据，中值滤波
data = group_shipid.apply(pinghua_cols)

data = data[(data['x']> 0) & (data['y']>0)]
data['x+y_rate'] = rate(data,'x+y')['x+y_rate']

## 这里只是 提取 唯一值
feat = pd.DataFrame()
feat['shipid'] = group_shipid['shipid'].head(1)
feat['label'] = group_shipid['type'].head(1)
feat = feat.reset_index(drop=True)
feat['index'] = feat['shipid']
feat.set_index(["index"], inplace=True)

## 基本统计特征
feax = group_shipid['x'].apply(basic_digital_features)
feay = group_shipid['y'].apply(basic_digital_features)
feaxy = group_shipid['x+y'].apply(basic_digital_features)

st = ['_'+ n for n in ['mode','max','min','mean','ptp',
                       'median','kurt','skew','mad',
                       'max_mode','min_mode','mean_mode','median_mode']]

for i in range(len(st)):
    feat['x'+st[i]] = feax.map(lambda x:x[i])
    feat['y'+st[i]] = feay.map(lambda x:x[i])
    feat['x+y'+st[i]] = feaxy.map(lambda x:x[i])

feat['y_max_x_min'] = feat['y_max'] - feat['x_min']
feat['x_max_y_min'] = feat['x_max'] - feat['y_min']

feat['x_max_x_min'] = feat['x_max'] - feat['x_min']
feat['y_max_y_min'] = feat['y_max'] - feat['y_min']  


# df_new = group_shipid[['x','y']].agg({'x_y_cov':lambda x: x['x'].cov(x['y'])}).reset_index()
# df_new = pd.merge(feat['shipid'],df_new,on="shipid",how='left')
# df_new.columns = ['shipid',  'x_y_cov', 'x_y_cov_two']
# df_new = df_new.drop_duplicates('shipid')

#feat = pd.merge(feat, df_new[['shipid','x_y_cov']], on='shipid', how='left')
print(feat.columns)


# v、d 的统计特征
feav = group_shipid['v'].apply(basic2_digital_features)
fead = group_shipid['d'].apply(basic2_digital_features)

st = ['_'+ n for n in ['mode','max','mean','ptp','median',
                       'kurt','skew','mad']]
for i in range(len(st)):
    feat['v'+st[i]] = feav.map(lambda x:x[i])
    feat['d'+st[i]] = fead.map(lambda x:x[i])

print("- "*6+ "feat.shape="+str(feat.shape)+" -"*6)

st = ['_'+ n for n in ['01','05','25','75','95','99','75-25']]
for i in ['x','y','v','d','x+y']:
    feaq = group_shipid[i].apply(feaquan)
    for j in range(len(st)):
        feat[i+st[j]] = feaq.map(lambda x:x[j])

#统计速度类型，按区间划分，求解区间统计变量，需要改进
print(data.columns)
print(feat.columns)
feat['cx']=feat['shipid'].map(data['shipid'].value_counts())

fx=[0,1,3,5,8,12]
data['v_n']=pd.cut(data['v'].values,fx,right=False).codes
df_n=data.groupby(['shipid','v_n']).size().reset_index()

for i in range(5):
    f1=df_n[df_n['v_n']== i]
    f1.index=f1['shipid']
    feat['v_'+str(i)]=feat['shipid'].map(f1[0])

for i in range(5):
    feat['v_rate_'+str(i)] = feat['v_'+str(i)]/feat['cx']

print('feat.columns = ',feat.columns)

#  改进一下
data[['x_diff','y_diff','v_diff','d_diff']] = group_shipid.agg({'x':'diff',
                                                                      'y':'diff',
                                                                      'v':'diff',
                                                                      'd':'diff'})

##  这里不太清楚 为啥要 x/y 比较，加偏移量是防止 除数为 0 ，0.00001
data['x/y'] = data['x']/(data['y']+1)
data['v/x'] = data['v']/(data['x']+1)
data['v/y'] = data['v']/(data['y']+1)


print("- "*6+ "feat.shape="+str(feat.shape)+" -"*6)


cols = ['x_diff','y_diff','v_diff','d_diff','x/y','v/x','v/y']
st = ['_'+ n for n in ['mode','max','min','mean','ptp','std','median']]
for col in cols:
    feadi = data.groupby('shipid')[col].apply(basic3_digital_features)
    for i in range(len(st)):
        feat[col+st[i]] = feadi.map(lambda x:x[i])

# 交叉乘除         
fi = ['x/y_min', 'x/y_mode', 'x/y_max', 'x_mode','y_mode']
for i in range(len(fi)):
    for j in range(i+1,len(fi)):
        feat[fi[i]+'-'+fi[j]] = feat[fi[i]]-feat[fi[j]]
        feat[fi[i]+'+'+fi[j]] = feat[fi[i]]+feat[fi[j]]
        feat[fi[i]+'*'+fi[j]] = feat[fi[i]]*feat[fi[j]]

## 


print("- "*6+'提取特征结束'+" -"*6)


# In[ ]:



# #data_train = train

# df_train = get_feature(train)

# #data_test = test

# df_test = get_feature(test)


# In[17]:



filter_fea = ['label','shipid']


filter_fea_two = ['x_median','y_median','x+y_median','x_median_mode','y_median_mode','x+y_median_mode'
                  ,'d_diff_median','x_y_cov','x_diff_median','y_diff_median','x_diff_median','v/y_median']

#filter_fea.extend(filter_fea_two)

features = [f for f in feat.columns if f not in filter_fea ]

print(len(features), ','.join(features))


f = feat
df_train = f[~f['label'].isnull()]

df_test = f[f['label'].isnull()]
X_test = df_test[features]

# X = dfr[[col for col in dfr.columns if col not in ['label','shipid']]]
# y = dfr['label']
# y = y.map({'拖网':0,'围网':1,'刺网':2})

X = df_train[features].copy()

y_train = df_train['label']
y_train = y_train.map({'拖网':0,'围网':1,'刺网':2})

y = y_train

#X_test = df_test[features]

print(X.head(3))


# In[18]:


def f1_score_vail(pred, data_vail):
    labels = data_vail.get_label()
    pred = np.argmax(pred.reshape(3, -1), axis=0)      
    score_vail = f1_score(y_true=labels, y_pred=pred, average='macro')
    return 'f1_score', score_vail, True


skf = StratifiedKFold(n_splits=10, random_state=64, shuffle=True)

oof_lgb = np.zeros((len(X),3))
prediction = np.zeros((len(X_test),3))
clfs=[]
param = {'boosting_type': 'gbdt', 
         'objective': 'multiclassova', 
         'num_class':3,
         'learning_rate': 0.1, 
         'max_depth':-1,   
         'subsample': 0.5, 
         'colsample_bytree': 0.4,
         'is_unbalance': 'true',
         'metric':'None'
         }
for index, (trn_idx, val_idx) in enumerate(skf.split(X,y)):
    print('{}折交叉验证开始'.format(index+1))
    trn_data = lgb.Dataset(X.iloc[trn_idx], y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], feval=f1_score_vail,verbose_eval = 100, 
                    early_stopping_rounds = 400)
    clfs.append(clf)
    oof_lgb[val_idx] = clf.predict(X.iloc[val_idx], num_iteration = clf.best_iteration)
    prediction += clf.predict(X_test, num_iteration=clf.best_iteration)
oof_lgb_final = np.argmax(oof_lgb, axis=1)
print('finanl score = ',f1_score(y.values, oof_lgb_final, average='macro'))


# In[103]:


ret = []
for index, model in enumerate(clfs):
    df = pd.DataFrame()
    df['name'] = model.feature_name()
    df['score'] = model.feature_importance()
    df['fold'] = index
    ret.append(df)
df = pd.concat(ret)

df = df.groupby('name', as_index=False)['score'].mean()
df = df.sort_values(['score'], ascending=False)
df_50 = df.head(180)
print(df_50)


# In[10]:



pred = np.argmax(prediction, axis=1)
sub = pd.DataFrame()
sub['shipid'] = df_test.index
sub['label'] = pred



pred_index = df_test.index.values

test_index = test.groupby('shipid')['shipid'].head(1).values

index_new = [x for x in test_index if  x not in pred_index ]
print('index_new.len=',len(index_new))
feat_test = pd.DataFrame()
feat_test['shipid'] = index_new
feat_test["label"] = [1 for x in range(len(index_new))]

sub = pd.concat([sub,feat_test],ignore_index=True)
#df_res = pd.concat([df_res,df_respon],ignore_index=True)

print(sub.shape)

sub['label'] = sub['label'].map({0:'拖网',1:'围网',2:'刺网'})

sub.to_csv('result.csv',index=None, header=False)


# In[ ]:




