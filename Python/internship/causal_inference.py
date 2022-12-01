#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install causalml shap category_encoders')


# In[2]:


# 导入包
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

from causalml.inference.meta import (
    BaseXRegressor,
    LRSRegressor,
    MLPTRegressor,
    XGBTRegressor,
)

from xgboost import XGBClassifier, XGBRegressor
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.max_info_columns', 500)


# In[3]:


df = pd.read_csv('user_transform_content_suply_causalml_data_0910_300w.csv', sep = ',')


# In[4]:


df.info()


# In[22]:


df.head()


# In[5]:


# 数据预处理
#数据类型
for i in df.columns[2:29]:
     df[i] = df[i].replace('\\N',0).astype(float)
df['profile_gender_v1'] = np.where((df['profile_gender'] != '男') & (df['profile_gender'] != '女'), '未知', df['profile_gender'])
df['profile_gender'] = df['profile_gender_v1']
df['profile_city_level']= df['profile_city_level'].fillna('\\N')
df['profile_city_level'] = np.where(df['profile_city_level'] == '\\N', '其他', df['profile_city_level'])
# # 中活用户活跃度提高
# df['week_active_type'] = np.where(df['active_day_cnt_s6d0'] >= 5 , '高活', '非高活')
# df['sign'] = np.where(df['week_active_type'] == '高活', 1,0)


# In[6]:


# 数据清洗
#过滤曝光为0的用户
df = df[df['article_imp_pv'] != 0]


# In[6]:


df = df[df['article_imp_pv'] >= 10]


# In[7]:


df.describe(percentiles = np.arange(0,1.0,0.05))


# In[8]:


df['sign'] = df['is_retain']


# In[30]:


df.to_csv('处理后-transform_content_suply_causalml_data_300_v3.csv', sep = ',')


# In[9]:


# category处理 one hot encoding
import category_encoders
from category_encoders import OneHotEncoder
ohe = OneHotEncoder()
df_gender = ohe.fit_transform(df['profile_gender'])
df_age = ohe.fit_transform(df['profile_age_level'])
df_city_level = ohe.fit_transform(df['profile_city_level'])
df_brand = ohe.fit_transform(df['brand'])
df = pd.concat([df, df_brand], axis = 1)
df = pd.concat([df, df_gender], axis = 1)
df = pd.concat([df, df_city_level], axis = 1)
df = pd.concat([df, df_age], axis = 1)


# # 相关性分析

# In[10]:


factors = ['sign','imgtext_cnt_pro', 'video_cnt_pro',
       'hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'factive_imgtext_cnt_pro',
       'factive_video_cnt_pro', 'cp_level4_cnt_pro', 
       'cp_level2_cnt_pro','profile_age_level', 'profile_city_level',
       'profile_gender', 'brand', 'icon_vst_cnt', 'weixin_vst_cnt',
       'push_vst_cnt',]
df_cor = df[factors]
Y = 'sign'
corr = pd.DataFrame(df_cor.corr()).sort_values(by = Y, ascending = False)[[Y]]
print(corr)


# In[17]:


# 画图
import matplotlib.pyplot as plt 
import warnings
import seaborn as sns
plt.figure(figsize=(8, 8))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(df_cor.corr(), vmin=-1, vmax=1, annot=True,cmap='BrBG')
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':10}, pad=12);


# In[28]:


df[baseconfounders].info()


# In[24]:


baseconfounders = ['sign','imgtext_cnt_pro', 'video_cnt_pro',
       'hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'factive_imgtext_cnt_pro',
       'factive_video_cnt_pro', 'cp_level4_cnt_pro', 'cp_level3_cnt_pro',
       'cp_level2_cnt_pro']
y = 'sign'

df_x_shap = df[baseconfounders].drop(y, axis = 1)
df_y_shap = df[y]
X_train, X_test, y_train, y_test = train_test_split(df_x_shap, df_y_shap, test_size=0.1, random_state=0)
model = XGBClassifier(
    max_depth=6,
    learning_rate=0.05,
    booster="gbtree",
    gamma=0.47,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.95,
    eval_metric="auc",
    eta=0.025,
    seed=0,
    nthread=8,
    scale_pos_weight=3,
    random_state=42,
    n_estimators=200
)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
print("trainACC: %.4f" % metrics.accuracy_score(y_train, y_train_pred))
y_pred = model.predict(X_test)
print("testACC: %.4f" % metrics.accuracy_score(y_test, y_pred))


# In[26]:


# Feature importance
confounders = df_x_shap.columns
feature = model.feature_importances_
features_importance = pd.concat([pd.DataFrame(confounders, columns = ['confounders']), pd.DataFrame(feature,  columns = ['score'])], axis = 1).sort_values(by='score', axis = 0, ascending=False, inplace = False)

# feature importance 可视化
n = 10
plt.figure(figsize=(15, 5))
plt.bar(range(n), features_importance['score'][:n])
plt.xticks(range(n), features_importance['confounders'][:n], rotation=-45, fontsize=14)
plt.title('Feature importance', fontsize=14)
plt.show()


# In[27]:


features_importance


# In[36]:


df[df['week_active_type'] == '高活'].describe(np.quantile)


# In[38]:


df.describe(percentiles = np.arange(0.1,1.0,0.1))


# In[89]:


df['imgtext_cnt_pro'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]).reset_index(drop = True)


# In[11]:


# 中位数分桶
treatments = ['imgtext_cnt_pro', 'video_cnt_pro',
       'hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'factive_imgtext_cnt_pro',
       'factive_video_cnt_pro', 'cp_level4_cnt_pro', 'cp_level3_cnt_pro',
       'cp_level2_cnt_pro']

# interval = {
#     'imgtext_cnt_pro': [0.0,df['imgtext_cnt_pro'].median(), df['imgtext_cnt_pro'].max()],
#     'video_cnt_pro': [0.0,df['video_cnt_pro'].median(), df['video_cnt_pro'].max()],
#     'hot_imgtext_cnt_pro': [0.0,df['hot_imgtext_cnt_pro'].median(), df['hot_imgtext_cnt_pro'].max()],
#     'hot_video_cnt_pro': [0.0,df['hot_video_cnt_pro'].median(), df['hot_video_cnt_pro'].max()],
#     'factive_imgtext_cnt_pro': [0.0,df['factive_imgtext_cnt_pro'].median(), df['factive_imgtext_cnt_pro'].max()],
#     'factive_video_cnt_pro': [0.0,df['factive_video_cnt_pro'].median(), df['factive_video_cnt_pro'].max()],
#     'cp_level4_cnt_pro': [0.0,df['cp_level4_cnt_pro'].median(), df['cp_level4_cnt_pro'].max()],
#     'cp_level3_cnt_pro': [0.0,df['cp_level3_cnt_pro'].median(), df['cp_level3_cnt_pro'].max()],
#     'cp_level2_cnt_pro': [0.0,df['cp_level2_cnt_pro'].median(), df['cp_level2_cnt_pro'].max()],
   
# }  
# interval = {
#     'imgtext_cnt_pro': [0.0,df['imgtext_cnt_pro'].mean(), df['imgtext_cnt_pro'].max()],
#     'video_cnt_pro': [0.0,df['video_cnt_pro'].mean(), df['video_cnt_pro'].max()],
#     'hot_imgtext_cnt_pro': [0.0,df['hot_imgtext_cnt_pro'].mean(), df['hot_imgtext_cnt_pro'].max()],
#     'hot_video_cnt_pro': [0.0,df['hot_video_cnt_pro'].mean(), df['hot_video_cnt_pro'].max()],
#     'factive_imgtext_cnt_pro': [0.0,df['factive_imgtext_cnt_pro'].mean(), df['factive_imgtext_cnt_pro'].max()],
#     'factive_video_cnt_pro': [0.0,df['factive_video_cnt_pro'].mean(), df['factive_video_cnt_pro'].max()],
#     'cp_level4_cnt_pro': [0.0,df['cp_level4_cnt_pro'].mean(), df['cp_level4_cnt_pro'].max()],
#     'cp_level3_cnt_pro': [0.0,df['cp_level3_cnt_pro'].mean(), df['cp_level3_cnt_pro'].max()],
#     'cp_level2_cnt_pro': [0.0,df['cp_level2_cnt_pro'].mean(), df['cp_level2_cnt_pro'].max()],
   
# }
# interval = {
#     'imgtext_cnt_pro': [0.0,df['imgtext_cnt_pro'].quantile(0.25), df['imgtext_cnt_pro'].quantile(0.5), df['imgtext_cnt_pro'].max()],
#     'video_cnt_pro': [0.0,df['video_cnt_pro'].quantile(0.25), df['video_cnt_pro'].quantile(0.5), df['video_cnt_pro'].max()],
#     'hot_imgtext_cnt_pro': [0.0,df['hot_imgtext_cnt_pro'].quantile(0.25),  df['hot_imgtext_cnt_pro'].quantile(0.5),df['hot_imgtext_cnt_pro'].max()],
#     'hot_video_cnt_pro': [0.0,df['hot_video_cnt_pro'].quantile(0.25),  df['hot_video_cnt_pro'].quantile(0.5),df['hot_video_cnt_pro'].max()],
#     'factive_imgtext_cnt_pro': [0.0,df['factive_imgtext_cnt_pro'].quantile(0.25),  df['factive_imgtext_cnt_pro'].quantile(0.5),df['factive_imgtext_cnt_pro'].max()],
#     'factive_video_cnt_pro': [0.0,df['factive_video_cnt_pro'].quantile(0.25), df['factive_video_cnt_pro'].quantile(0.5),df['factive_video_cnt_pro'].max()],
#     'cp_level4_cnt_pro': [0.0,df['cp_level4_cnt_pro'].quantile(0.25),  df['cp_level4_cnt_pro'].quantile(0.5),df['cp_level4_cnt_pro'].max()],
#     'cp_level3_cnt_pro': [0.0,df['cp_level3_cnt_pro'].quantile(0.25),  df['cp_level3_cnt_pro'].quantile(0.5),df['cp_level3_cnt_pro'].max()],
#     'cp_level2_cnt_pro': [0.0,df['cp_level2_cnt_pro'].quantile(0.25),  df['cp_level2_cnt_pro'].quantile(0.5),df['cp_level2_cnt_pro'].max()],
   
# }

# interval = {
#     'imgtext_cnt_pro': df['imgtext_cnt_pro'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]).reset_index(drop = True),
#     'video_cnt_pro': df['video_cnt_pro'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]).reset_index(drop = True),
#     'hot_imgtext_cnt_pro': df['hot_imgtext_cnt_pro'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]).reset_index(drop = True),
#     'hot_video_cnt_pro': df['hot_video_cnt_pro'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]).reset_index(drop = True),
#     'factive_imgtext_cnt_pro': df['factive_imgtext_cnt_pro'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]).reset_index(drop = True),
#     'factive_video_cnt_pro': df['factive_video_cnt_pro'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]).reset_index(drop = True),
#     'cp_level4_cnt_pro': df['cp_level4_cnt_pro'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]).reset_index(drop = True),
#     'cp_level3_cnt_pro': df['cp_level3_cnt_pro'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]).reset_index(drop = True),
#     'cp_level2_cnt_pro': df['cp_level2_cnt_pro'].quantile([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]).reset_index(drop = True),
   
# } 

interval = {
    'imgtext_cnt_pro': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1],
    'video_cnt_pro': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1],
    'hot_imgtext_cnt_pro': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1],
    'hot_video_cnt_pro': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1],
    'factive_imgtext_cnt_pro': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1],
    'factive_video_cnt_pro': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1],
    'cp_level4_cnt_pro': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1],
    'cp_level3_cnt_pro':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1],
    'cp_level2_cnt_pro':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1],
   
} 


for i in treatments:
    bins_num = len(interval[i]) - 1
    df[i + '_bucket'] = -1
    df[i + '_bucket'] = np.where(df[i] == interval[i][0],
                                 0,df[i + '_bucket'])
    for k in range(len(interval[i]) - 1):
        df[i + '_bucket'] = np.where((df[i] <= interval[i][k+1]) & (df[i] > interval[i][k]), 
                                     k,df[i + '_bucket'])


# In[15]:


# sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

from causalml.inference.meta import (
    BaseXRegressor,
    LRSRegressor,
    MLPTRegressor,
    XGBTRegressor,
    BaseXClassifier,
    BaseTClassifier,
)


from xgboost import XGBClassifier, XGBRegressor
# from catboost import CatBoostClassifier


# In[13]:


def psm_rfc(df_x, df_y):
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0)
    # fisrt model
    rfc_model = RandomForestClassifier(random_state = 3, 
                                       max_depth = 16, 
                                       n_estimators = 80,
                                       n_jobs = -1)
    rfc_model.fit(X_train, y_train)
    rfc_prediction = rfc_model.predict(X_test)
    con_matrix = metrics.confusion_matrix(y_test, rfc_prediction)
    print('confusion_matrix: ', con_matrix)
    accuracy = (con_matrix[0,0] + con_matrix[1,1])/np.sum(con_matrix)
    print('accuracy: ',accuracy)
    ps_score =  rfc_model.predict_proba(df_x)[:,1]
    return ps_score


def te_function_cla(df_y, df_x, df_z, df_p):
    # STEP 2 计算TE
    xl1 = BaseXClassifier(outcome_learner=XGBClassifier(random_state=42),
                                   effect_learner=XGBRegressor(random_state=42))
    te = xl1.fit_predict(df_x, df_y, df_z, p=df_p)
    return te

def te_function_cla_T(df_y, df_x, df_z, df_p):
    # STEP 2 计算TE
    xl1 = BaseTClassifier(
        learner=XGBClassifier(
            max_depth=11,
            learning_rate=0.05,
            booster="gbtree",
            random_state=42,
        ),
    )
    # num_boost_round=2000,
    # early_stopping_rounds=50)
    te = xl1.fit_predict(df_x, df_y, df_z, p=df_p)
    # df_x:Counfounders(x), df_y:Treatment, df['active_days_30']:Y, p:Propensity_Score
    return te


# In[14]:


def psm_rfc(df_x, df_y):
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0)
    # fisrt model
    rfc_model = RandomForestClassifier(random_state = 3, 
                                       max_depth = 16, 
                                       # min_samples_leaf = 10, 
                                       # min_samples_split = 50, 
                                       n_estimators = 100,
                                       # oob_score = True,
                                       n_jobs = -1)
    rfc_model.fit(X_train, y_train)
    rfc_prediction = rfc_model.predict(X_test)
    con_matrix = metrics.confusion_matrix(y_test, rfc_prediction)
    print('confusion_matrix: ', con_matrix)
    accuracy = (con_matrix[0,0] + con_matrix[1,1])/np.sum(con_matrix)
    print('accuracy: ',accuracy)
    ps_score =  rfc_model.predict_proba(df_x)[:,1]
    return ps_score

# KNN函数
def knn_match(df, bins_num): 
    treatment = df[df['fake_sign'] == 1]
    treatment_index = df[df['fake_sign'] == 1].index
    # print(treatment_index)
    control = df[df['fake_sign'] == 0]
    nbrs = NearestNeighbors(n_neighbors = 5, n_jobs = -1).fit(control[['propensity_score']]) # 修改
    indices = nbrs.kneighbors(treatment[['propensity_score']], return_distance=False)
    control_index = control.index[np.unique(indices.flatten())]
    return treatment_index, control_index

# 标准均值差SMD
# 在评估组间的均衡性时，SMD<0.1通常表示均衡性较好，可以认为研究组之间的差异很小。
def smd(t, c):
    smd_result = abs(t.mean() - c.mean()) / np.sqrt(.5 * (t.var() + c.var()))
    return(smd_result)


# In[43]:


df.columns


# In[78]:


for i in ['profile_gender', 'profile_age_level', 'profile_city_level', 'brand']:
    print(i)
    print(set(df[i]))


# In[77]:


df['profile_city_level']= df['profile_city_level'].fillna('\\N')
df['profile_city_level'] = np.where(df['profile_city_level'] == '\\N', '其他', df['profile_city_level'])


# In[62]:


df.columns


# In[9]:


# category处理 one hot encoding
import category_encoders
from category_encoders import OneHotEncoder
ohe = OneHotEncoder()
df_gender = ohe.fit_transform(df['profile_gender'])
df_age = ohe.fit_transform(df['profile_age_level'])
df_city_level = ohe.fit_transform(df['profile_city_level'])
df_brand = ohe.fit_transform(df['brand'])
df = pd.concat([df, df_brand], axis = 1)
df = pd.concat([df, df_gender], axis = 1)
df = pd.concat([df, df_city_level], axis = 1)
df = pd.concat([df, df_age], axis = 1)


# In[16]:





# In[18]:



treatments = ['imgtext_cnt_pro', 'video_cnt_pro',
       'hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'factive_imgtext_cnt_pro',
       'factive_video_cnt_pro', 'cp_level4_cnt_pro',
       'cp_level2_cnt_pro']

user_profile = ['profile_gender_' + str(x+1) for x in range(3)] +  ['profile_city_level_' + str(x+1) for x in range(7)] + ['profile_age_level_' + str(x+1) for x in range(6)] 
# + ['brand_' + str(x+1) for x in range(6)]

# baseconfounders = ['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
#           'active_day_7_pre','weixin_vst_cnt','push_vst_cnt'] + user_profile


# baseconfounders = ['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',] + user_profile

# baseconfounders = {'imgtext_cnt_pro': ['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
#           'active_day_cnt_s6d','weixin_vst_cnt','push_vst_cnt',]+['hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'factive_imgtext_cnt_pro',
#        'factive_video_cnt_pro', 'cp_level4_cnt_pro',
#        'cp_level2_cnt_pro'] + user_profile,
#                    'video_cnt_pro': ['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
#           'active_day_cnt_s6d','weixin_vst_cnt','push_vst_cnt',]+[ 'hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'factive_imgtext_cnt_pro',
#        'factive_video_cnt_pro', 'cp_level4_cnt_pro',
#        'cp_level2_cnt_pro'] + user_profile,
#                    'hot_imgtext_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
#           'active_day_cnt_s6d','weixin_vst_cnt','push_vst_cnt',]+['factive_imgtext_cnt_pro',
#        'factive_video_cnt_pro', 'cp_level4_cnt_pro',
#        'cp_level2_cnt_pro'] + user_profile,
#                    'hot_video_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
#           'active_day_cnt_s6d','weixin_vst_cnt','push_vst_cnt',]+['factive_imgtext_cnt_pro',
#        'factive_video_cnt_pro', 'cp_level4_cnt_pro',
#        'cp_level2_cnt_pro'] + user_profile,
#                    'factive_imgtext_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
#           'active_day_cnt_s6d','weixin_vst_cnt','push_vst_cnt',]+['hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'cp_level4_cnt_pro',
#        'cp_level2_cnt_pro'] + user_profile,
#                   'factive_video_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
#           'active_day_cnt_s6d','weixin_vst_cnt','push_vst_cnt',]+['hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'cp_level4_cnt_pro',
#        'cp_level2_cnt_pro'] + user_profile,
#                     'cp_level4_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
#           'active_day_cnt_s6d','weixin_vst_cnt','push_vst_cnt',]+['hot_imgtext_cnt_pro', 'hot_video_cnt_pro','factive_imgtext_cnt_pro',
#        'factive_video_cnt_pro', ] + user_profile,
#                     'cp_level2_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
#           'active_day_cnt_s6d','weixin_vst_cnt','push_vst_cnt',]+['hot_imgtext_cnt_pro', 'hot_video_cnt_pro','factive_imgtext_cnt_pro',
#        'factive_video_cnt_pro', ] + user_profile,
#                   }


baseconfounders = {'imgtext_cnt_pro': ['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
          'active_day_cnt_s6d',]+['hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'factive_imgtext_cnt_pro',
       'factive_video_cnt_pro', 'cp_level4_cnt_pro',
       'cp_level2_cnt_pro'] + user_profile,
                   'video_cnt_pro': ['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
          'active_day_cnt_s6d',]+[ 'hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'factive_imgtext_cnt_pro',
       'factive_video_cnt_pro', 'cp_level4_cnt_pro',
       'cp_level2_cnt_pro'] + user_profile,
                   'hot_imgtext_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
          'active_day_cnt_s6d',]+['factive_imgtext_cnt_pro',
       'factive_video_cnt_pro', 'cp_level4_cnt_pro',
       'cp_level2_cnt_pro'] + user_profile,
                   'hot_video_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
          'active_day_cnt_s6d',]+['factive_imgtext_cnt_pro',
       'factive_video_cnt_pro', 'cp_level4_cnt_pro',
       'cp_level2_cnt_pro'] + user_profile,
                   'factive_imgtext_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
          'active_day_cnt_s6d',]+['hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'cp_level4_cnt_pro',
       'cp_level2_cnt_pro'] + user_profile,
                  'factive_video_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
          'active_day_cnt_s6d',]+['hot_imgtext_cnt_pro', 'hot_video_cnt_pro', 'cp_level4_cnt_pro',
       'cp_level2_cnt_pro'] + user_profile,
                    'cp_level4_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
          'active_day_cnt_s6d',]+['hot_imgtext_cnt_pro', 'hot_video_cnt_pro','factive_imgtext_cnt_pro',
       'factive_video_cnt_pro', ] + user_profile,
                    'cp_level2_cnt_pro':['article_imp_pv', 'imgtext_imp_pv', 'video_imp_pv',
          'active_day_cnt_s6d',]+['hot_imgtext_cnt_pro', 'hot_video_cnt_pro','factive_imgtext_cnt_pro',
       'factive_video_cnt_pro', ] + user_profile,
                  }

# grid调参
param_grid = {
    'max_depth':[12, 14, 16], #修改
    'n_estimators':[80, 100, 120] #修改
}
print('treatments: ', list(treatments))
smd_failed = {}
ate_10 = {}
result = {}
xate_all = []
tate_all = []
psm_all = []
y = 'sign'
# grid调参
# param_grid = {
#     'max_depth':[12], #修改
#     'n_estimators':[80] #修改
# }
eps = np.finfo(float).eps #解决p值不在0-1之间的问题


for treatment in treatments:
    print('-----------------当前treatment：{}-----------------'.format(treatment))
    ate_10[treatment] = pd.DataFrame()
    confounders = list(set(baseconfounders[treatment]) - {treatment} - {y})
    bins_num = len(interval[treatment]) - 1
    # 选择confouners
    # df_confounder_x = df[confounders]
    # df_confounder_y = df[treatment]
    # confounders_importance = xgb_feature_importance(df_confounder_x, df_confounder_y, 'gain')
    # confounders = list(set(confounders_importance[confounders_importance['score'] >= 0.01]['confounders']) | {'age', 'gender', 'city_level'})
    # print('confounders:', confounders)
    
    # 异常值清洗
    dff = df#.copy(deep=True)
    print('原始样本数：', dff.shape[0])
    # for confounder in confounders:
    #     if confounder not in ['age', 'gender', 'city_level']:
    #         dff = dff.drop(dff[dff[confounder] >= df[confounder].quantile(.999)].index)
    #         dff = dff.drop(dff[dff[confounder] <= dff[confounder].quantile(.001)].index)
    # print('清洗后样本数：', dff.shape[0])
    
    xate_tmp = [0.0,]
    tate_tmp = [0.0,]
    psm_tmp = [0.0,]
    num_tmp = [dff[dff[treatment + '_bucket'] == 0].shape[0],]
    # num_tmp = [0.0,]
    median_tmp = [dff[dff[treatment + '_bucket'] == 0][treatment].median(),]
    
    for j in range(bins_num - 1): #修改
        # 拆分样本
        print('第0桶与第{}桶'.format(str(j+1)))
        df_tmp = dff[(dff[treatment + '_bucket'] == 0) | (dff[treatment + '_bucket' ] == j+1)]
        df_tmp['fake_sign'] = np.where(df_tmp[treatment + '_bucket'] > 0, 1, 0)
        
        # 检验是否缺失桶
        if len(set(df_tmp['fake_sign'])) < 2:
            print('第{}桶缺失'.format(str(j+1)))
            xate_tmp.append(np.nan)
            tate_tmp.append(np.nan)
            num_tmp.append(np.nan)
            median_tmp.append(np.nan)
            psm_tmp.append(np.nan)

            continue
        df_y = df_tmp['fake_sign']
        df_x = df_tmp[confounders]
        
        # 倾向得分计算
        df_tmp['propensity_score'] = psm_rfc(df_x, df_y)
        # print(df_tmp[['fake_sign','propensity_score']])
        print('匹配前样本数：', df_tmp.shape[0], df_tmp[df_tmp['fake_sign'] == 0].shape[0], df_tmp[df_tmp['fake_sign'] == 1].shape[0])
        
#         # 匹配前画图
#         print('匹配前分布')
#         plt.figure(dpi=120)
#         sns.distplot(df_tmp[df_tmp['fake_sign'] == 0]['propensity_score'], color='r', label='Control')
#         sns.distplot(df_tmp[df_tmp['fake_sign'] == 1]['propensity_score'], color='b', label='Treatment')##shade=True填充
#         plt.legend()
#         plt.show()
        
        # KNN匹配
        treatment_index, control_index = knn_match(df_tmp, bins_num)
        df_new = df_tmp.loc[treatment_index | control_index,:]
        # display(df_new)
        print('匹配后样本数：', df_new.shape[0], df_new[df_new['fake_sign'] == 0].shape[0], df_new[df_new['fake_sign'] == 1].shape[0])
        # print(df_new[['fake_sign','propensity_score']])
        
#         # 匹配后画图
#         print('匹配后分布')
#         plt.figure(dpi=120)
#         sns.distplot(df_new[df_new['fake_sign'] == 0]['propensity_score'], color='r', label='Control')
#         sns.distplot(df_new[df_new['fake_sign'] == 1]['propensity_score'], color='b', label='Treatment')
#         plt.legend()
#         plt.show()
        
        # 平衡性检验
        smd_result = pd.DataFrame(smd(df_new.loc[treatment_index,confounders], df_new.loc[control_index,confounders])).reset_index()
        smd_failed[treatment] = list(smd_result[smd_result[0] > 0.15]['index'])
        # print(list(smd_result[smd_result[0] <= 0.1]['index']))
        print('未通过检验：', smd_failed[treatment])
        # confounders_new = list(set(confounders) - set(smd_failed[treatment]) | {'age', 'gender', 'city_level'})
        confounders_new = list(set(confounders) - set(smd_failed[treatment]))
        print('confounders_new:', confounders_new)
        # print(confounders_new == list(smd_result[smd_result[0] <= 0.1]['index']))
        

        # PSM 模型
        psm_result = df_new.groupby('fake_sign')['sign'].mean()[1] - df_new.groupby('fake_sign')['sign'].mean()[0]
        psm_tmp.append(psm_result)
        
        # 因果模型
        df_x = df_new[confounders] 
        df_y = df_new['fake_sign']
        df_z = df_new[y]
        # p = ps_function_xgb(df_x, df_y)
        p = pd.DataFrame()
        p[0] = 1 - df_new['propensity_score']
        p[1] = df_new['propensity_score']
        # print(p)
        df_p = {col:np.array(p[col].tolist()) for col in p.columns}
        df_p={i:np.array([(np.maximum(x,0+10*eps)+np.minimum(x,1-10*eps))/2 for x in df_p[i]]) for i in range(2)}
        # print(df_p)
        
        # X-LEARNER
        xte = te_function_cla(df_y, df_x, df_z, df_p)
        xate_tmp.append(xte.mean())
        #T-LEARNER
        tte = te_function_cla_T(df_y, df_x, df_z, df_p)
        tate_tmp.append(tte.mean())
        
        num_tmp.append(df_new[df_new['fake_sign'] == 1].shape[0])
        median_tmp.append(df_new[df_new['fake_sign'] == 1][treatment].median())
        
        print('x-te: ', xte.mean())
        print('t-te: ', tte.mean())
        print('psm: ', psm_result)
    # 分桶边界
    ate_10[treatment]['INTERVAL'] = ['[{}, {}]'.format(str(round(interval[treatment][0],2)), 
                                                       str(round(interval[treatment][1],2))),] + ['({}, {}]'.format(str(round(interval[treatment][x],2)), str(round(interval[treatment][x+1],2))) 
                                                                                                  for x in range(1,len(interval[treatment])-1)]
    
    
    ate_10[treatment]['NUM'] = pd.DataFrame(num_tmp)[0]
    ate_10[treatment]['MEDIAN'] = pd.DataFrame(median_tmp)[0]
    
    
        
    ate_10[treatment]['X-ATE'] = pd.DataFrame(xate_tmp)[0]
    ate_10[treatment]['New X-ATE'] = [ate_10[treatment]['X-ATE'][x] / (ate_10[treatment]['MEDIAN'][x] - ate_10[treatment]['MEDIAN'][0]) for x in range(bins_num)]
    
    ate_10[treatment]['T-ATE'] = pd.DataFrame(tate_tmp)[0]
    ate_10[treatment]['New T-ATE'] = [ate_10[treatment]['T-ATE'][x] / (ate_10[treatment]['MEDIAN'][x] - ate_10[treatment]['MEDIAN'][0]) for x in range(bins_num)]
    
    ate_10[treatment]['PSM'] = pd.DataFrame(psm_tmp)[0]
    ate_10[treatment]['New PSM'] = [ate_10[treatment]['PSM'][x] / (ate_10[treatment]['MEDIAN'][x] - ate_10[treatment]['MEDIAN'][0]) for x in range(bins_num)]
    
    print('treatment \"' + treatment + '\" 汇总: ')
    display(ate_10[treatment])
    
    xate_mean = (ate_10[treatment]['New X-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum()
    tate_mean = (ate_10[treatment]['New T-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum()
    psm_mean = (ate_10[treatment]['New PSM'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum()
    print('X-ate_mean: ', xate_mean)
    print('T-ate_mean: ', tate_mean)
    print('psm_mean: ', psm_mean)
    
    xate_all.append(xate_mean)
    tate_all.append(tate_mean)
    psm_all.append(psm_mean)

print('ate_10:')
for treatment, value in ate_10.items():
    print('treatment: ', treatment)
    display(value)
    print('X-ate_mean: ', (ate_10[treatment]['New X-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('T-ate_mean: ', (ate_10[treatment]['New T-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('psm_mean: ', (ate_10[treatment]['New PSM'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
result['Treatment'] = treatments 
result['X-ATE'] = pd.DataFrame(xate_all)[0]
result['T-ATE'] = pd.DataFrame(xate_all)[0]
result['PSM'] = pd.DataFrame(psm_all)[0]
display(pd.DataFrame(result).sort_values(by = 'X-ATE', ascending = False))


# In[19]:


print('200万样本 + classifer + 10%分桶 + 带PSM + 不带启动方式')
for treatment, value in ate_10.items():
    print('treatment: ', treatment)
    display(value)
    print('X-ate_mean: ', (ate_10[treatment]['New X-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('T-ate_mean: ', (ate_10[treatment]['New T-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('psm_mean: ', (ate_10[treatment]['New PSM'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('   ')
result['Treatment'] = treatments 
result['X-ATE'] = pd.DataFrame(xate_all)[0]
result['T-ATE'] = pd.DataFrame(xate_all)[0]
result['PSM'] = pd.DataFrame(psm_all)[0]
print('单位ATE汇总结果')
display(pd.DataFrame(result).sort_values(by = 'X-ATE', ascending = False))


# In[17]:


print('200万样本 + classifer + 10%分桶 + 带PSM')
for treatment, value in ate_10.items():
    print('treatment: ', treatment)
    display(value)
    print('X-ate_mean: ', (ate_10[treatment]['New X-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('T-ate_mean: ', (ate_10[treatment]['New T-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('psm_mean: ', (ate_10[treatment]['New PSM'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('   ')
result['Treatment'] = treatments 
result['X-ATE'] = pd.DataFrame(xate_all)[0]
result['T-ATE'] = pd.DataFrame(xate_all)[0]
result['PSM'] = pd.DataFrame(psm_all)[0]
print('单位ATE汇总结果')
display(pd.DataFrame(result).sort_values(by = 'X-ATE', ascending = False))


# In[24]:


print('200万样本 + classifer + 20%分桶 + 带PSM')
for treatment, value in ate_10.items():
    print('treatment: ', treatment)
    display(value)
    print('X-ate_mean: ', (ate_10[treatment]['New X-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('T-ate_mean: ', (ate_10[treatment]['New T-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('psm_mean: ', (ate_10[treatment]['New PSM'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('   ')
result['Treatment'] = treatments 
result['X-ATE'] = pd.DataFrame(xate_all)[0]
result['T-ATE'] = pd.DataFrame(xate_all)[0]
result['PSM'] = pd.DataFrame(psm_all)[0]
print('单位ATE汇总结果')
display(pd.DataFrame(result).sort_values(by = 'X-ATE', ascending = False))


# In[21]:


print('200万样本 + classifer + 0.2分桶 + 带PSM')
for treatment, value in ate_10.items():
    print('treatment: ', treatment)
    display(value)
    print('X-ate_mean: ', (ate_10[treatment]['New X-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('T-ate_mean: ', (ate_10[treatment]['New T-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('psm_mean: ', (ate_10[treatment]['New PSM'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('   ')
result['Treatment'] = treatments 
result['X-ATE'] = pd.DataFrame(xate_all)[0]
result['T-ATE'] = pd.DataFrame(xate_all)[0]
result['PSM'] = pd.DataFrame(psm_all)[0]
print('单位ATE汇总结果')
display(pd.DataFrame(result).sort_values(by = 'X-ATE', ascending = False))


# In[85]:


print('中活300万样本 + classifer + 均值分桶 + 带PSM')
for treatment, value in ate_10.items():
    print('treatment: ', treatment)
    display(value)
    print('X-ate_mean: ', (ate_10[treatment]['New X-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('T-ate_mean: ', (ate_10[treatment]['New T-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('psm_mean: ', (ate_10[treatment]['New PSM'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('   ')
result['Treatment'] = treatments 
result['X-ATE'] = pd.DataFrame(xate_all)[0]
result['T-ATE'] = pd.DataFrame(xate_all)[0]
result['PSM'] = pd.DataFrame(psm_all)[0]
print('单位ATE汇总结果')
display(pd.DataFrame(result).sort_values(by = 'X-ATE', ascending = False))


# In[81]:


print('中活300万样本 + classifer + 中位数分桶 + 带PSM')
for treatment, value in ate_10.items():
    print('treatment: ', treatment)
    display(value)
    print('X-ate_mean: ', (ate_10[treatment]['New X-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('T-ate_mean: ', (ate_10[treatment]['New T-ATE'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('psm_mean: ', (ate_10[treatment]['New PSM'] * ate_10[treatment]['NUM']).sum() / ate_10[treatment]['NUM'][1:].sum())
    print('   ')
result['Treatment'] = treatments 
result['X-ATE'] = pd.DataFrame(xate_all)[0]
result['T-ATE'] = pd.DataFrame(xate_all)[0]
result['PSM'] = pd.DataFrame(psm_all)[0]
print('单位ATE汇总结果')
display(pd.DataFrame(result).sort_values(by = 'X-ATE', ascending = True))


# In[66]:


df_tmp


# In[ ]:




