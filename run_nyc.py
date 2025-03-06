import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import os
import random
from model import *
# from model import *
from evaluation import evaluation

torch.manual_seed(123)

# <editor-fold, desc='load train, validation and test dataset'>
all_data = pd.read_csv('NYC.csv')
num_users = max(all_data.userid.tolist()) + 1
num_items = max(all_data.itemid.tolist()) + 1
num_timesegment = 8

print('#user:{:.0f}, #category:{:.0f}, #interation:{:.0f}'.format(num_users, num_items, len(all_data)))

tr_df = pd.read_csv('train.csv')
va_df = pd.read_csv('validation.csv')
ts_df = pd.read_csv('test.csv')


# validation
va_u_ls = va_df.userid.tolist()
va_i_ls = va_df.itemid.tolist()
va_t_ls = va_df.timesegment.tolist()

va_target = torch.zeros((len(va_df), num_items))
for idx, i in enumerate(va_i_ls):
    va_target[idx, i] = 1

# test
ts_u_ls = ts_df.userid.tolist()
ts_i_ls = ts_df.itemid.tolist()
ts_t_ls = ts_df.timesegment.tolist()

ts_target = torch.zeros((len(ts_df), num_items))
for idx, i in enumerate(ts_i_ls):
    ts_target[idx, i] = 1
# </editor-fold>


# ##############################################################################
# ########################## 构建三阶张量，用于捕获用户长期兴趣 ####################
# ##############################################################################
# <editor-fold, desc='3rd order tensor completion'>
tr_u_ls = tr_df.userid.tolist()
tr_i_ls = tr_df.itemid.tolist()
tr_t_ls = tr_df.timesegment.tolist()
M_3rd = torch.zeros((num_users, num_items, num_timesegment))
for u,i,t in zip(tr_u_ls, tr_i_ls, tr_t_ls):
    M_3rd[u,i,t] += 1
# 标准化
for u in range(num_users):
    for t in range(num_timesegment):
        if M_3rd[u, :, t].sum() == 0: pass
        else:
            M_3rd[u, :, t] = M_3rd[u, :, t] / M_3rd[u,:,t].sum()

omega_3rd = np.nonzero(M_3rd)   # 三阶张量中可见的元素的索引集合
# </editor-fold>


# #################################################################################
# ############################### 构建四阶张量 ######################################
# #################################################################################
# <editor-fold, desc=4th-order TC>
tr_4th = tr_df.groupby('userid')
u_4th_ls = []
last_4th_ls = []
next_4th_ls = []
timesegment_4th_ls = []

for uid, u_data in tr_4th:
    u_data = u_data.sort_values(by='timestamp')
    u_4th_ls += [uid for u in range(len(u_data)-1)]
    i_ls = u_data.itemid.tolist()
    last_4th_ls += i_ls[:-1]
    next_4th_ls += i_ls[1:]
    timesegment_4th_ls += u_data.timesegment.tolist()[1:]

M_4th = torch.zeros((num_users, num_items, num_items, num_timesegment))
for u, l, n, t in zip(u_4th_ls, last_4th_ls, next_4th_ls, timesegment_4th_ls):
    M_4th[u,l,n,t] += 1
# 标准化
for u in range(num_users):
    for t in range(num_timesegment):
        if M_4th[u,:,:,t].sum() == 0: pass
        else:
            M_4th[u,:,:,t] = M_4th[u,:,:,t] / M_4th[u,:,:,t].sum()

omega_4th = np.nonzero(M_4th)
# </editor-fold>



# ##########################################################################
# ############################### 合并 ######################################
# ##########################################################################
# <editor-fold, desc='tuning lambda'>
ru, ri, rt = [int(0.3*M_3rd.shape[0]), int(0.3*M_3rd.shape[1]), 8]
R3 = [ru,ri,rt]
X_3rd, rse = ORPTC(M_3rd, omega_3rd, R=R3)

pred_3rd = torch.zeros((len(va_df), num_items))
idx = 0
for u, t in zip(va_u_ls, va_t_ls):
    pred_3rd[idx] = X_3rd[u, :, t]
    idx += 1


tr_4th = tr_df.groupby('userid')
last_period_data_dict = {}
for u, u_data in tr_4th:
    u_data = u_data.sort_values(by='timestamp')
    last_week = u_data.num_of_week.unique()[-1]
    last_period_data_dict[u] = u_data[u_data['num_of_week']==last_week].itemid.tolist()

R4 = [ru, ri, ri, rt]
X_4th, rse = ORPTC(M_4th, omega_4th, R=R4, maxiter=20)

pred_4th = torch.zeros((len(va_df), num_items))
idx = 0
tr_user_ls = tr_df.userid.tolist()
for u, t in zip(va_u_ls, va_t_ls):
    if u in tr_user_ls:
        pred_4th[idx] = torch.mean(X_4th[u, last_period_data_dict[u], :, t], dim=0)
    else:
        pred_4th[idx] = torch.mean(X_4th[u, :, :, t], dim=0)
    idx += 1

best_lamb = 0
best_hr1, best_hr3, best_hr5, best_ndcg3, best_ndcg5 = 0,0,0,0,0
lamb_list = [i*0.05 for i in range(int(1/0.05) + 1)]
for lamb in lamb_list:
    prediction = lamb * pred_3rd + (1-lamb) * pred_4th
    hr1, hr3, hr5, ndcg3, ndcg5 = evaluation(va_target, prediction)
    # print('hr1:{:.4f}'.format(hr1))
    # print('hr3:{:.4f}'.format(hr3))
    # print('hr5:{:.4f}'.format(hr5))
    # print('ndcg3:{:.4f}'.format(ndcg3))
    # print('ndcg5:{:.4f}'.format(ndcg5))
    # print('\n')
    if best_hr1 < hr1:
        best_hr1 = hr1
        best_hr3 = hr3
        best_hr5 = hr5
        best_ndcg3 = ndcg3
        best_ndcg5 = ndcg5
        best_lamb = lamb
print('best lambda:', best_lamb)
print('hr1:{:.4f}'.format(best_hr1))
print('hr3:{:.4f}'.format(best_hr3))
print('hr5:{:.4f}'.format(best_hr5))
print('ndcg3:{:.4f}'.format(best_ndcg3))
print('ndcg5:{:.4f}'.format(best_ndcg5))

# </editor-fold>

# ##########################################################################
# ############################### 测试 ######################################
# ##########################################################################
# <editor-fold, desc='test'>
pred_3rd = torch.zeros((len(ts_df), num_items))
idx = 0
for u, t in zip(ts_u_ls, ts_t_ls):
    pred_3rd[idx] = X_3rd[u, :, t]
    idx += 1
hr1, hr3, hr5, ndcg3, ndcg5 = evaluation(ts_target, pred_3rd)
print('3rd-order test:')
print('hr1:{:.4f}'.format(hr1))
print('hr3:{:.4f}'.format(hr3))
print('hr5:{:.4f}'.format(hr5))
print('ndcg3:{:.4f}'.format(ndcg3))
print('ndcg5:{:.4f}'.format(ndcg5))
print('\n')


tr_4th = tr_df.groupby('userid')
last_period_data_dict = {}
for u, u_data in tr_4th:
    u_data = u_data.sort_values(by='timestamp')
    last_week = u_data.num_of_week.unique()[-1]
    last_period_data_dict[u] = u_data[u_data['num_of_week']==last_week].itemid.tolist()

pred_4th = torch.zeros((len(ts_df), num_items))
idx = 0
tr_user_ls = tr_df.userid.tolist()
for u, t in zip(ts_u_ls, ts_t_ls):
    if u in tr_user_ls:
        pred_4th[idx] = torch.mean(X_4th[u, last_period_data_dict[u], :, t], dim=0)
    else:
        pred_4th[idx] = torch.mean(X_4th[u, :, :, t], dim=0)
    idx += 1

hr1, hr3, hr5, ndcg3, ndcg5 = evaluation(ts_target, pred_4th)
print('4th-order test:')
print('hr1:{:.4f}'.format(hr1))
print('hr3:{:.4f}'.format(hr3))
print('hr5:{:.4f}'.format(hr5))
print('ndcg3:{:.4f}'.format(ndcg3))
print('ndcg5:{:.4f}'.format(ndcg5))
print('\n')


prediction = best_lamb * pred_3rd + (1-best_lamb) * pred_4th
hr1, hr3, hr5, ndcg3, ndcg5 = evaluation(ts_target, prediction)
print('test:')
print('ru:', ru, ',ri:', ri, ',rt:', rt)
print('lambda:', best_lamb)
print('hr1:{:.4f}'.format(hr1))
print('hr3:{:.4f}'.format(hr3))
print('hr5:{:.4f}'.format(hr5))
print('ndcg3:{:.4f}'.format(ndcg3))
print('ndcg5:{:.4f}'.format(ndcg5))
# </editor-fold>


