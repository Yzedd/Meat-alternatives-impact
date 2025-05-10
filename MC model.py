import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy import stats
import statsmodels.api as sm


os.chdir('D:/Work/GPI/Data/USA/')

def correlated_monte_carlo(means, stds, rho, num_samples, lower_limits, upper_limits):

    np.random.seed(6)
    chol = np.linalg.cholesky(rho)
    
    z = np.random.standard_normal(size=(num_samples, len(means)))
    
    correlated_samples = np.matmul(z, chol)
    
    samples = correlated_samples * stds + means
    
    if lower_limits is not None and upper_limits is not None:
        samples = np.clip(samples, lower_limits, upper_limits)    
    
    return samples

def Food_Relocate(local_need,meat_type):
    #alt_ratio = np.array([1/3,1/3,1/3])
    alt_ratio = np.array([80/905,800/905,25/905])
    z = pd.read_csv('CM/CoA/STCOFIPS_2010.csv')
    update_need = local_need.copy()   
    if meat_type=='insect':
        update_need[meat_type] = 0.0
        df1 = pd.read_csv('CM/Food flow/Insect-based company.csv')
        df1 = pd.merge(df1[['HQ State (U.S. only)', 'City']], z[['STCOFIPS','livestock1','livestoc_1']], left_on=['HQ State (U.S. only)', 'City'], right_on=['livestock1','livestoc_1'])
        df1['total_need'] = local_need['protein'].sum()*alt_ratio[0]/df1.shape[0]
        df0 = df1['STCOFIPS'].unique()
        for i in range(df0.shape[0]):
            update_need.loc[(update_need['STCOFIPS'] == df0[i]),[meat_type]] = df1[df1['STCOFIPS']==df0[i]]['total_need'].sum()
    if meat_type=='plant':
        update_need[meat_type] = 0.0
        df1 = pd.read_csv('CM/Food flow/Plant-based company.csv')
        df1 = pd.merge(df1[['HQ State (U.S. only)', 'City']], z[['STCOFIPS','livestock1','livestoc_1']], left_on=['HQ State (U.S. only)', 'City'], right_on=['livestock1','livestoc_1'])
        df1['total_need'] = local_need['protein'].sum()*alt_ratio[1]/df1.shape[0]
        df0 = df1['STCOFIPS'].unique()
        for i in range(df0.shape[0]):
            update_need.loc[(update_need['STCOFIPS'] == df0[i]),[meat_type]] = df1[df1['STCOFIPS']==df0[i]]['total_need'].sum()
    if meat_type=='cultured':
        update_need[meat_type] = 0.0
        df1 = pd.read_csv('CM/Food flow/Cell-based company.csv')
        df1 = pd.merge(df1[['HQ State (U.S. only)', 'City']], z[['STCOFIPS','livestock1','livestoc_1']], left_on=['HQ State (U.S. only)', 'City'], right_on=['livestock1','livestoc_1'])
        df1['total_need'] = local_need['protein'].sum()*alt_ratio[2]/df1.shape[0]
        df0 = df1['STCOFIPS'].unique()
        for i in range(df0.shape[0]):
            update_need.loc[(update_need['STCOFIPS'] == df0[i]),[meat_type]] = df1[df1['STCOFIPS']==df0[i]]['total_need'].sum()
    return update_need

def Feed_Relocate(local_need):
    update_need = local_need.copy()
    update_need['tran_corn']  = 0.0
    update_need['tran_soy']  = 0.0
    update_need['tran_corn_v2']  = 0.0
    update_need['tran_soy_v2']  = 0.0
    z = pd.read_csv('CM/Food flow/import_percent.csv')
    for i in range (local_need.shape[0]):
        df0 = z[z['des']==local_need.loc[i,'STCOFIPS']].reset_index(drop=True)
        update_need.loc[i,'c_meat_corn'] = local_need.loc[i,'c_meat_corn']*(1-df0['corn_perc'].sum())
        update_need.loc[i,'c_meat_soy'] = local_need.loc[i,'c_meat_soy']*(1-df0['soy_perc'].sum())
        for j in range(df0.shape[0]):
            update_need.loc[update_need['STCOFIPS']==df0.loc[j,'ori'],'tran_corn'] = update_need.loc[update_need['STCOFIPS']==df0.loc[j,'ori'],'tran_corn']+local_need.loc[i,'c_meat_corn']*df0.loc[j,'corn_perc']
            update_need.loc[update_need['STCOFIPS']==df0.loc[j,'ori'],'tran_soy'] = update_need.loc[update_need['STCOFIPS']==df0.loc[j,'ori'],'tran_soy']+local_need.loc[i,'c_meat_soy']*df0.loc[j,'soy_perc']
        if local_need.loc[i,'c_alt_corn'] > 0.2*local_need.loc[i,'corng']*1000/39.368:
            update_need.loc[i,'c_alt_corn'] = 0.2*local_need.loc[i,'corng']*1000/39.368
            for j in range(df0.shape[0]):
                update_need.loc[update_need['STCOFIPS']==df0.loc[j,'ori'],'tran_corn_v2'] = update_need.loc[update_need['STCOFIPS']==df0.loc[j,'ori'],'tran_corn_v2']+(local_need.loc[i,'c_alt_corn']-update_need.loc[i,'c_alt_corn'])*df0.loc[j,'corn_perc_v2']
        if local_need.loc[i,'c_alt_soy'] > 0.2*local_need.loc[i,'soybean']*1000/36.744:
            update_need.loc[i,'c_alt_soy'] = 0.2*local_need.loc[i,'soybean']*1000/36.744
            for j in range(df0.shape[0]):
                update_need.loc[update_need['STCOFIPS']==df0.loc[j,'ori'],'tran_soy_v2'] = update_need.loc[update_need['STCOFIPS']==df0.loc[j,'ori'],'tran_soy_v2']+(local_need.loc[i,'c_alt_soy']-update_need.loc[i,'c_alt_soy'])*df0.loc[j,'soy_perc_v2']
    update_need.loc[i,'c_meat_corn'] = update_need.loc[i,'c_meat_corn']+update_need.loc[i,'tran_corn']
    update_need.loc[i,'c_meat_soy'] =  update_need.loc[i,'c_meat_soy']+ update_need.loc[i,'tran_soy']
    update_need.loc[i,'c_alt_corn'] = update_need.loc[i,'c_alt_corn']+update_need.loc[i,'tran_corn_v2']
    update_need.loc[i,'c_alt_soy'] =  update_need.loc[i,'c_alt_soy']+ update_need.loc[i,'tran_soy_v2']
    update_need['c_corn'] = update_need['c_meat_corn']-update_need['c_alt_corn']
    update_need['c_soy'] = update_need['c_meat_soy']-update_need['c_alt_soy']
    
    h_corn = update_need.sort_values('corng',ascending=False).head(10)
    h_soy = update_need.sort_values('soybean',ascending=False).head(10)
    
    for i in range (update_need.shape[0]):
        if update_need.loc[i,'c_corn'] > update_need.loc[i,'corng']*1000/39.368:
            h_corn_v2 = h_corn.copy()
            h_corn_v2['dif'] = h_corn_v2['corng']*1000/39.368-h_corn_v2['c_corn']
            h_corn_v2 = h_corn_v2[h_corn_v2['dif']>0].reset_index(drop=True)
            h_corn_v2['perc'] = h_corn_v2['corng']/(h_corn_v2['corng'].sum())
            for j in range(h_corn_v2.shape[0]):
                update_need.loc[h_corn_v2.index[j],'c_corn'] = update_need.loc[h_corn_v2.index[j],'c_corn']+(update_need.loc[i,'c_corn']-update_need.loc[i,'corng']*1000/39.368)*h_corn_v2.loc[h_corn_v2.index[j],'perc']
            update_need.loc[i,'c_corn'] = update_need.loc[i,'corng']*1000/39.368
        if update_need.loc[i,'c_soy'] > update_need.loc[i,'soybean']*1000/36.744:
            h_soy_v2 = h_soy.copy()
            h_soy_v2['dif'] = h_soy_v2['soybean']*1000/36.744-h_soy_v2['c_soy']
            h_soy_v2 = h_soy_v2[h_soy_v2['dif']>0].reset_index(drop=True)
            h_soy_v2['perc'] = h_soy_v2['soybean']/(h_soy_v2['soybean'].sum())
            for j in range(h_soy_v2.shape[0]):
                update_need.loc[h_soy_v2.index[j],'c_soy'] = update_need.loc[h_soy_v2.index[j],'c_soy']+(update_need.loc[i,'c_soy']-update_need.loc[i,'soybean']*1000/36.744)*h_soy_v2.loc[h_soy_v2.index[j],'perc']
            update_need.loc[i,'c_soy'] = update_need.loc[i,'soybean']*1000/36.744
    return update_need

df1 = pd.read_csv('CM/Spatial/fip_region.csv')
df1 = df1.sort_values('A_inter',ascending=False)
df1 = df1.drop_duplicates(subset=['STCOFIPS'],keep='first',ignore_index=True)
df1['R_water'] = 0.0
df1.loc[(df1['Region'] == 'Northeast'),['R_water']] = 0.080555445
df1.loc[(df1['Region'] == 'Southeast'),['R_water']] = 0.12300626
df1.loc[(df1['Region'] == 'Midwest'),['R_water']] = 0.142553985
df1.loc[(df1['Region'] == 'Northern Plains'),['R_water']] = 0.885041344
df1.loc[(df1['Region'] == 'Southern Plains'),['R_water']] = 0.965904382
df1.loc[(df1['Region'] == 'Northwest'),['R_water']] = 4.221184593
df1.loc[(df1['Region'] == 'Southwest'),['R_water']] = 2.592962714
#df1['R_water'] = 1.0

df2 = pd.read_csv('CM/Spatial/N_need.csv')
df0 = pd.merge(df1,df2, left_on='STCOFIPS', right_on='FIPS', how='left')
df00 = df0.drop(df0[df0['N_needs_kg_ha'].isnull()].index)
df00['STCOFIPS'] = df00['STCOFIPS'].astype(str)
df00['head'] = df00['STCOFIPS'].str[:-3]
for i in range(df0.shape[0]):
    if np.isnan(df0.loc[i,'N_needs_kg_ha']):
        df0.loc[i,'N_needs_kg_ha'] = df00[df00['head']==str(df0.loc[i,'STCOFIPS'])[:-3]]['N_needs_kg_ha'].mean()
df0['R_N'] = df0['N_needs_kg_ha']/(np.sum(df0['N_needs_kg_ha']*df0['Area'])/np.sum(df0['Area']))
df0.loc[(df0[df0['R_N'].isnull()].index),['R_N']] = 1
#df0['R_N'] = 1.0

df3 = pd.read_csv('CM/CoA/STCOFIPS_2010.csv')
df0 = pd.merge(df0,df3, on='STCOFIPS')

df = pd.read_csv('CM/cm_data_v2.csv')
meatlabel = np.array([0,4,7])
altlabel = np.array([9,11,13])
feed_type = np.array([[2,4,7],[3,5,8]])
feed_source = np.array([[0.5,0,1],[0.5,1,1]])
N = 1000
num_perc_change = 6
meatprotein = np.zeros(len(meatlabel))
altprotein = np.zeros(len(altlabel))
altfeed = np.zeros(len(altlabel))
nprod = np.zeros((2,N,num_perc_change))
ppc = np.zeros((len(meatlabel),N,num_perc_change))
afr = np.zeros((len(altlabel),N,num_perc_change))
fr1 = np.zeros((3,N,num_perc_change))
fr2 = np.zeros((3,N,num_perc_change))
fm1 = np.zeros((3,N,num_perc_change))
fm2 = np.zeros((3,N,num_perc_change))
cc = np.zeros((2,N,num_perc_change))
nfc = np.zeros((2,N,num_perc_change))
wfc = np.zeros((6,N,num_perc_change))
mc = np.zeros((6,N,num_perc_change))
df2 = df0.copy()
total = np.zeros((5,df2.shape[0],N,num_perc_change))

limit = [[0.014,df.iloc[meatlabel[0],8],df.iloc[meatlabel[0],5],
          1,df.iloc[meatlabel[1],8],df.iloc[meatlabel[1],5],
          5,df.iloc[meatlabel[2],8],df.iloc[meatlabel[2],5]],
         [1/46,df.iloc[meatlabel[0],9],df.iloc[meatlabel[0],6],
          2,df.iloc[meatlabel[1],9],df.iloc[meatlabel[1],6],
          6,df.iloc[meatlabel[2],9],df.iloc[meatlabel[2],6]]]

means = [np.mean(np.linspace(limit[0][0],limit[1][0],1000)),np.mean(np.linspace(limit[0][1],limit[1][1],1000)),np.mean(np.linspace(limit[0][2],limit[1][2],1000)),
         np.mean(np.linspace(limit[0][3],limit[1][3],1000)),np.mean(np.linspace(limit[0][4],limit[1][4],1000)),np.mean(np.linspace(limit[0][5],limit[1][5],1000)),
         np.mean(np.linspace(limit[0][6],limit[1][6],1000)),np.mean(np.linspace(limit[0][7],limit[1][7],1000)),np.mean(np.linspace(limit[0][8],limit[1][8],1000))]
stds = [np.std(np.linspace(limit[0][0],limit[1][0],1000)),np.std(np.linspace(limit[0][1],limit[1][1],1000)),np.std(np.linspace(limit[0][2],limit[1][2],1000)),
         np.std(np.linspace(limit[0][3],limit[1][3],1000)),np.std(np.linspace(limit[0][4],limit[1][4],1000)),np.std(np.linspace(limit[0][5],limit[1][5],1000)),
         np.std(np.linspace(limit[0][6],limit[1][6],1000)),np.std(np.linspace(limit[0][7],limit[1][7],1000)),np.std(np.linspace(limit[0][8],limit[1][8],1000))]
rho = df4.corr()
sample = correlated_monte_carlo(means, stds, rho, 1000, limit[0], limit[1])

for k in range (0,51,10):
#for k in range (10,11,10): 
    change_variable = k/100
    for i in range (N):
    #for i in range (1):
        
        random.seed(i)
        df2['beefcow_v2'] = df2['beefcow']*346.1
        df2['broiler_v2'] = df2['broiler']*2
        df2['hog_v2'] = df2['hog']*115
        df2['c_bcows'] = df2['beefcow_v2']*(change_variable)*(df.iloc[meatlabel[0],7]/sum(df2['beefcow_v2']))
        df2['c_broilers'] = df2['broiler_v2']*(change_variable)*(df.iloc[meatlabel[1],7]/sum(df2['broiler_v2']))
        df2['c_hogs'] = df2['hog_v2']*(change_variable)*(df.iloc[meatlabel[2],7]/sum(df2['hog_v2']))
        df2['c_bcows_protein'] = df2['c_bcows']*random.uniform(df.iloc[meatlabel[0],3],df.iloc[meatlabel[0],4])/100
        df2['c_broilers_protein'] = df2['c_broilers']*random.uniform(df.iloc[meatlabel[1],3],df.iloc[meatlabel[1],4])/100
        df2['c_hogs_protein'] = df2['c_hogs']*random.uniform(df.iloc[meatlabel[2],3],df.iloc[meatlabel[2],4])/100
        df2['protein'] = df2['c_bcows_protein']+df2['c_broilers_protein']+df2['c_hogs_protein']
        # Food relocate
        df2 = Food_Relocate(df2,'insect')
        df2 = Food_Relocate(df2,'plant')
        df2 = Food_Relocate(df2,'cultured')
        df2['insect_corn'] = (df2['insect']*random.uniform(df.iloc[altlabel[0],2],df.iloc[altlabel[0]+1,2])*feed_source[0,0])
        df2['plant_corn'] = (df2['plant']*random.uniform(df.iloc[altlabel[1],2],df.iloc[altlabel[1]+1,2])*feed_source[0,1])
        df2['cultured_corn'] = (df2['cultured']*random.uniform(df.iloc[altlabel[2],2],df.iloc[altlabel[2]+1,2])*feed_source[0,2])
        df2['insect_soy'] = (df2['insect']*random.uniform(df.iloc[altlabel[0],2],df.iloc[altlabel[0]+1,2])*feed_source[1,0])
        df2['plant_soy'] = (df2['plant']*random.uniform(df.iloc[altlabel[1],2],df.iloc[altlabel[1]+1,2])*feed_source[1,1])
        df2['cultured_soy'] = (df2['cultured']*random.uniform(df.iloc[altlabel[2],2],df.iloc[altlabel[2]+2,2])*feed_source[1,2])
        df2['insect'] = df2['insect']/(random.uniform(df.iloc[altlabel[0],3],df.iloc[altlabel[0],4])/100)
        df2['plant'] = df2['plant']/(random.uniform(df.iloc[altlabel[1],3],df.iloc[altlabel[1],4])/100)
        df2['cultured'] = df2['cultured']/(random.uniform(df.iloc[altlabel[2],3],df.iloc[altlabel[2],4])/100)
        df2['c_bcows_corn'] = (df2['c_bcows_protein']*df.iloc[feed_type[0,0],2])
        df2['c_broilers_corn'] = (df2['c_broilers_protein']*df.iloc[feed_type[0,1],2])
        df2['c_hogs_corn'] = (df2['c_hogs_protein']*df.iloc[feed_type[0,2],2])
        df2['c_bcows_soy'] = (df2['c_bcows_protein']*df.iloc[feed_type[1,0],2])
        df2['c_broilers_soy'] = (df2['c_broilers_protein']*df.iloc[feed_type[1,1],2])
        df2['c_hogs_soy'] = (df2['c_hogs_protein']*df.iloc[feed_type[1,2],2])
        df2['c_meat_corn'] = df2['c_bcows_corn']+df2['c_broilers_corn']+df2['c_hogs_corn']
        df2['c_alt_corn'] = df2['insect_corn']+df2['plant_corn']+df2['cultured_corn']
        df2['c_meat_soy'] = df2['c_bcows_soy']+df2['c_broilers_soy']+df2['c_hogs_soy']
        df2['c_alt_soy'] = df2['insect_soy']+df2['plant_soy']+df2['cultured_soy']
        # Resource relocate
        df2 = Feed_Relocate(df2)
        total[0,:,i,int(k/10)] = df2['c_corn']
        total[1,:,i,int(k/10)] = df2['c_soy']
        nprod[0,i,int(k/10)] = sample[i,0]#random.uniform(164/10920, 1/46)
        nprod[1,i,int(k/10)] = random.uniform(0, 80/4225)
        df2['c_farmn'] = df2['c_corn']*nprod[0,i,int(k/10)]*df2['R_N']+df2['c_soy']*nprod[1,i,int(k/10)]
        total[2,:,i,int(k/10)] = df2['c_farmn']
        # Water relocate ???
        #df2['c_bcows_water'] = df2['c_bcows']*(random.uniform(df.iloc[meatlabel[0],5],df.iloc[meatlabel[0],6])/df.iloc[meatlabel[0],7])*df2['R_water']
        #df2['c_broilers_water'] = df2['c_broilers']*(random.uniform(df.iloc[meatlabel[1],5],df.iloc[meatlabel[1],6])/df.iloc[meatlabel[1],7])
        #df2['c_hogs_water'] = df2['c_hogs']*(random.uniform(df.iloc[meatlabel[2],5],df.iloc[meatlabel[2],6])/df.iloc[meatlabel[2],7])
        df2['c_bcows_water'] = df2['c_bcows']*(sample[i,2]/df.iloc[meatlabel[0],7])*df2['R_water']
        df2['c_broilers_water'] = df2['c_broilers']*(sample[i,5]/df.iloc[meatlabel[1],7])
        df2['c_hogs_water'] = df2['c_hogs']*(sample[i,8]/df.iloc[meatlabel[2],7])
        df2['insect_water'] = df2['insect']*random.uniform(df.iloc[altlabel[0],5], df.iloc[altlabel[0],6])/1000/(1e6)
        df2['plant_water'] = df2['plant']*random.uniform(df.iloc[altlabel[1],5], df.iloc[altlabel[1],6])/1000/(1e6)
        df2['cultured_water'] = df2['cultured']*random.uniform(df.iloc[altlabel[2],5], df.iloc[altlabel[2],6])/1000/(1e6)
        df2['c_wu_mgd'] = df2['c_bcows_water']+df2['c_broilers_water']+df2['c_hogs_water']-df2['insect_water']-df2['plant_water']-df2['cultured_water']
        total[3,:,i,int(k/10)] = df2['c_wu_mgd']
        #df2['c_bcows_manure'] = df2['c_bcows']*random.uniform(df.iloc[meatlabel[0],8],df.iloc[meatlabel[0],9])
        #df2['c_broilers_manure'] = df2['c_broilers']*random.uniform(df.iloc[meatlabel[1],8],df.iloc[meatlabel[1],9])
        #df2['c_hogs_manure'] = df2['c_hogs']*random.uniform(df.iloc[meatlabel[2],8],df.iloc[meatlabel[2],9])
        df2['c_bcows_manure'] = df2['c_bcows']*sample[i,1]
        df2['c_broilers_manure'] = df2['c_broilers']*sample[i,4]
        df2['c_hogs_manure'] = df2['c_hogs']*sample[i,7]
        df2['c_N_MAN_TOT'] = df2['c_bcows_manure']+df2['c_broilers_manure']+df2['c_hogs_manure']
        total[4,:,i,int(k/10)] = df2['c_N_MAN_TOT']
        
        # Alternative fertilizer requirment
        fr1[0,i,int(k/10)] = sum(df2['insect_corn']*nprod[0,i,int(k/10)]*df2['R_N'])
        fr1[1,i,int(k/10)] = sum(df2['plant_corn']*nprod[0,i,int(k/10)]*df2['R_N'])
        fr1[2,i,int(k/10)] = sum(df2['cultured_corn']*nprod[0,i,int(k/10)]*df2['R_N'])
        fr2[0,i,int(k/10)] = sum(df2['insect_soy']*nprod[1,i,int(k/10)])
        fr2[1,i,int(k/10)] = sum(df2['plant_soy']*nprod[1,i,int(k/10)])
        fr2[2,i,int(k/10)] = sum(df2['cultured_soy']*nprod[1,i,int(k/10)])
        
        # Fertilizer for meat
        fm1[0,i,int(k/10)] = sum(df2['c_bcows_corn']*nprod[0,i,int(k/10)]*df2['R_N'])
        fm1[1,i,int(k/10)] = sum(df2['c_broilers_corn']*nprod[0,i,int(k/10)]*df2['R_N'])
        fm1[2,i,int(k/10)] = sum(df2['c_hogs_corn']*nprod[0,i,int(k/10)]*df2['R_N'])
        fm2[0,i,int(k/10)] = sum(df2['c_bcows_soy']*nprod[1,i,int(k/10)])
        fm2[1,i,int(k/10)] = sum(df2['c_broilers_soy']*nprod[1,i,int(k/10)])
        fm2[2,i,int(k/10)] = sum(df2['c_hogs_soy']*nprod[1,i,int(k/10)])
        
        # Crop change
        cc[0,i,int(k/10)] = sum(df2['c_corn'])
        cc[1,i,int(k/10)] = sum(df2['c_soy'])
        
        # N fertilizer change
        nfc[0,i,int(k/10)] = (sum(fm1[:,i,int(k/10)])-sum(fr1[:,i,int(k/10)]))
        nfc[1,i,int(k/10)] = (sum(fm2[:,i,int(k/10)])-sum(fr2[:,i,int(k/10)]))
        
        # Water footprint change (million m3)
        wfc[0,i,int(k/10)] = sum(df2['c_bcows_water'])
        wfc[1,i,int(k/10)] = sum(df2['c_broilers_water'])
        wfc[2,i,int(k/10)] = sum(df2['c_hogs_water'])
        wfc[3,i,int(k/10)] = sum(df2['insect_water'])
        wfc[4,i,int(k/10)] = sum(df2['plant_water'])
        wfc[5,i,int(k/10)] = sum(df2['cultured_water'])

        # Manure change (kg N)
        mc[0,i,int(k/10)] = sum(df2['c_bcows_manure'])
        mc[1,i,int(k/10)] = sum(df2['c_broilers_manure'])
        mc[2,i,int(k/10)] = sum(df2['c_hogs_manure'])
        mc[3,i,int(k/10)] = 0
        mc[4,i,int(k/10)] = 0
        mc[5,i,int(k/10)] = 0
