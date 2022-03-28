## Connectome-based Predictive Modeling using HCP dataset
## TH Kim, 211015, ver1

## reference
# 1. Barch, D. M., Burgess, G. C., Harms, M. P., Petersen, S. E., Schlaggar, B. L., Corbetta, M., ... & Van Essen, D. C. (2013). Function in the human connectome: task-fMRI and individual differences in behavior. Neuroimage, 80, 169-189.
# 2. https://github.com/esfinn/cpm_tutorial
# 3. Finn ES, Shen X, Scheinost D, Rosenberg MD, Huang J, Chun MM, Papademetris X, Constable RT. (2015) Functional connectome fingerprinting: Identifying individuals using patterns of brain connectivity. Nature Neuroscience, 18: 1664–1671.
# 4. Shen X, Finn ES, Scheinost D, Rosenberg MD, Chun MM, Papademetris X, Constable RT. (2017). Using connectome-based predictive modeling to predict individual behavior from brain connectivity. Nature Protocols 12: 506-18.

## Preparation

# uitility
import os, warnings, sys, datetime, argparse, time
from argparse import RawTextHelpFormatter
from copy import deepcopy

# data handling & statistics
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg

# modeling
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import f_regression, SelectKBest, SelectPercentile
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

## define Class
class Cpm_Auto():
    def __init__( self
                , tBehav
                , tfMRI
                , tNet
                , tScale
                , tFs
                , tFs_alpha
                , tMdl
                , opt_tComp
                , opt_nComp):
        
        self.tBehav = tBehav
        self.tfMRI = tfMRI
        self.tNet = tNet
        self.tScale = tScale
        self.tFs = tFs 
        self.tFs_alpha = tFs_alpha
        self.tMdl = tMdl
        self.opt_tComp = opt_tComp
        self.opt_nComp = opt_nComp

    # load data & preparation
    def load_behavData(self, tBehav):
        target_name = behav_list[tBehav]
        behav_dat = pd.read_csv('%s/behav.csv' % (hcp_dir))
        behav_dat.columns = ['sn', 'gender', 'age',
                             'fi', 'wm', 'viq1', 'viq2', 'ps', 'lm', 'dd']
        label = behav_dat.loc[:, ['sn', ('%s' % (target_name))]]
        subj_idx = list(behav_dat.sn.values)
        return label, subj_idx

    def load_edgeData(self):
        edge = pd.read_csv('%s/shen_edge_info.csv' % (hcp_dir))
        return edge
    
    def load_fcData(self, tfMRI, edge, tNet):
        target_name = fMRI_list[tfMRI]
        fMRI_dat = pd.read_csv(('%s/fc_%s_df.csv' % (hcp_dir, target_name)))
        fMRI_dat.rename(columns={'Unnamed: 0': 'sn'}, inplace=True)
        if tNet == 0:
            data = fMRI_dat.copy()
        elif tNet != 0:
            tmp_df = fMRI_dat.loc[:, '0':]
            tmp_df = pd.concat([pd.DataFrame(edge['net']),
                                tmp_df.T.reset_index(drop=True)], axis=1).T
            data = tmp_df.loc[0:, tmp_df.loc['net', :] == net_list[tNet]]
            nEdge = data.shape[1]
            data = pd.concat([fMRI_dat['sn'], data], axis=1)
        return data
    
    def load_data(self, tBehav, tfMRI, tNet):
        label, subj_idx = self.load_behavData(tBehav)
        edge = self.load_edgeData()
        data = self.load_fcData(tfMRI, edge, tNet)
        return data, label, subj_idx
        
        
    # preprocessing & feature engineering
    def scaling(self, tScale, x1, x2):
        if scale_list[tScale] == 'none':
            x_train = x1
            x_test = x2
        elif scale_list[tScale] != 'none':
            if scale_list[tScale] == 'std':
                scaler = StandardScaler()
            elif scale_list[tScale] == 'robust':
                scaler = RobustScaler()
            elif scale_list[tScale] == 'minmax':
                scaler = MinMaxScaler()

            targ_col = x1.columns.values
            x_train = scaler.fit_transform(x1)
            x_train = pd.DataFrame(x_train)
            x_train.columns = targ_col
            x_test = scaler.transform(x2)
            x_test = pd.DataFrame(x_test)
            x_train.columns = targ_col
        return x_train, x_test

    def feature_selecting(self, tFs, x1, y1, x2, y2, tFs_alpha = 0.05):
        x_train = x1
        y_train = y1
        x_test = x2
        y_test = y2
        
        if tFs == 'none':
            y_train = np.array(x_train)
            y_test = np.array(x_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
        elif tFs == 'f-reg':
            selector = SelectKBest(f_regression, k='all')
            selector.fit(x_train, y_train)
            is_support = select.pvalues_ < tFs_alpha
            x_train = np.array(x_train[x_train.columns[is_support].values])
            x_test = np.array(x_test[x_test.columns[is_support].values])
            y_train = np.array(y_train)
            y_test = np.array(y_test)
        return x_train, y_train, x_test, y_test

    def pc_optimization(self, tMdl, x, y, n_comp, cv=10):
        model_list = ['Linear', 'Ridge', 'PCAreg', 'PLSreg']
        cMdl = model_list[tMdl]
        r2 = []
        rmse = []
        mae = []
        mape = []

        if cMdl == 'PCAreg':
            for nComp in range(1, n_comp+1):
                pca_mdl = PCA(n_components = nComp)
                PCs = pca_mdl.fit_transform(x)
                mdl = LinearRegression()
                y_cv = cross_val_predict(mdl, PCs, y, cv=cv)
                r2.append(r2_score(y, y_cv))
                rmse.append(np.sqrt(mean_squared_error(y, y_cv)))
                mae.append(mean_absolute_error(y, y_cv))
                mape.append(mean_absolute_percentage_error(y, y_cv))

        elif cMdl == 'PLSreg':
            for nComp in range(1, n_comp+1):
                mdl = PLSRegression(n_components = nComp)
                y_cv = cross_val_predict(mdl, x, y, cv=cv)
                r2.append(r2_score(y, y_cv))
                rmse.append(np.sqrt(mean_squared_error(y, y_cv)))
                mae.append(mean_absolute_error(y, y_cv))
                mape.append(mean_absolute_percentage_error(y, y_cv))

        return r2, rmse, mae, mape

    def pick_nComp(self, v1, v2, v3, v4, label1, label2, label3, label4, obj1, obj2, obj3, obj4):
        if obj1 == 'min':
            idx1 = np.argmin(v1)+1
        else:
            idx1 = np.argmax(v1)+1

        if obj2 == 'min':
            idx2 = np.argmin(v2)+1
        else:
            idx2 = np.argmax(v2)+1

        if obj3 == 'min':
            idx3 = np.argmin(v3)+1
        else:
            idx3 = np.argmax(v3)+1

        if obj4 == 'min':
            idx4 = np.argmin(v4)+1
        else:
            idx4 = np.argmax(v4)+1

        picked = np.array([idx1, idx2, idx3, idx4])
        nc = sp.stats.mode(picked)
        nc = nc[0][0]

        return nc, picked

    # modeling
    def Pred_modeling(self, tMdl, x_train, y_train, x_test, y_test, nComp = 2):
        model_list = ['Linear', 'Ridge', 'PCAreg', 'PLSreg']
        cMdl = model_list[tMdl]

        if cMdl == 'Linear':
            mdl = LinearRegression()

        elif cMdl == 'Ridge':
            mdl = Ridge()

        elif cMdl == 'PCAreg':
            pca_mdl = PCA(n_components = nComp)
            x_train = pca_mdl.fit_transform(x_train)
            x_test = pca_mdl.transform(x_test)
            mdl = LinearRegression()

        elif cMdl == 'PLSreg':
            mdl = PLSRegression(n_components = nComp)

        mdl.fit(x_train, y_train)
        y_pred = mdl.predict(x_test)       
        
        return y_test, y_pred
    
def parse_args():
    opt = argparse.ArgumentParser(description="==== simple CPM for fMRI fc data  ====\n\nexample usage : python Cpm_auto.py -tBehav 0 -tfMRI 2 -tNet 1 -tScale 1 -tFs 0 -tFs_alpha 0.05 -tMdl 2 -opt_tComp 0 -opt_nComp 10", formatter_class=RawTextHelpFormatter)
    
    opt.add_argument('-tBehav', dest='tBehav', type=int, 
                     help=': behavior measure (0-fi, 1-wm, 2-viq1, 3-viq2, 4-ps, 5-lm, 6-dd), default %i'%1, default=1)
    opt.add_argument('-tfMRI', dest='tfMRI', type=int, 
                     help=': fMRI fc sate (0-rest, 1-wm, 2-lang, 3-gamble, 4-social), default %i'%1, default=1)
    opt.add_argument('-tNet', dest='tNet', type=int, 
                     help=': fc network (0-all, 1-mfn, 2-fpn, 3-dmn, 4-subcor, 5-mo, 6-vI, 7-vII, 8-vA), default %i'%0, default=0)
    opt.add_argument('-tScale', dest='tScale', type=int, 
                     help=': scaling method (0-none, 1-std, 2-robust, 3-minmax), default %i'%1, default=1)
    opt.add_argument('-tFs', dest='tFs', type=int, 
                     help=': feature selection method (0-none, 1-f_reg), default %i'%1, default=1)
    opt.add_argument('-tFs_alpha',dest='tFs_alpha', type=float, 
                     help=': feature selection threshold, default %f'%0.05, default=0.05)
    opt.add_argument('-tMdl', dest='tMdl', type=int, 
                     help=': model (0-Linear, 1-Ridge, 2-PCAreg, 3-PLSreg), default %i'%1, default=1)
    opt.add_argument('-opt_tComp', dest='opt_tComp', type=int, 
                     help=': target PC for PCR/PLSR, default %i'%0, default=0)
    opt.add_argument('-opt_nComp',dest='opt_nComp', type=int, 
                     help=': max PC for PCR/PLSR, default %i'%10, default=10)
    args = opt.parse_args()
    return args    

# main
def main():
    args = parse_args()
    print(args)
    
    print("\n====== simple CPM for fMRI fc data  ======\n")

    # set path
    global proj_dri, hcp_dir, results_dir
    proj_dir=os.getcwd()
    hcp_dir=(proj_dir +  '/hcp_data')
    results_dir=(proj_dir + '/results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # set variables
    global behav_list, fMRI_list, fs_list, scale_list, net_list, net_label, model_list
    behav_list = ['fi', 'wm', 'viq1', 'viq2', 'ps', 'lm', 'dd']
    fMRI_list = ['rest', 'wm', 'lang', 'gamble', 'social']
    fs_list = ['none', 'f-reg']
    fs_alpha = 0.05
    scale_list = ['none', 'std', 'robust', 'minmax']
    net_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    net_label = ['All', 'Medial frontal', 'Fronto Parietal',
                'Default Mode', 'Subcortical-cereb',
                'Motor', 'Visual I', 'Visual II',
                 'Visual Association']
    model_list = ['Linear', 'Ridge', 'PCAreg', 'PLSreg']    
    
    # main class
    cpm = Cpm_Auto( tBehav = args.tBehav
                  , tfMRI = args.tfMRI
                  , tNet = args.tNet
                  , tScale = args.tScale
                  , tFs = args.tFs
                  , tFs_alpha = args.tFs_alpha
                  , tMdl = args.tMdl
                  , opt_tComp = args.opt_tComp
                  , opt_nComp = args.opt_nComp)


    # load data
    data, label, subj_idx = cpm.load_data(args.tBehav, args.tfMRI, args.tNet)
    print('1. load data - behav: %s , state: %s, network: %s, sn/edge: %s ...'
    	% (behav_list[args.tBehav], fMRI_list[args.tfMRI], net_label[args.tNet], data.shape))

    # args.tBehav
    
    # save
    x_behav = [];
    x_fMRI = [];
    x_net = [];
    x_scale = [];
    x_fs = [];
    x_fsAlpha =[];
    x_model = [];
    x_nComp = [];
    x_r = [];
    x_r_pval = [];
    x_rho = [];
    x_rho_pval = [];
    x_tau = [];
    x_tau_pval = [];
    x_r2 = [];
    x_rmse = [];
    x_mae = [];
    x_mape = [];
    
    x_target = [];
    x_y_test = [];
    x_y_pred = [];    
    x_tComp = [];
    
    
    # running cpm loop
    startTime = time.time()
    print('2. data scaling - method: %s ...' % (scale_list[args.tScale]))    
    print('3. feature selection - method: %s, a: %.03f ...' % (fs_list[args.tFs], args.tFs_alpha)) 
    print('4. running cpm at %s ... model: %s, nComp: %d ...' % (time.ctime(), model_list[args.tMdl], args.opt_tComp)) 
    for idx, this_sn in enumerate(subj_idx):
        tr_sn = deepcopy(subj_idx)
        tr_sn.remove(this_sn)
        te_sn = this_sn
        
        x_train = data.loc[data['sn'] != this_sn, :].drop(['sn'], axis = 1)
        y_train = label.loc[label['sn'] != this_sn, :].drop(['sn'], axis = 1)
        x_test = data.loc[data['sn'] == this_sn, :].drop(['sn'], axis = 1)
        y_test = label.loc[label['sn'] == this_sn, :].drop(['sn'], axis = 1)
        # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        
        # scaling
        x_train, x_test = cpm.scaling(tScale = args.tScale, x1 = x_train, x2 = x_test)
        
        # feature selection
        x_train, y_train, x_test, y_test = cpm.feature_selecting(tFs = args.tFs, x1 = x_train, y1 = y_train, 
                                                             x2 = x_test, y2 = y_test, tFs_alpha = args.tFs_alpha)

        # model optimization
        if args.tMdl > 1: #['Linear', 'Ridge', 'PCAreg', 'PLSreg']    
        	if args.opt_tComp == 0:
	            r2, rmse, mae, mape = cpm.pc_optimization(tMdl = args.tMdl, x = x_train, y = y_train, 
	                                                  n_comp = args.opt_nComp, cv = 10)
	            tComp, picked = cpm.pick_nComp(v1=r2, v2=rmse, v3=mae, v4=mape, 
	            	label1='r2', label2='rmse', label3='mae', label4='mape', 
	            	obj1='max', obj2='min', obj3='min', obj4='min')
	        elif args.opt_tComp > 0:
	        	tComp = args.opt_tComp
        elif args.tMdl < 2: #['Linear', 'Ridge', 'PCAreg', 'PLSreg']    
            tComp = args.opt_tComp
        
        # predictive modeling
        y_test, y_pred = cpm.Pred_modeling(args.tMdl, x_train, y_train, x_test, y_test, nComp = tComp)
        
        # saving
        x_target.append(this_sn)
        x_y_test.append(y_test)
        x_y_pred.append(y_pred)
        x_tComp.append(tComp)

    endTime = time.time()
    durTime = endTime - startTime
    print('5. done at %s, it took %.02f sec ... now saving ...' % (time.ctime(), durTime))
   
    # results
    target = np.array(x_target)
    test = np.array(x_y_test).flatten()
    pred = np.array(x_y_pred).flatten()
    tmp_nComp = sp.stats.mode(np.array(x_tComp))
    nComp_fnl = tmp_nComp[0]
    
    tmp_r = pg.corr(test, pred, method="pearson")
    tmp_rho = pg.corr(test, pred, method="spearman")
    tmp_tau = pg.corr(test, pred, method="kendall")
    r2_fnl = r2_score(test, pred)
    rmse_fnl = np.sqrt(mean_squared_error(test, pred))
    mae_fnl = mean_absolute_error(test, pred)
    mape_fnl = mean_absolute_percentage_error(test, pred)*100

    x_r.append(tmp_r['r'][0])
    x_r_pval.append(tmp_r['p-val'][0])
    x_rho.append(tmp_rho['r'][0])
    x_rho_pval.append(tmp_rho['p-val'][0])
    x_tau.append(tmp_tau['r'][0])
    x_tau_pval.append(tmp_tau['p-val'][0])
    x_r2.append(r2_fnl)
    x_rmse.append(rmse_fnl)
    x_mae.append(mae_fnl)
    x_mape.append(mape_fnl)
    
    x_behav.append(behav_list[args.tBehav])
    x_fMRI.append(fMRI_list[args.tfMRI])
    x_net.append(net_label[args.tNet])
    x_scale.append(scale_list[args.tScale])
    x_fs.append(fs_list[args.tFs])
    x_fsAlpha.append(args.tFs_alpha)
    x_model.append(model_list[args.tMdl])
    x_nComp.append(nComp_fnl)
    
    now = datetime.datetime.now()
    curr_time = now.strftime('%Y-%m-%d, %H:%M')
    
    results = pd.DataFrame({'date'     : curr_time,
                            'behav'    : x_behav,
                            'state'    : x_fMRI,
                            'network'  : x_net,
                            'scaler'   : x_scale,
                            'fs_method': x_fs,                            
                            'fs_alpha' : x_fsAlpha,                            
                            'model'    : x_model,
                            'nComp'    : x_nComp[0],
                            'r'        : x_r,
                            'r_pval'   : x_r_pval,
                            'rho'      : x_rho,
                            'rho_pval' : x_rho_pval,
                            'tau'      : x_tau,
                            'tau_pval' : x_tau_pval,                            
                            'r2'       : x_r2,
                            'rmse'     : x_rmse,
                            'mae'      : x_mae,
                            'mape'     : x_mape})
    
     
    fileName = 'results.csv'
    filePath = (results_dir + '/' + fileName)
    if os.path.isfile(filePath):
        results.round(3).to_csv(filePath, mode = 'a', header = False, index = False)
    elif not os.path.isfile(filePath):
        results.round(3).to_csv(filePath, mode = 'w', header = True, index = False)

if __name__=='__main__':
    main()