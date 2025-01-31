from sklearn.neural_network import *
import numpy as np
import pdb
import scipy.io as scio
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

data=scio.loadmat("data.mat")
X=data['quantitative_indexes']
Y=data['labels'].squeeze()

def acc(predict,true):
    return np.sum(1*(predict==true))/true.shape[0]
def sen(predict,true):
    return (np.logical_and(predict==1,true==1)).sum()/(true==1).sum()
def spe(predict,true):
    return (np.logical_and(predict==0,true==0)).sum()/(true==0).sum()
def auc(predict,true):
    return roc_auc_score(true,predict,multi_class='ovo')
def test_findfile(directory, fileType, file_prefix):
    fileList = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(fileType) and fileName.startswith(file_prefix):
                fileList.append(os.path.join(root, fileName))
    return fileList
run_id='194Jan13-1804'
para1=40
para2=7
para3=0.001

aucs=np.zeros([4])
accs=np.zeros([4])
sens=np.zeros([4])
spes=np.zeros([4])
with open('test_idx.txt','r') as f:
    for count in range (4):
        test_index_loo=f.readline()
        test_index_loo=[int(i) for i in test_index_loo.split('\n')[0].split(',')]
        train_index_loo=[i for i in range (Y.shape[0]) if i not in test_index_loo]
        X_train=X[train_index_loo,:]
        Y_train=Y[train_index_loo]
        X_test=X[test_index_loo,:]
        Y_test=Y[test_index_loo]
        sc=MinMaxScaler()
        sc.fit(X_train)
        X_train=sc.transform(X_train)
        X_test=sc.transform(X_test)
        model_list=test_findfile('./weights/','.pickle',str(run_id)+'_'+str(para1)+'_'+str(para2)+'_'+str(para3)+'_'+str(count))
        fmodel = open(model_list[-1],'rb')
        model = pickle.load(fmodel)
        fmodel.close()
        pred=model.predict(X_test)
        true=Y_test
        aucs[count]=auc(model.predict_proba(X_test)[:,1],true)
        pred_prob=model.predict_proba(X_test)[:,1]
        pred_decision=model.predict_proba(X_test)
        accs[count]=acc(pred,true)
        sens[count]=sen(pred,true)
        spes[count]=spe(pred,true)
print('hidden_layer={:.2f},n_estimators={},lr={}'.format(para1,para2,para3))
print('acc :{:.4f},sen:{:.4f},spe:{:.4f},auc:{:.4f}'.format(np.mean(accs),np.mean(sens),np.mean(spes),np.mean(aucs)))
lower, upper = stats.t.interval(0.95, len(aucs) - 1, loc=np.mean(aucs), scale=stats.sem(aucs))
print(lower)
print(upper)
pdb.set_trace()
