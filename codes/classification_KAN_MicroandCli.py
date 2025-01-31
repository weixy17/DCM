from kan import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import numpy as np
import scipy.io as scio
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
dtype = torch.get_default_dtype()
data=scio.loadmat("data.mat")
X=torch.from_numpy(data['quantitative_indexes']).type(dtype)
Y=torch.from_numpy(data['labels'].squeeze()).type(dtype)



###########13个指标
run_id='457Dec27-1642'
para1=4
para2=12
para3=4
lr=0.01

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

    
aucs=np.zeros([4])
accs=np.zeros([4])
sens=np.zeros([4])
spes=np.zeros([4])
fi=torch.zeros([1,X.shape[1]])
fprs, tprs,value = [], [], []
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
        X_train=torch.from_numpy(sc.transform(X_train)).to(device)
        X_test=torch.from_numpy(sc.transform(X_test)).to(device)
        model = KAN(width=[X.shape[1],para1,para3,2], grid=3, k=3, seed=2024, device=device)
        
        model_list=test_findfile('./weights/','.pth',str(run_id)+'_'+str(para1)+'_'+str(para2)+'_'+str(para3)+'_'+str(lr)+'_'+str(count)+'_')
        try:
            state_dict=torch.load(model_list[-1], map_location={'cuda:1': 'cuda:1'})
        except:
            state_dict=torch.load(model_list[-1], map_location={'cuda:0': 'cuda:1'})
        model.load_state_dict(state_dict)
        pred_decision=model.forward(dataset['test_input'])
        model.attribute()
        fi=fi+model.feature_score.detach().cpu()/model.feature_score.detach().cpu().sum()
        pred_decision=pred_decision.cpu().detach().numpy()
        pred=np.argmax(pred_decision,1)
        true=Y_test.numpy()
        aucs[count]=auc(pred_decision[:,1],true)
        accs[count]=acc(pred,true)
        sens[count]=sen(pred,true)
        spes[count]=spe(pred,true)
        print('count={} test_acc :{:.4f},test_sen:{:.4f},test_spe:{:.4f},test_auc:{:.4f}'.format(count,accs[count],sens[count],spes[count],aucs[count]))
    print('para1={},para2={},para3={},lr={}'.format(para1,para2,para3,lr))
    print('test_accs :{:.4f},test_sens:{:.4f},test_spes:{:.4f},test_aucs:{:.4f}'.format(np.mean(accs),np.mean(sens),np.mean(spes),np.mean(aucs)))
    lower, upper = stats.t.interval(0.95, len(aucs) - 1, loc=np.mean(aucs), scale=stats.sem(aucs))
    print(lower)
    print(upper)
