import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

import torch
from torch.nn import NLLLoss
from voltron.train_utils import LearnGPCV, TrainVolModel, TrainVoltMagpieModel, TrainBasicModel
from voltron.rollout_utils import GeneratePrediction, Rollouts
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np





# def NLL(targets, outputs):
#     dist = torch.distributions.Normal(outputs[:, 0], outputs[:, 1])
#     return -dist.log_prob(targets).sum()


def multivariate_gaussian_nll(mean, covariance, data):
    d = mean.size(0)
    c = covariance.shape[0]
    n = data.size(0)
    mean = mean.squeeze()
    # Reshaping mean vector

    
    # Reshaping covariance matrix
    # covariance = covariance.reshape(1, c,c)
    
    # Calculate NLL

    term2 = n * torch.logdet(2 * torch.tensor(np.pi) * covariance) / 2
    term3 = torch.matmul((data - mean).unsqueeze(dim=1), torch.inverse(covariance))
    term3 = torch.matmul(term3, (data - mean).unsqueeze(dim=2)).squeeze()
    nll = 0.5 * (term2 + term3)
    
    return torch.mean(nll)


def cal_test_y(ticker, dat, date, save_samples,last_day,train_x, train_y, test_x,pred_mean, pred_cov,
                        forecast_horizon=20,
                        ntrain=400, mean='ewma', kernel='volt', 
                             save=False, k=300, ntimes=-1,):
    
    test_y = torch.FloatTensor(dat.Close[last_day.item():last_day.item()+forecast_horizon].to_numpy())
    return test_y


def cal_NLL(ticker, dat, date, save_samples,last_day,train_x, train_y, test_x,pred_mean, pred_cov,
                        forecast_horizon=20,
                        ntrain=400, mean='ewma', kernel='volt', 
                             save=False, k=300, ntimes=-1,):

    
    try:
        test_y = torch.FloatTensor(dat.Close[last_day.item():last_day.item()+forecast_horizon].to_numpy())
        test_ans = True
    except:
        test_ans = False

    if kernel == 'volt':    
        # var = torch.diagonal(pred_cov,dim1=1, dim2=2).reshape(1000,100,-1)        
        # mean_var_mat = torch.cat([var, pred_mean], dim=2)
        # nll = 0
        # for i in range(1000):
        #     nll += NLL(test_y.log()[75:], mean_var_mat[i,75:,:].to('cpu'))
        # nll = nll/1000

        # Calculating NLL
        nll = multivariate_gaussian_nll(pred_mean.to('cpu'), pred_cov.to('cpu'), test_y.log())
        
    else:
        var = pred_cov            
        mean_var_mat = torch.cat([var, pred_mean], dim=1)
        nll = NLL(test_y.log()[75:], mean_var_mat[75:,:].to('cpu'))
    

    print(nll.item())

    return nll.item()
    