import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

import torch
from voltron.train_utils import LearnGPCV, TrainVolModel, TrainVoltMagpieModel, TrainBasicModel
from voltron.rollout_utils import GeneratePrediction, Rollouts
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
palette = ["#1b4079", "#C6DDF0", "#50723C", "#B9E28C", "#8C2155", "#AF7595", "#E6480F", "#FA9500"]
sns.set(palette = palette, font_scale=2.0, style="white", rc={"lines.linewidth": 3.0})


def stock_plot(ticker, dat, date, save_samples,last_day,train_x, train_y, test_x, 
                        forecast_horizon=20,
                        ntrain=400, mean='ewma', kernel='volt', 
                             save=False, k=300, ntimes=-1,  ):
    

    ntest = forecast_horizon
    dt = 1./252
    
    train_x = train_x.cpu()
    train_y = train_y.cpu()
    test_x = test_x.cpu()
    save_samples = save_samples.cpu()
    
    try:
        test_y = torch.FloatTensor(dat.Close[last_day.item():last_day.item()+forecast_horizon].to_numpy())
        test_ans = True
    except:
        test_ans = False
        
    # model_name = kernel + "_" + mean + str(k) + "_"
    # par_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    # savepath = os.path.join(par_dir ,'saved-outputs' , ticker , model_name + date) +'.pt'
    
    # save_samples = torch.load(savepath)
    
    plt.figure(figsize= (16,10))
    plt.plot(train_x, train_y[1:].log(), color=palette[0], alpha =0.8, label='Data')

    plt.plot(test_x, save_samples[:1000,:].T, color= palette[2], alpha = 0.3, label= 'Predictions')
    
    if test_ans:
        plt.plot(test_x, test_y.log(), color=palette[-2], label='Answer')
    
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.legend( labels = ['Data', 'Answer', 'Predictions'],loc="lower left")
    
    plt.show()
    