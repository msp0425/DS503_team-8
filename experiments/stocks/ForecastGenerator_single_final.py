import numpy as np
import torch
import pandas as pd
import gpytorch
import argparse
import datetime
import warnings
from stock_plot import stock_plot 
from cal_NLL import *
import warnings
warnings.filterwarnings('ignore')
from GBI import *
## VOLT-MAIN 모든 module 접근을 위해서 설정
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import pickle

from voltron.data import make_ticker_list, GetStockHistory
import sys
from GenerateMultiMeanPreds import GenerateStockPredictions, GenerateStockPredictions_single,GenerateBasicPredictions
from gpytorch.utils.warnings import NumericalWarning
warnings.simplefilter("ignore", NumericalWarning)


import random



def main(args):
    

    if args.gbi == False:
                  
        ticker_file = args.ticker_fname + ".txt"
        ticker_file = os.path.join('voltron','data',ticker_file) ## directory 오류 수정
        tckr_list = make_ticker_list(ticker_file)
        
        if args.end_date.lower() == "none":
            end_date = datetime.date.today()
        else:
                end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
        
        tckr = args.tckr
        tckr2 = args.tckr2
        tckr3 = args.tckr3
        data = GetStockHistory(tckr, end_date= '2024-03-25',history=args.ntrain + args.lookback) ## end-date 임시 설정 상태
        print("data downloaded: ", tckr)
        data2 = GetStockHistory(tckr2, end_date= '2024-03-25',history=args.ntrain + args.lookback)
        print("data downloaded: ", tckr2)
        data3 = GetStockHistory(tckr3, end_date= '2024-03-25',history=args.ntrain + args.lookback)
        print("data downloaded: ", tckr3)

        

        dat, save_samples, date, last_day, train_x, train_y, test_x, pred_mean, pred_cov = GenerateStockPredictions_single(tckr, data, forecast_horizon=args.forecast_horizon,
                                train_iters=args.train_iters, 
                                nsample=args.nsample, mean=args.mean,
                                ntrain=args.ntrain, save=args.save,
                                ntimes=args.ntimes, data_num = 0, bm = args.bm) ## Data_num으로 몇번째 데이터로 학습할지 결정
        test_y1 = cal_test_y(tckr, dat, forecast_horizon=args.forecast_horizon, mean=args.mean,
                                        ntrain=args.ntrain, save=args.save,
                                        ntimes=args.ntimes, date=date, save_samples = save_samples, last_day = last_day, train_x = train_x, train_y = train_y,
                                        test_x = test_x, pred_mean= pred_mean, pred_cov= pred_cov, kernel = args.kernel.lower()) 
        nll = cal_NLL(tckr, dat, forecast_horizon=args.forecast_horizon, mean=args.mean,
                                        ntrain=args.ntrain, save=args.save,
                                        ntimes=args.ntimes, date=date, save_samples = save_samples, last_day = last_day, train_x = train_x, train_y = train_y,
                                        test_x = test_x, pred_mean= pred_mean, pred_cov= pred_cov, kernel = args.kernel.lower())    

        save_folder = os.path.join('results_nll_________', str(args.tckr), str(args.bm))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        results_file = os.path.join(save_folder, f"results.csv")
        with open(results_file, 'a') as f:
            f.write('{},{}\n'.format('Nll:', nll))        



        r_upper = torch.max((torch.exp(save_samples)[:,1:] / torch.exp(save_samples)[:,:-1])-1, dim=0)[0].unsqueeze(dim=1)
        r_lower = torch.min((torch.exp(save_samples)[:,1:] / torch.exp(save_samples)[:,:-1])-1, dim=0)[0].unsqueeze(dim=1)
        r = (test_y1[1:]/test_y1[:-1]-1).unsqueeze(dim=1)


        dat2, save_samples2, date2, last_day2, train_x2, train_y2, test_x2, pred_mean2, pred_cov2 = GenerateStockPredictions_single(tckr2, data2, forecast_horizon=args.forecast_horizon,
                                train_iters=args.train_iters, 
                                nsample=args.nsample, mean=args.mean,
                                ntrain=args.ntrain, save=args.save,
                                ntimes=args.ntimes, data_num = 0, bm = args.bm) ## Data_num으로 몇번째 데이터로 학습할지 결정
        test_y2 = cal_test_y(tckr2, dat2, forecast_horizon=args.forecast_horizon, mean=args.mean,
                                        ntrain=args.ntrain, save=args.save,
                                        ntimes=args.ntimes, date=date2, save_samples = save_samples2, last_day = last_day2, train_x = train_x2, train_y = train_y2,
                                        test_x = test_x2, pred_mean= pred_mean2, pred_cov= pred_cov2, kernel = args.kernel.lower()) 
        nll2 = cal_NLL(tckr2, dat2, forecast_horizon=args.forecast_horizon, mean=args.mean,
                                        ntrain=args.ntrain, save=args.save,
                                        ntimes=args.ntimes, date=date2, save_samples = save_samples2, last_day = last_day2, train_x = train_x2, train_y = train_y2,
                                        test_x = test_x2, pred_mean= pred_mean2, pred_cov= pred_cov2, kernel = args.kernel.lower())    

        save_folder = os.path.join('results_nll_________', str(args.tckr2), str(args.bm))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        results_file = os.path.join(save_folder, f"results.csv")
        with open(results_file, 'a') as f:
            f.write('{},{}\n'.format('Nll:', nll2))   
        r_upper = torch.cat([r_upper, torch.max((torch.exp(save_samples2)[:,1:] / torch.exp(save_samples2)[:,:-1])-1, dim=0)[0].unsqueeze(dim=1)], dim=1)
        r_lower = torch.cat([r_lower, torch.min((torch.exp(save_samples2)[:,1:] / torch.exp(save_samples2)[:,:-1])-1, dim=0)[0].unsqueeze(dim=1)], dim=1)
        r = torch.cat([r,(test_y2[1:]/test_y2[:-1]-1).unsqueeze(dim=1)], dim=1)


        dat3, save_samples3, date3, last_day3, train_x3, train_y3, test_x3, pred_mean3, pred_cov3 = GenerateStockPredictions_single(tckr3, data3, forecast_horizon=args.forecast_horizon,
                                train_iters=args.train_iters, 
                                nsample=args.nsample, mean=args.mean,
                                ntrain=args.ntrain, save=args.save,
                                ntimes=args.ntimes, data_num = 0, bm = args.bm) ## Data_num으로 몇번째 데이터로 학습할지 결정
        
        test_y3 = cal_test_y(tckr3, dat3, forecast_horizon=args.forecast_horizon, mean=args.mean,
                                        ntrain=args.ntrain, save=args.save,
                                        ntimes=args.ntimes, date=date3, save_samples = save_samples3, last_day = last_day3, train_x = train_x3, train_y = train_y3,
                                        test_x = test_x3, pred_mean= pred_mean3, pred_cov= pred_cov3, kernel = args.kernel.lower()) 
        r = torch.cat([r,(test_y3[1:]/test_y3[:-1]-1).unsqueeze(dim=1)], dim=1)

        nll3 = cal_NLL(tckr3, dat3, forecast_horizon=args.forecast_horizon, mean=args.mean,
                                        ntrain=args.ntrain, save=args.save,
                                        ntimes=args.ntimes, date=date3, save_samples = save_samples3, last_day = last_day3, train_x = train_x3, train_y = train_y3,
                                        test_x = test_x3, pred_mean= pred_mean3, pred_cov= pred_cov3, kernel = args.kernel.lower())    

        save_folder = os.path.join('results_nll_________', str(args.tckr3), str(args.bm))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        results_file = os.path.join(save_folder, f"results.csv")
        with open(results_file, 'a') as f:
            f.write('{},{}\n'.format('Nll:', nll3))          
            
        r_upper = torch.cat([r_upper, torch.max((torch.exp(save_samples3)[:,1:] / torch.exp(save_samples3)[:,:-1])-1, dim=0)[0].unsqueeze(dim=1)], dim=1)
        r_lower = torch.cat([r_lower, torch.min((torch.exp(save_samples3)[:,1:] / torch.exp(save_samples3)[:,:-1])-1, dim=0)[0].unsqueeze(dim=1)], dim=1)
        r_mean = (r_upper + r_lower)/2
        r_upper = r_upper.cpu().detach()
        r_lower = r_lower.cpu().detach()
        r_mean = r_mean.cpu().detach()

        samples_filename_write = f"sample.pkl"
        with open(samples_filename_write, 'wb') as filehandle:
            pickle.dump([r_upper, r_lower, r_mean, r], filehandle)

    else:
        samples_filename_read = f"sample.pkl"
        with open(samples_filename_read, 'rb') as filehandle:
            r_upper, r_lower, r_mean, r = pickle.load(filehandle)

    a,b,c,d,e = create_cvxpy_problem(r_upper, r_lower, r_mean)
    Z = [c,d,e]

    goal = get_objective(r, Z)
    print(goal)
    save_folder = os.path.join('results_________', str(args.bm))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    results_file = os.path.join(save_folder, f"results.csv")
    with open(results_file, 'a') as f:
        f.write('{},{}\n'.format('Goal achievement:', goal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ticker_fname",
        type=str,
        # default='test_tickers',
        default='nasdaq100',
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--tckr",
        type=str,
        default='AAPL',
    )
    parser.add_argument(
        "--tckr2",
        type=str,
        default='NVDA',
    )
    parser.add_argument(
        "--tckr3",
        type=str,
        default='TSLA',
    )
    parser.add_argument(
        "--ntimes",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=20,
    )
    parser.add_argument(
        '--kernel',
        type=str,
        default="volt",
    )
    parser.add_argument(
        '--gbi',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--mean',
        type=str,
        default="constant",
    )
    parser.add_argument(
        '--bm',
        type=str,
        default="bm3",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--printing",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=2000,
        # default=300,
    )
    parser.add_argument(
        "--end_date",
        default="none",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    main(args)