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

## VOLT-MAIN 모든 module 접근을 위해서 설정
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


from voltron.data import make_ticker_list, GetStockHistory
import sys
from GenerateMultiMeanPreds import GenerateStockPredictions, GenerateStockPredictions_single,GenerateBasicPredictions
from gpytorch.utils.warnings import NumericalWarning
warnings.simplefilter("ignore", NumericalWarning)


import random



def main(args):
    
    ticker_file = args.ticker_fname + ".txt"
    ticker_file = os.path.join('voltron','data',ticker_file) ## directory 오류 수정
    tckr_list = make_ticker_list(ticker_file)
    
    if args.end_date.lower() == "none":
        end_date = datetime.date.today()
    else:
            end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
    
    tckr = args.tckr
     
    data = GetStockHistory(tckr, end_date= '2024-03-25',history=args.ntrain + args.lookback) ## end-date 임시 설정 상태
    
    print("data downloaded: ", tckr)
    

    dat, save_samples, date, last_day, train_x, train_y, test_x, pred_mean, pred_cov = GenerateStockPredictions_single(tckr, data, forecast_horizon=args.forecast_horizon,
                            train_iters=args.train_iters, 
                            nsample=args.nsample, mean=args.mean,
                            ntrain=args.ntrain, save=args.save,
                            ntimes=args.ntimes, data_num = 0) ## Data_num으로 몇번째 데이터로 학습할지 결정



    cal_NLL(tckr, dat, forecast_horizon=args.forecast_horizon, mean=args.mean,
                                        ntrain=args.ntrain, save=args.save,
                                        ntimes=args.ntimes, date=date, save_samples = save_samples, last_day = last_day, train_x = train_x, train_y = train_y,
                                        test_x = test_x, pred_mean= pred_mean, pred_cov= pred_cov, kernel = args.kernel.lower())    
    # # print('-----------Prediction End-----------')
    # stock_plot(tckr, dat, forecast_horizon=args.forecast_horizon, mean=args.mean,
    #                                 ntrain=args.ntrain, save=args.save,
    #                                 ntimes=args.ntimes, date=date, save_samples = save_samples, last_day = last_day, train_x = train_x, train_y = train_y,
    #                                 test_x = test_x)
    
    
    
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
        '--mean',
        type=str,
        default="constant",
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