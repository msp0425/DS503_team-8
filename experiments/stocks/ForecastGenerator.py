import numpy as np
import torch
import pandas as pd
import gpytorch
import argparse
import datetime
import warnings
import stock_plot 

## VOLT-MAIN 모든 module 접근을 위해서 설정
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


from voltron.data import make_ticker_list, GetStockHistory
import sys
from GenerateMultiMeanPreds import GenerateStockPredictions, GenerateBasicPredictions
from gpytorch.utils.warnings import NumericalWarning
warnings.simplefilter("ignore", NumericalWarning)

def main(args):
    
    print(os.getcwd())
    ticker_file = args.ticker_fname + ".txt"
    ticker_file = os.path.join('voltron','data',ticker_file) ## directory 오류 수정
    tckr_list = make_ticker_list(ticker_file)
    
    if args.end_date.lower() == "none":
        end_date = datetime.date.today()
    else:
            end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
    
    tckr_list = tckr_list[:3]
    for tckr in tckr_list:
        try:    
            data = GetStockHistory(tckr, end_date= '2024-03-25',history=args.ntrain + args.lookback) ## end-date 임시 설정 상태
            print("data downloaded: ", tckr)
            if args.kernel.lower() == 'volt':
                dat, save_samples =GenerateStockPredictions(tckr, data, forecast_horizon=args.forecast_horizon,
                                        train_iters=args.train_iters, 
                                        nsample=args.nsample, mean=args.mean,
                                        ntrain=args.ntrain, save=args.save,
                                        ntimes=args.ntimes)
            else:
                GenerateBasicPredictions(tckr, data, forecast_horizon=args.forecast_horizon,
                                        kernel_name=args.kernel, mean_name=args.mean,
                                        k=args.k, train_iters=args.train_iters, 
                                            nsample=args.nsample, ntimes=args.ntimes,
                                                    ntrain=args.ntrain, save=args.save)
        except:
            print("FAILED ", tckr)

    
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
        "--ntimes",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=100,
    )
    parser.add_argument(
        '--kernel',
        type=str,
        default="volt",
    )
    parser.add_argument(
        '--mean',
        type=str,
        default="ewma",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--printing",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=300,
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