import xgboost as xgb
# load the python modules we will use
import uproot # for data loading


import pandas as pd
import matplotlib as mpl # for plotting
import matplotlib.pyplot as plt # common shorthand
from mpl_toolkits.mplot3d import Axes3D
import mplhep # style of plots
import numpy as np
import argparse
import pickle as pkl

mpl.style.use(mplhep.style.ROOT) # set the plot style

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


def main(args):

    with uproot.open(f"m20_g_0.00352_ntuple.root") as f:
        data = f['Features'].arrays(library='pd')
        num_data = 50000 #len(signal)

    data = pd.concat([data], ignore_index=True)

    print(len(data))
    #want everything except isSignal and label and xs and ys and zs
    features = data.columns
    features = features.drop(['isSignal'])
    data = data.drop_duplicates(features)

    #getting some data to test
    sampled_data = data

    nbins=100
    #loading the model
    with open(f'./testw.pkl', 'rb') as model_file:
        gbm = pkl.load(model_file)


    test = xgb.DMatrix(data=sampled_data[features],
                    missing=-999.0,feature_names=features, enable_categorical=True)

    predictions = gbm.predict(test)

    sig_hist, _ = np.histogram(predictions,bins=np.linspace(0,1,nbins),
            density=True);
    cut = 0.99
    passes=0
    all=0
    for i, j in enumerate(predictions):
        all+=1
        if j > cut:
            passes +=1
    print(passes/all)
    #setting up histogram
    bins = np.linspace(0,1,nbins)
    bin_centers = (np.linspace(0, 1, nbins)[:-1] + np.linspace(0, 1, nbins)[1:]) / 2

    bin_width = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure()

    # Plot signal histogram with error bars
    plt.bar(bin_centers, sig_hist, width=bin_width, edgecolor='midnightblue', alpha=0.2, label=f'50MeV, inclusive',  error_kw=dict(ecolor='black', capsize=3))

    plt.xlabel('Prediction from BDT', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.axvline(cut, color='y', linestyle='--')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(f'test_m50.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)
