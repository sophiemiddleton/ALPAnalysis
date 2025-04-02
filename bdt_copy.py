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

nbins = 50
a=2
#load the data
def main(args):
    sig_file = uproot.open("signal_ntuple.root")
    sig_features = sig_file['Features']
    signal = sig_features.arrays(library='pd')

    bkgd_file = uproot.open("PN_ntuple.root")
    bkgd_features = bkgd_file['Features']
    bkgd = bkgd_features.arrays(library='pd')

    print("nSig",len(signal))
    print("nBkg",len(bkgd))

    #assigning labels
    signal['Label'] = 'signal'
    bkgd['Label'] = 'bkgd'

    signal['Label'] = signal.Label.astype('category')
    bkgd['Label'] = bkgd.Label.astype('category')

    #print(signal)

    data = pd.concat([bkgd, signal], ignore_index=True)

    data['Label'] = pd.Categorical(data['Label'])

    #want everything except isSignal and label and xs and ys and zs
    features = data.columns[:-1]
    features = features.drop(["isSignal"])


    data = pd.concat([bkgd, signal], ignore_index=True)

    data['Label'] = pd.Categorical(data['Label'])

    X = data
    y = data['Label']


    signal_data = data[data['Label'] == 'signal']
    background_data = data[data['Label'] == 'bkgd']

    param = {}

    # Booster parameters
    param['eta']              = 0.1 # learning rate
    param['max_depth']        = 10  # maximum depth of a tree
    param['subsample']        = 0.8# fraction of events to train tree on
    param['colsample_bytree'] = 1.0 # fraction of features to train tree on

    # Learning task parameters
    param['objective']   = 'binary:logistic' # objective function
    param['eval_metric'] = ['error']        # evaluation metric for cross validation
    #param['sampling_method'] = 'gradient_based'

    num_trees = 100  # number of trees to make

    sig_hists = []
    bkgd_hists = []

    samplesize = 100000
    for fold in range(1):

        #sampling 10000 events from each type
        sampled_signal = signal_data.sample(n=samplesize, random_state=42 + fold, replace=False).reset_index(drop=True)
        sampled_background = background_data.sample(n=samplesize, random_state=42 + fold, replace=False).reset_index(drop=True)

        # Combine the samples
        sampled_data = pd.concat([sampled_signal, sampled_background]).reset_index(drop=True)

        # Split into training and testing sets, preserving the 1:1 ratio
        train_set, test_set = train_test_split(sampled_data, test_size=0.5, stratify=sampled_data['Label'], random_state=42)

        print("test set length",len(test_set))

        train = xgb.DMatrix(data=train_set[features],label=train_set.Label.cat.codes,
                        missing=-999.0,feature_names=features)
        test = xgb.DMatrix(data=test_set[features],label=test_set.Label.cat.codes,
                    missing=-999.0,feature_names=features)



        booster = xgb.train(param,train,num_boost_round=num_trees)
        output = open("testw.pkl", 'wb')
        pkl.dump(booster, output)
        print("test",booster.eval(test))

        predictions = booster.predict(test)
        # plot all predictions (both signal and background)
        ax = xgb.plot_importance(booster,grid=False);
        plt.savefig("example_20.pdf")


        # store histograms of signal and background separately
        sig_hist, _ = np.histogram(predictions[test.get_label().astype(bool)],bins=np.linspace(0,1,nbins),
                density=True);
        bkgd_hist, _ = np.histogram(predictions[~(test.get_label().astype(bool))],bins=np.linspace(0,1,nbins),
                density=True);

        print("predictions",len(predictions[test.get_label().astype(bool)]))
        sig_hists.append(sig_hist)
        bkgd_hists.append(bkgd_hist)

        # choose score cuts:\

    thresholds = np.linspace(0, 1, 10000)

    labels = test_set.Label.cat.codes

    sig_efficiencies = []
    bkgd_efficiencies = []
    significances = []
    diffs = []
    optimal_threshold = None
    max_diff = 0
    max2_diff = 0
    optimal_sig = None
    optimal_bkgd = None
    next_thresh = None
    next_sig = None
    next_bkgd = None

    for threshold in thresholds:

        predicted_labels = (predictions >= threshold).astype(int)

        #true positives
        TP = np.sum((predicted_labels == 1) & (labels == 1))

        #false neg
        FN = np.sum((predicted_labels == 0) & (labels == 1))

        #true neg
        TN = np.sum((predicted_labels == 0) & (labels == 0))

        #false pos
        FP = np.sum((predicted_labels == 1) & (labels == 0))

        if (TP + FN) != 0:
            sig_efficiency = TP / (samplesize/2)
        else:
            sig_efficiency = 0


        if (FP + TN) != 0:
            bkgd_efficiency = FP / (samplesize/2)
        else:
            bkgd_efficiency = 0


        S = TP
        B = FP

        # Compute significance using Z = S / sqrt(S + B)
        if (S+B) > 0:
            significance = sig_efficiency /  (a/2+(B**(1/2))) #S / (S + B)**2 #
        else:
            significance = 0  # Handle cases where S + B is 0

        #compute max distance between efficiencies

        #diff = sig_efficiency - bkgd_efficiency
        diff = significance
        if diff > max_diff:
            max_diff = diff
            optimal_threshold = 0.99 #threshold
            optimal_sig = sig_efficiency
            optimal_bkgd = bkgd_efficiency
        elif diff > max2_diff:
            max2_diff = diff
            next_thresh = 0.99#threshold
            next_sig = sig_efficiency
            next_bkgd = bkgd_efficiency


        # Append the significance value
        significances.append(significance)
        diffs.append(diff)
        sig_efficiencies.append(sig_efficiency)
        bkgd_efficiencies.append(bkgd_efficiency)


    #creating hist
    bins = np.linspace(0,1,nbins)
    mean_signal_hist = np.mean(sig_hists, axis=0)
    std_signal_hist = np.std(sig_hists, axis=0)

    mean_background_hist = np.mean(bkgd_hists, axis=0)
    std_background_hist = np.std(bkgd_hists, axis=0)

    bin_centers = (np.linspace(0, 1, nbins)[:-1] + np.linspace(0, 1, nbins)[1:]) / 2

    bin_width = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure()

    # Plot signal histogram with error bars
    plt.bar(bin_centers, mean_signal_hist, width=bin_width, edgecolor='midnightblue', alpha=0.2, label=f'ALP (sig)',  error_kw=dict(ecolor='black', capsize=3))

    # Plot background histogram with error bars
    plt.bar(bin_centers, mean_background_hist, width=bin_width, edgecolor='firebrick', alpha=0.2, label='Ecal PN (bkg)',  error_kw=dict(ecolor='black', capsize=3))
    plt.axvline(optimal_threshold, color='y', linestyle='--', label=f'Optimal Thresh: {optimal_threshold:.3f}')
    plt.xlabel('Prediction from BDT', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig("score.pdf")

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, sig_efficiencies, label=f'Signal Efficiency: {optimal_sig:.3f}', color='r')
    plt.plot(thresholds, bkgd_efficiencies, label=f'Background efficiency: {optimal_bkgd:.5f}', color='b')
   # plt.plot(thresholds, significances, label='Significances', color='g')
    plt.axvline(optimal_threshold, color='y', linestyle='--', label=f'Optimal Thresh: {optimal_threshold:.3f}')
    plt.xlabel('BDT score cuts')
    plt.ylabel('Efficiency')
    plt.title('Signal vs. Background Efficiency Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("sigeff.pdf")

    plt.figure(figsize=(8, 6))
    plt.plot(bkgd_efficiencies, sig_efficiencies, label=f'combined sample', color='r')
   # plt.plot(thresholds, significances, label='Significances', color='g')
    #plt.axvline(optimal_threshold, color='y', linestyle='--', label=f'Optimal Thresh: {optimal_sig:.2f}')
    plt.xlabel('Bkg. Eff.')
    plt.ylabel('Sig. Eff.')
    # Set the x-axis range
    plt.xlim(0,0.01)

    # Set the y-axis range
    plt.ylim(0.8,1)
    plt.title('Signal vs. Background Efficiency Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("FOMcurve.pdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)
