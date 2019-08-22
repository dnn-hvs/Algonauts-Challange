#!/usr/bin/env python
# This script computes the score for the comparison of Model RDM with
# the MEG data
# Input
#   -target_rdm.mat is the file that contains MEG RDM matrices from two time intervals.
#   -submit_rdm.mat is the file consisting of the model RDM to be compared against the MEG data
# Output
#   -MEG_RDMs_early and MEG_RDMs_late is the correlation of model RDMs to early and late time intervals of the MEG RDM respectively
#   -pval is the corresponding p-value showing the significance of the correlation
# Note: Remember to use the appropriate noise ceiling correlation values for the dataset you are testing
# e.g. nc118_early_R2 for the 118-image training set.

import os
import sys

import h5py
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform
from scipy import io

# defines the noise ceiling squared correlation values for early and late time intervals, for the training (92, 118) and test (78) image sets
nc92_early_R2 = 0.4634
nc92_late_R2 = 0.2275
nc92_avg_R2 = (nc92_early_R2+nc92_late_R2)/2.

nc118_early_R2 = 0.3468
nc118_late_R2 = 0.2265
nc118_avg_R2 = (nc118_early_R2+nc118_late_R2)/2.

nc78_early_R2 = 0.3562
nc78_late_R2 = 0.4452
nc78_avg_R2 = (nc78_early_R2+nc78_late_R2)/2.


# loads the input files if in .mat format
def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}


def loadnpy(npyfile):
    return np.load(npyfile)


def load(data_file):
    root, ext = os.path.splitext(data_file)
    return {'.npy': loadnpy,
            '.mat': loadmat
            }.get(ext, loadnpy)(data_file)


def sq(x):
    return squareform(x, force='tovector', checks=False)


# defines the spearman correlation
def spearman(model_rdm, rdms):
    model_rdm_sq = sq(model_rdm)
    return [stats.spearmanr(sq(rdm), model_rdm_sq)[0] for rdm in rdms]


# computes spearman correlation (R) and R^2, and ttest for p-value
def meg_rdm(model_rdm, meg_rdms):
    corr = np.mean([spearman(model_rdm, rdms) for rdms in meg_rdms], 1)
    corr_squared = np.square(corr)
    return np.mean(corr_squared), stats.ttest_1samp(corr_squared, 0)[1]


def evaluate_meg(submission, targets, target_names=['MEG_RDMs_early', 'MEG_RDMs_late']):
    out = {name: meg_rdm(submission[name], targets[name])
           for name in target_names}
    out['score'] = np.mean([x[0] for x in out.values()])
    return out


# function that evaluates the RDM comparison. For Local machine usage.
def test_meg_submission_92(submit_file_dir, results_file_name):
    target_file = '../data/Training_Data/92_Image_Set/target_meg.mat'
    target = load(target_file)
    results_file = open(results_file_name + ".txt", "a+")
    results_file.write('=' * 20)
    results_file.write('Start of test run for 92 images MEG')
    results_file.write('=' * 20)

    target = load(target_file)
    for subdir, dirs, files in os.walk(submit_file_dir):
        if len(dirs) == 0 and len(files) != 0:
            file = subdir + '/submit_meg.mat'
            results_file.write('=' * 20)
            results_file.write('\nLayer: {}'.format(file.split('/')[-2]))

            submit = load(file)
            out = evaluate_meg(submit, target)
            early_percentNC = ((out['MEG_RDMs_early'][0]) /
                               nc92_early_R2)*100.  # early percent of noise ceiling
            late_percentNC = ((out['MEG_RDMs_late'][0])/nc92_late_R2) * \
                100.  # late percent of noise ceiling
            score_percentNC = ((out['score'])/nc92_avg_R2) * \
                100.  # avg (score) percent of noise ceiling
            results_file.write('=' * 20)
            results_file.write('\nMEG results:\n')
            res_str = 'Squared correlation of model to earlier time interval (R**2): {}'.format(out['MEG_RDMs_early'][0]) + ' Percentage of noise ceiling: {}'.format(
                early_percentNC) + '%' + '  and significance: {}\n'.format(out['MEG_RDMs_early'][1])
            results_file.write(res_str)
            res_str = 'Squared correlation of model to later time interval (R**2): {}'.format(out['MEG_RDMs_late'][0]) + '  Percentage of noise ceiling: {}'.format(
                late_percentNC) + '%' + '  and significance: {}\n'.format(out['MEG_RDMs_late'][1])
            results_file.write(res_str)
            res_str = 'SCORE (average of the two correlations): {}'.format(
                out['score']) + '  Percentage of noise ceiling: {}'.format(score_percentNC) + '%\n'
            results_file.write(res_str)
    results_file.write('=' * 20)
    results_file.write('End of test run')
    results_file.write('=' * 20)
    results_file.write('\n\n\n')


def test_meg_submission_118(submit_file_dir, results_file_name):
    target_file = '../data/Training_Data/118_Image_Set/target_meg.mat'
    target = load(target_file)
    results_file = open(results_file_name + "_meg.txt", "a+")
    results_file.write('=' * 20)
    results_file.write('Start of test run for 118 images MEG')
    results_file.write('=' * 20)

    target = load(target_file)
    for subdir, dirs, files in os.walk(submit_file_dir):
        if len(dirs) == 0 and len(files) != 0:
            print(subdir,  dirs, files)
            file = subdir + '/submit_meg.mat'
            results_file.write('=' * 20)
            print(file.split('/')[-3])
            results_file.write('\nLayer: {}'.format(file.split('/')[-2]))
            submit = load(file)
            out = evaluate_meg(submit, target)
            early_percentNC = ((out['MEG_RDMs_early'][0]) /
                               nc118_early_R2)*100.  # early percent of noise ceiling
            late_percentNC = ((out['MEG_RDMs_late'][0])/nc118_late_R2) * \
                100.  # late percent of noise ceiling
            score_percentNC = ((out['score'])/nc118_avg_R2) * \
                100.  # avg (score) percent of noise ceiling
            results_file.write('=' * 20)
            results_file.write('\nMEG results:\n')
            res_str = 'Squared correlation of model to earlier time interval (R**2): {}'.format(out['MEG_RDMs_early'][0]) + ' Percentage of noise ceiling: {}'.format(
                early_percentNC) + '%' + '  and significance: {}\n'.format(out['MEG_RDMs_early'][1])
            results_file.write(res_str)
            res_str = 'Squared correlation of model to later time interval (R**2): {}'.format(out['MEG_RDMs_late'][0]) + '  Percentage of noise ceiling: {}'.format(
                late_percentNC) + '%' + '  and significance: {}\n'.format(out['MEG_RDMs_late'][1])
            results_file.write(res_str)
            res_str = 'SCORE (average of the two correlations): {}'.format(
                out['score']) + '  Percentage of noise ceiling: {}'.format(score_percentNC) + '%\n'
            results_file.write(res_str)
    results_file.write('=' * 20)
    results_file.write('End of test run')
    results_file.write('=' * 20)
    results_file.write('\n\n\n')


if __name__ == '__main__':
    for subdir, dirs, files in os.walk(sys.argv[1] + '/118images_rdms/pearson'):
        if len(dirs) == 0 and len(files) != 0:
            net = subdir.split('/')[-2]
            test_meg_submission_118(subdir, net)
    for subdir, dirs, files in os.walk(sys.argv[1] + '/92images_rdms/pearson'):
        if len(dirs) == 0 and len(files) != 0:
            net = subdir.split('/')[-2]
            test_meg_submission_92(subdir, net)
