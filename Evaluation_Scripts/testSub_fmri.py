#!/usr/bin/env python
# This script computes the score for the comparison of Model RDM with
# the fMRI data
# Input
#   -target_rdm.mat is the file that contains EVC and IT fMRI RDM matrices.
#   -submit_rdm.mat is the file that is the model RDM to be compared against the fMRI data submitted
# Output
#   -EVC_corr and IT_corr is the correlation of model RDMs to EVC RDM and IT RDM respectively
#   -pval is the corresponding p-value showing the significance of the correlation
# Note: Remember to use the appropriate noise ceiling correlation values for the dataset you are testing
# e.g. nc118_EVC_R2 for the 118-image training set.

import os
import sys

import h5py
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform
from scipy import io

# defines the noise ceiling squared correlation values for EVC and IT, for the training (92, 118) and test (78) image sets
nc92_EVC_R2 = 0.1589
nc92_IT_R2 = 0.3075
nc92_avg_R2 = (nc92_EVC_R2+nc92_IT_R2)/2.

nc118_EVC_R2 = 0.1048
nc118_IT_R2 = 0.0728
nc118_avg_R2 = (nc118_EVC_R2+nc118_IT_R2)/2.

nc78_EVC_R2 = 0.0640
nc78_IT_R2 = 0.0647
nc78_avg_R2 = (nc78_EVC_R2+nc78_IT_R2)/2.


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


# computes spearman correlation (R) and R^2, and ttest for p-value.
def fmri_rdm(model_rdm, fmri_rdms):
    corr = spearman(model_rdm, fmri_rdms)
    corr_squared = np.square(corr)
    return np.mean(corr_squared), stats.ttest_1samp(corr_squared, 0)[1]


def evaluate(submission, targets, target_names=['EVC_RDMs', 'IT_RDMs']):
    out = {name: fmri_rdm(submission[name], targets[name])
           for name in target_names}
    out['score'] = np.mean([x[0] for x in out.values()])
    return out


# function that evaluates the RDM comparison.
def test_fmri_submission():
    target_file = '../Training_Data/92_Image_Set/target_fmri.mat'
    # Sq-Net1_0
    # submit_file = ['../Feature_Extract/rdms/92images_rdms/sqnet1_0/pearson/maxpool1/submit_fmri.mat',
    #                '../Feature_Extract/rdms/92images_rdms/sqnet1_0/pearson/maxpool4/submit_fmri.mat',
    #                '../Feature_Extract/rdms/92images_rdms/sqnet1_0/pearson/maxpool8/submit_fmri.mat',
    #                '../Feature_Extract/rdms/92images_rdms/sqnet1_0/pearson/fire8/submit_fmri.mat',
    #                '../Feature_Extract/rdms/92images_rdms/sqnet1_0/pearson/conv10/submit_fmri.mat',
    #                ]
    submit_file = [
        '../Feature_Extract/rdms/92images_rdms/alexnet/pearson/conv1/submit_fmri.mat',
        '../Feature_Extract/rdms/92images_rdms/alexnet/pearson/conv2/submit_fmri.mat',
        '../Feature_Extract/rdms/92images_rdms/alexnet/pearson/conv3/submit_fmri.mat',
        '../Feature_Extract/rdms/92images_rdms/alexnet/pearson/conv4/submit_fmri.mat',
        '../Feature_Extract/rdms/92images_rdms/alexnet/pearson/conv5/submit_fmri.mat',
        '../Feature_Extract/rdms/92images_rdms/alexnet/pearson/fc6/submit_fmri.mat',
        '../Feature_Extract/rdms/92images_rdms/alexnet/pearson/fc7/submit_fmri.mat',
        '../Feature_Extract/rdms/92images_rdms/alexnet/pearson/fc8/submit_fmri.mat',

    ]
    target = load(target_file)
    # results_file = open("sqnet1_0.txt", "w+")
    results_file = open("alexnet.txt", "w+")

    for file in submit_file:
        results_file.write('=' * 20)
        results_file.write('\n{}'.format(file))

        submit = load(file)
        out = evaluate(submit, target)
        evc_percentNC = ((out['EVC_RDMs'][0])/nc92_EVC_R2) * \
            100.  # evc percent of noise ceiling
        it_percentNC = ((out['IT_RDMs'][0])/nc92_IT_R2) * \
            100.  # it percent of noise ceiling
        score_percentNC = ((out['score'])/nc92_avg_R2) * \
            100.  # avg (score) percent of noise ceiling
        results_file.write('\nfMRI results:\n')
        res_str = 'Squared correlation of model to EVC (R**2): {}'.format(out['EVC_RDMs'][0]) + ' Percentage of noise ceiling: {}'.format(
            evc_percentNC) + '%' + '  and significance: {}'.format(out['EVC_RDMs'][1])+'\n'
        results_file.write(res_str)
        res_str = 'Squared correlation of model to IT (R**2): {}'.format(out['IT_RDMs'][0]) + '  Percentage of noise ceiling: {}'.format(
            it_percentNC) + '%' + '  and significance: {}'.format(out['IT_RDMs'][1])+'\n'
        results_file.write(res_str)
        res_str = 'SCORE (average of the two correlations): {}'.format(
            out['score']) + '  Percentage of noise ceiling: {}'.format(score_percentNC) + '%'+'\n'
        results_file.write(res_str)


if __name__ == '__main__':
    test_fmri_submission()
