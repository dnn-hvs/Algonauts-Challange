This readme describes the evaluation scripts for the Algonauts Project Challenge 2019.

Consists of two scripts, “testSub_fmri.py” and “testSub_meg.py”, to compare the participant’s model RDM to fMRI and MEG brain data respectively.


“testSub_fmri.py”
Input:
- “submit_fmri.mat” will consist of the model RDMs that is evaluated against fMRI data.
The “submit_fmri.mat” file should be of the following format:
Contains two fields. First field named EVC_RDMs is the model RDM to be compared against EVC fMRI and second field named IT_RDMs to be compared against IT fMRI data.
- “target_fmri.mat” is the provided fMRI data file.
(the above data can be either in “.npy” or “.mat” format)

Output:
On executing the script, the output will consist of three results:
(1) Model to EVC correlation, its percentage of the noise ceiling, and correlation significance value
(2) Model to IT correlation, its percentage of the noise ceiling, and correlation significance value
(3) Mean of the above two correlation values (Score) and percentage of the noise ceiling

The brain data is limited by the measurement noise and the amount of data. Therefore, we do not expect that a model RDM reaches a correlation of 1 with the brain data RDMs. The noise ceiling is the expected RDM correlation achieved by the (unknown) ideal model, given the noise in the data. 

The EVC and IT correlations are obtained from the comparison of respective model RDMs to the corresponding fMRI RDMs. The significance values are obtained from a t-test over fMRI data of 15 participants. The score is obtained by averaging the two correlations. The model with the highest score on fMRI data wins the Algonauts Project Challenge Track 1!


“testSub_meg.py”
Input:
- “submit_meg.mat” will consist of the model RDMs that is evaluated against MEG data.
The “submit_meg.mat” should be of the following format:
Contains two fields. First field named MEG_RDMs_early is the model RDM to be compared against MEG RDM of the first time interval and second field named MEG_RDMs_late to be compared against second time interval.
- “target_fmri.mat” is the provided MEG data file.
(the above data can be either in .npy or .mat format)

Output:
On executing the script, the output will consist of three results:
(1) Model correlation to early time interval, its percentage of the noise ceiling, and correlation significance value
(2) Model correlation to later time interval, its percentage of the noise ceiling, and correlation significance value
(3) Mean of the above two correlation values (Score) and its percentage of the noise ceiling

The brain data is limited by the measurement noise and the amount of data. Therefore, we do not expect that a model RDM reaches a correlation of 1 with the brain data RDMs. The noise ceiling is the expected RDM correlation achieved by the (unknown) ideal model, given the noise in the data. 

The correlations for the two time intervals are obtained by comparing respective model RDMs to the corresponding MEG RDMs. The significance values are obtained from a t-test over MEG data of 15 participants. The score is obtained by averaging the two correlations. The model with the highest score on MEG data wins the Algonauts Project Challenge Track 2!

