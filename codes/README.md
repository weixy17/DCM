## Introduction

Objectives: The mechanism of neurological recovery after surgical decompression is still unclear for degenerative cervical myelopathy (DCM). This study aims to analyze the relationship between microvasculature in the postdecompressive spinal cord and neurological recovery and further design a prediction model of recovery. 
Materials & Methods: This retrospective study analyzed data from a prospectively enrolled cohort of patients with DCM who underwent surgical decompression from May 2023 to May 2024 at a single center. Intraoperative US ultrafast Doppler imaging was performed to evaluate microvasculature of postdecompressive spinal cords. Differences in five microvascular indexes between postdecompressive spinal cords and normal ones were calculated for each participant. All participants were followed up for 6 months to obtain modified Japanese Orthopedic Association (mJOA) scores. Participants were divided into favorable and unfavorable recovery groups based on a threshold of 50% for the recovery rate of mJOA scores. The Mann-Whitney U test compared microvascular indexes between the two groups. Kolmogorov-Arnold Network was further introduced to predict neurological recovery based on the proposed microvascular indexes.
Results: 48 participants (median age, 59 years [IQR, 54–66 years]; 36 males) were included. 37 participants had a favorable recovery. All five microvascular indexes were associated with neurological recovery (all p < 0.05). The proposed prediction model achieves an accuracy of 94% (45 of 48) and an area under the receiver operating characteristic curve (AUC) of 0.98 (95% CI: 0.90, 1.00). 
Conclusion: Microvasculature of postdecompressive spinal cords is associated with neurological recovery and the proposed model can effectively predict neurological recovery.

This repository includes:

- Codes for Kolmogorov-Arnold Network (KAN) based prediction model using Python and Pytorch
- Pretrained models.

Code has been tested with Python 3.9 and MXNet 1.10.1.
For more requirements for the proposed KAN-based model, please refer to https://github.com/KindXiaoming/pykan.

## Datasets

Dataset can be found in ./data.mat, which is a struct format includes "quantitative_indexes" and "labels". "quantitative_indexes" contains a 48*13 matrix. Each row represents 13 quantitative indexes for a participant. The 13 quantitative indexes are ∆MD, ∆MA, ∆MDia, ∆MT, ∆MV, sex, age at surgery, symptom duration, preoperative mJOA score, preoperative APD, preoperative W, preoperative APD/W, and preoperative cross-sectional area respectively. "labels" is a binary matrix of 0 or 1 in the form of 48*1, representing whether the participant achieve a favorable neurological recovery. Exactly, 1 refers to favorable recovery and 0 for unfavorable recovery. 

./test_idx.txt contains 4 rows, each row represents a fold of the four-fold cross-validation as mentioned in the paper. Each row contains 12 numbers, representing the participant number (row index of the 48*x matrix in ./data.mat) in the validation dataset for every fold. The remaining 36 participants are divided into training dataset.

## Pipeline

- 1. To test the KAN-based model with 5 microvascular indexes and 8 clinical indexes as inputs, run classification_KAN_MicroandCli.py
- 2. To test the KAN-based model with 5 microvascular indexes as inputs, run classification_KAN_Micro.py
- 3. To test the KAN-based model with 8 clinical indexes as inputs, run classification_KAN_Cli.py
- 4. To test classical machine learning models including support vector machine, random forest, logistic regression, and multilayer perceptron with 5 microvascular indexes and 8 clinical indexes as inputs, run classification_SVM.py, classificationRDF.py, classification_LR.py, and classification_MLP.py, respectively.

## Pretrained models

Pretrained models are provided in ./weights/

</center>
