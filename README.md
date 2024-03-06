[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyPI license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/LeonardoAlchieri/LateralizationPhysiologicalWearables/blob/main/LICENCE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/LeonardoAlchieri/LateralizationPhysiologicalWearables/graphs/commit-activity)

# Lateralization Effects in Electrodermal Activity Data Collected Using Wearable Devices

The code is organized in three main tasks:
1. Pre-processing
2. Statistical Analysis
3. Machine Learning Task

To install the used libraries, just run `conda env create -f environment.yml`. Keep in mind that some libraries used are not published on pip, but they are directly GitHub branches. For more information on them, feel free to contact me.

All scipts meant to be run can be found inside the `src/run` folder. Other folders contain classes and methods necessary for all scripts to run.

## Pre-processing

There are pre-processing scripts for the three main signals from the E4: Accelerometer (ACC), Photoplethysmography (PPG) and Electrodermal Activity (EDA). To reproduce the work we performed, it is necessary to run only the EDA pre-processing, as the other signals are not used in the paper. 
Each file is associated with a config file, which allows to change, for example, between the two datasets. For each specific configuration available, see the configs themselves.
To run the EDA pre-processing, use the following command:

```bash
python src/run/preprocessing/run_eda_filtering.py
```

## Feature Extraction

Some investigations performed in the paper, e.g., some statistical analysis and the machine learning experiments, required the extraction of features from the EDA signal. To extract the features, first you need to segment each signal into windows of fixed length. You can use:

```bash
python src/run/feature_extraction/run_segmentation.py
```

Then, for each window, we extracted the some hand-crafted features using:
    
```bash
python src/run/feature_extraction/run_feature_extraction.py
```

## Statistical Analysis

THe paper presents a statistical analysis. In particular, we performed correlation analysis using the Detrended Time-Lagged Correlation Coefficient, whose library we have also released (see https://pypi.org/project/dcca/). Then, we computed effect size analysis using Cliff's $\delta$ with confidence intervals over the hand-crafted features extracted. We released for this one as well a library to perform the calculation for Cliff's $\delta$ and its confidence interval (see https://pypi.org/project/effect-size-analysis/). Finally, we also computed the correlation between features extracted from the left and right side.

To run these, use:
1. Stationarity Tests
```bash
python src/statistical_analysis/run_stationarity_tests.py
```
2. Detrended Time-Lagged Correlation Coefficient between left and right-hand signals:
```bash
python src/statistical_analysis/run_dcca.py
```
These results can then be used to make a plot for the max-dcca using the notebook in:
```bash
plotting_notebooks/dtw-dcca-scatterplot.ipynb
```
3. Effect size analysis:
```bash
python src/statistical_analysis/run_cliff_delta_features.py
```
The code to make the plot for the results, use the following notebook:
```bash
plotting_notebooks/cliff-delta.ipynb
```
4. Correlation features:
```bash
python src/statistical_analysis/run_correlation_rl_features.py
```
The code to make the plot for the results, use the following notebook:
```bash
plotting_notebooks/correlation-rl.ipynb
```

## Training and Testing of ML models

2 different validation validation strategies were used in the paper: 5-5 fold nested cross validation and nested leave-one-participant-out (LOPO). The first one is used for the statistical analysis, while the second one is used for the machine learning task. 
Both validation paradigms are implemented using a nested approach, with a train, validation and test sets. 
All train-validation implementations are also run multiple time for different seeds.
The machine learning task is performed using the following command:
```bash
python src/run/ml/run_nested.py
```
```bash
python src/run/ml/run_nested_loso.py
```
To analyse these results, use the following notebooks:
```bash
nested.ipynb
```
```bash
nested_loso.ipynb
```

@2023, Leonardo Alchieri, Nouran Abdalazim, Lidia Alecci, Shkurta Gashi, Elena Di Lascio, Silvia Santini
 
Contact: leonardo.alchieri@usi.ch


