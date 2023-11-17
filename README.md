[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyPI license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/LeonardoAlchieri/LateralizationLaughter/blob/main/LICENCE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/LeonardoAlchieri/LateralizationLaughter/graphs/commit-activity)

# LateralizationLaughter

Code for the 7th International Workshop on Mental Health and Well-being: Sensing and Intervention @ UBICOMP 2022 paper: **Lateralization Analysis in Physiological Data from Wearable Sensors for Laughter Detection**.

The code can be divided into 3 main parts:
1. Pre-processing
2. Statistical Analysis
3. Machine Learning Task

To install the used libraries, just run `pip install -r requirements.txt`. Keep in mind that some libraries used are not published on pip, but they are directly GitHub branches. For more information on them, feel free to contact me.

## Pre-processing

For the first, everything is run using some custom scripts (`src/run/pre_processing`). Each of them is controlled by a specific configuration file, where some simple parameters can be changed (by default the ones used for our work are present).
While the order of scripts is not important, to run them using the order we performed you can simply do:
    ```bash
        sh src/run/pre_processing_run_all_preprocessing.sh
    ```
We suggest to use this, since other configurations would require to change a little bit the configuration files.
Finally, to prepare the data for effect size and correlation:

1. Segment the data:
```bash
python src/run/feature_extraction/run_segmentation.py
```
2. Extract hand-crafted features from each segment:
```bash
python src/run/feature_extraction/run_feature_extraction.py
```


## Machine Learning Task

There are several machine learning tasks available. To emulate the work performed in the paper, and specifically run the $5\times5$, Leave One Body side Out and Leave One Subject Out cross validations, use the following:

1. To perform $5\times5$ cv and LOBO (at the same time):
```bash
python src/run/ml/run_nested.py
```
2. To perform LOSO cv, run:
```bash
python src/run/ml/run_nested_loso.py
```
Each python file is associated with a config file, which allows to change, for example, between the two datasets. For each specific configuration available, see the configs themselves.

## Statistical Analysis

THe paper presents a statistical analysis. In particular, we performed correlation analysis using the Detrended Time-Lagged Correlation Coefficient, whose library we have also released (see https://pypi.org/project/dcca/). Then, we computed effect size analysis using Cliff's $\delta$ with confidence intervals. We released for this one as well a library to perform the calculation for Cliff's $\delta$ and its confidence interval (see https://pypi.org/project/effect-size-analysis/). Finally, we also computed the correlation between features extracted from the left and right side.

To run these, use:
1. Detrended Time-Lagged Correlation Coefficient between left and right-hand signals:
<!-- TODO -->
2. Effect size analysis:
```bash
python src/statistical_analysis/run_cliff_delta_features.py
```
3. Correlation features:
```bash
python src/statistical_analysis/run_correlation_rl_features.py
```

## Plots
The scripts above allow to calculate the results, but not to make the plots present in the paper. To make the plots, run the following scripts:
<!-- TODO -->


@2023, Leonardo Alchieri, Nouran Abdalazim, Lidia Alecci, Shkurta Gashi, Elena Di Lascio, Silvia Santini
 
Contact: leonardo.alchieri@usi.ch


