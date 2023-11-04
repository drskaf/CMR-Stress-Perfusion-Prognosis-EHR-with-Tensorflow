# Stress-CMR-and-EHR-Outcome-Prediction

This is an example of electronic health records (EHR) modelling for outcome prediction (mortality and ventricular arrhythmia) using conventional statistics with Cox model, then training different machine learning models (SVM, RF, XGBoost) to predict mortality outcome and comparing it with linear regression. 
The sample codes are not exclusive, but they give some starting points to explore EHR data using machine learning. 

The prediction variables are a mixture of clinical and cardiac magnetic resonance (CMR) reports data. 

## Files

file_build.py: will curate dataframes and organise relevant variable culumns. 

eda.py: uses conventional statistics to explore the data 

survival.py: performs all conventional statistics for survival analysis

clinical_survmodel_compare.py: will fit different machine learning models and compare their performance using AUC and F1 score


