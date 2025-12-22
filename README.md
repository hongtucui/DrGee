A tool for cancer Drug Ranking based on Gene Essentiality derived from Expression

<img width="962" height="248" alt="image" src="https://github.com/user-attachments/assets/6c61a274-fddb-45ba-935e-76ff1287b956" />

A cancer cell drug sensitivity prediction model, DeepEEAA, is embedded in DrGee. Its input features include gene essentiality data, gene expression data, drug-protein affinity data, and drug-gene associations.

**Installation**

Python version: 3.9.16


PyTorch version: 2.6.0+cu118


NumPy version: 2.0.1


Pandas version: 2.2.3


Scikit-learn version: 1.6.1


Hyperopt version: 0.2.7


**Instructions**

cal_ic50.R:   This study used four drug cell line sensitivity datasets (secondary, CTPR, GDSC1, and GDSC2) from the DepMap database as research objects, and re fitted the dose-response curves using the "dr4pl" package of R software; For repeated drug cell line data, take the median of their values and merge them.

predict_ic50.py:  Model Training and Optimal Parameter Optimization of a Regression Model for Predicting Drug Sensitivity (IC50).

predict_response.py: Model Training and Optimal Parameter Optimization of a classification Model for Predicting Patients’ Response to Drugs.

test_ic50.py:   IC50 test using the optimal parameter independent test set.

views.py:  Backend files of DrGee online tool.





