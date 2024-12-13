# loan-default-prediction
For assignment 3 - MATH5836 (UNSW)

**ABSTRACT**

This project, using the vehicle loand default provided by L&T Financial Services (and available on Kaggle), aim to compare the performance of bagging and boosting ensemble classifier (random forest and XGBoost) at different setting of hyperparameters (number of trees and maximum tree depth) in the context of an imbalanced dataset. 

Different methods of mitigating class imbalance - resampling (SMOTE and SMO-TEENN), cost-sensitive learning, and a combination of both - are also investigated and compared. The results suggest that both models seems to benefit from larger numbers of trees. Contrary to traditional view, however, random forest models’ performance peaks at shallower depth, and deteriorates as maximum tree depths increases due to overfitting. Out of all combination of hyperparameters and balance methods, models using only increased positive class weight outperform ones which combine both increased positive class weight and resampling.

**REPORT**

Full report uploaded.

**NOTE**
- Stacking.py not finished
