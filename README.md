# Comparing Classifiers for Predicting Death in Heart Failure Patients

## Introduction and Goal

Heart failure is a condition where the heart becomes too weak to pump blood effectively throughout the body. Patient outcomes can vary widely and often depend on many biological factors.

**The goal of this project is to train and compare three classifiers for predicting survival in heart failure patients.** The models I’ll be using are K-Nearest Neighbors, Gaussian Naive Bayes, and Logistic Regression. Each model will learn from biological features to predict whether a patient survived or not.

The process will be as follows: first, train all three models and evaluate them using multiple performance metrics. Next, refine the models through parameter tuning and dimensionality reduction. Finally, compare their performances and determine which model provides the most reliable predictions.

## Data Description

The dataset contains information on 299 patients with heart failure. Each row represents one patient, and each of the 13 columns provides a biological or clinical feature:

- Age
- Sex
- Whether they are anaemic
- Whether they have hypertension
- CPK enzyme level
- Whether they have diabetes
- Platelet count
- Serum creatinine level
- Serum sodium level
- Smoking status
- Follow-up period
- Death event

The follow-up period indicates how long a patient was monitored after their heart failure diagnosis, with an average of 130 days. The death event column is the target variable and records whether the patient survived (0) or died (1) during follow-up.

For prediction, I’ll use the first 11 features. Since all of them are numeric or already encoded numerically, the dataset is ready for training without additional preprocessing.

Access the data [here](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records).

## Analysis: Baseline Models 

First, I'm going to train and test the 3 models using the 11 training features directly and with default hyper-parameters. 

**K-Nearest Neighbors (KNN)** is an instance-based, non-parametric algorithm that predicts outcomes based on similarity to nearby data points. Using k = 5 neighbors as the baseline parameter, it classifies a patient by majority vote among the 5 closest points. Unlike probabilistic models, it doesn’t build an explicit model of the data distribution and instead makes predictions directly from the training set. It’s sensitive to feature scaling and can be slower at larger scales, but it’s useful because it makes very few assumptions about the data.

**Gaussian Naive Bayes** is a fast, simple probabilistic classifier that applies Bayes’ theorem under the assumption of feature independence. The Gaussian version models continuous features as normally distributed. The performance of this model will depend on how well the normality assumption holds for the training features.

**Logistic Regression** is a linear probabilistic model that predicts survival probabilities using a logistic function of the input features. With its default regularization parameter of 1.0, it balances fit and simplicity. Since it doesn’t rely on independence or normality assumptions, it may outperform Naive Bayes when features are correlated or deviate from a Gaussian distribution.

Model performance will be evaluated using 4 metrics:

- Accuracy: The overall proportion of correct predictions out of all predictions.
- Precision: Of the patients predicted to die, the proportion that actually died.
- Recall: Of the patients who actually died, the proportion that the model correctly identified.
- F1-Score: The harmonic mean of precision and recall.

Since the groups are unbalanced (<33% of participants died), it’s important to evaluate model performance using more than just accuracy. A model that predicts all heart failure patients will survive would have an accuracy >67%, despite being essentially useless. The F1-score is a better measure of model fit because it balances the effects of false negatives and false positives. A false negative in this context is a patient who dies but is predicted to survive, while a false positive is the opposite: a patient who survives but is predicted to die. As in most medical cases, it’s better to catch as many at-risk patients as possible, even if that means some patients are flagged who ultimately survive. That said, maintaining a balance is still important, which is why the F1-score is a valuable metric.

## Analysis: Refined Models

### Dimensionality Reducation via PCA

By transforming correlated features into a smaller set of uncorrelated components, PCA can help remove noise and redundant information, which may improve model performance. This is especially useful for algorithms like KNN that are sensitive to irrelevant or highly correlated features, and it can also make probabilistic models like Naive Bayes and Logistic Regression more robust. 

### Parameter Tuning 

Choosing the right parameters can significantly improve model performance by balancing underfitting and overfitting. I will test different combinations of parameters using cross-validation and f1-score as the performance metric. 

The hyperparameters I will test for the KNN algorithm:
- K: the number of neighbors considered per computation (3-21)
- Weighting schema: uniform or weighted votes

GNB has no major hyperparameters to tune. 

The hyperparameter to be tuned for the LR algorithm is C which defines the degree of regularization. I will test values on an expontial scale: 0.01, 0.1, 1, 10, 100. 

Once the optimal parameters were identified, each model was re-trained and re-tested. 

## Results

![image]()

![image]()

Looking at the baseline model performances, Logistic Regression (LR) achieves the highest accuracy (0.70) and F1-score (0.4375), followed by Gaussian Naive Bayes (GNB), with K-Nearest Neighbors (KNN) performing the worst. KNN struggles in this dataset, particularly in recall (0.0526), meaning it misses almost all patients who actually died. This poor performance may be due to the small dataset size and the fact that KNN is highly sensitive to feature scaling and irrelevant or correlated features. GNB and LR show more balanced results, with LR slightly outperforming GNB overall.

After refinement with parameter tuning and PCA, the models show modest improvements. LR sees the largest gain in accuracy (0.7167) and precision (0.6), although its recall slightly decreases (0.3158), resulting in a small drop in F1 (0.4138). GNB improves in recall (0.3158) and F1-score (0.3871), indicating it catches more of the at-risk patients without sacrificing precision. KNN remains essentially unchanged, reinforcing that it is not well-suited to this dataset even after tuning and dimensionality reduction.

Applying PCA reduces the dataset to fewer orthogonal features, which can affect the models differently. GNB may perform better because the independence assumption becomes more valid when features are uncorrelated. LR could lose some interpretability since the principal components are combinations of original features, but it may gain speed and stability. KNN might benefit if PCA reduces noise, though in this case the improvement appears limited.

The differences in performance also reflect the underlying model assumptions. GNB assumes feature independence and normally distributed features, which may not fully hold in the raw dataset, limiting its predictive power. LR, by contrast, does not make these assumptions and can capture correlations between features, making it more reliable here.

## Conclusion 

This analysis suggests that LR provides the most accurate and interpretable predictions for heart failure patient survival. GNB may still be useful for quickly identifying at-risk patients, especially after refinement and dimensionality reduction, but KNN appears unsuitable for this type of structured, correlated clinical data. These results highlight the importance of selecting models whose assumptions align with the characteristics of the dataset.

View the full project [here](). 