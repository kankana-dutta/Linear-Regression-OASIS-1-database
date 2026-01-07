# Linear-Regression-OASIS-1-database
Applies linear regression to the OASIS-1 neuroimaging dataset

**Overview**
This project applies linear regression to the OASIS-1 neuroimaging dataset to study what can be reliably predicted from demographic and structural brain features. It contrasts two tasks:
1. Predicting cognitive performance (MMSE) — difficult and unstable
2. Predicting brain atrophy (nWBV) — strong and biologically meaningful

The project demonstrates how target selection critically affects model performance.
**Key variables**:
a.Age, Education
b.eTIV (intracranial volume)
c.nWBV (normalized whole brain volume)
d.ASF (atlas scaling factor)
e.Handedness
f.MMSE, CDR

Task 1:

**Initial Attempt: MMSE Prediction**

1. MMSE contains many missing values
2. Remaining samples are biased
3. Linear regression shows very low test
4. Predictions collapse toward the mean

<img width="600" height="500" alt="Figure_2_MMSA" src="https://github.com/user-attachments/assets/b61df84a-88f3-4df9-b7b2-6b4a89314974" />
<img width="600" height="600" alt="Figure_1_MMSA" src="https://github.com/user-attachments/assets/b629e293-c19c-4124-94e1-5e18e4ebc22d" />

**Conclusion:**
Global brain measures are insufficient to predict cognition using simple linear models.


Task 2:

**Modified Task: Predicting Brain Atrophy (nWBV)**
Target:   nWBV
Features: Age, eTIV, ASF, Hand

**Methods**
1. Median imputation + standardization
2. One-hot encoding for handedness
3. Linear Regression 

**Observations**
a.Over 80% of variance in brain atrophy is explained
b. Strong generalization to unseen data
c.Ridge confirms model stability

<img width="600" height="500" alt="Figure_2_volume" src="https://github.com/user-attachments/assets/5e188f37-d455-4082-8bdf-d62f89173e3e" />
<img width="600" height="600" alt="Figure_1_volume" src="https://github.com/user-attachments/assets/5c615a79-90b5-4cb9-b98a-438a401f37ea" />

**Conclusion**
1. Linear regression works well when the target is biologically direct
2. Brain structure is predictable from age and anatomy
3. Cognitive scores require richer features and more complex models
