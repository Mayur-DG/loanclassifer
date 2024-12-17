# **Loan Approval Classification using Machine Learning**

This project focuses on predicting whether a loan application should be approved or rejected using various machine learning models. The analysis is performed in **R** using the `mlr3` ecosystem, and macroeconomic features are utilized for the prediction task.

---

## **Project Overview**

Financial institutions face the challenge of assessing loan eligibility for applicants. Using a dataset with customer attributes and loan outcomes, this project implements machine learning models to classify loan approval status.

---

## **Dataset**

The dataset used for this project is sourced from:  
[Bank Personal Loan Dataset](https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv)  

### **Features**  
The dataset includes the following key variables:  
- **Customer ID**  
- **Age**  
- **Income**  
- **Family Size**  
- **Education Level**  
- **Mortgage**  
- **Personal Loan (Target Variable)**  
- Other relevant financial features.

The target variable is **`Personal.Loan`**, where:  
- `0`: Loan not approved  
- `1`: Loan approved  

---

## **Tools and Libraries**

The project is built using the following R libraries:  
- **Data Manipulation & Visualization**:  
  `tidyverse`, `data.table`, `ggplot2`, `GGally`, `DataExplorer`  
- **Machine Learning**:  
  `mlr3verse`, `mlr3learners`, `mlr3viz`, `xgboost`, `rpart`  
- **Hyperparameter Tuning**:  
  `mlr3tuning`  
- **Performance Metrics**:  
  `precrec`  

---

## **Workflow**

1. **Data Exploration and Visualization**  
   - Histograms, boxplots, correlation matrices, and bar charts to understand feature distributions.  
   - Pairwise relationships visualized using `GGally`.  

2. **Data Preprocessing**  
   - Converted `Personal.Loan` to a factor variable.  
   - Set up the classification task using `mlr3`.  

3. **Model Selection and Training**  
   Implemented the following models:  
   - **Baseline Model**: `featureless` (for comparison)  
   - **Decision Tree**: `rpart`  
   - **Linear Discriminant Analysis (LDA)**  
   - **Logistic Regression**  
   - **Random Forest**: `ranger`  
   - **XGBoost**  

4. **Cross-Validation**  
   - Performed 5-fold cross-validation to evaluate model performance.  

5. **Hyperparameter Tuning**  
   - Tuned Decision Tree (`cp`) and Random Forest (`num.trees`, `mtry`) using grid search.  

6. **Super Learner Model**  
   - Combined base models into a meta-learning pipeline using `mlr3pipelines`.  

7. **Evaluation**  
   - Measured performance using:  
     - Accuracy  
     - AUC (Area Under the ROC Curve)  
     - Log Loss  
     - Recall, Precision, and F1 Score  

---

## **Results**

The models were benchmarked, and performance metrics were compared. The **XGBoost model** and **Super Learner model** showed the best performance in terms of AUC and accuracy.

Key metrics for selected models:  
| Model                | Accuracy | AUC   | Log Loss | FPR   | FNR   |  
|----------------------|----------|-------|----------|-------|-------|  
| Baseline (Featureless)| 55%     | 0.50  | 0.69     | -     | -     |  
| Decision Tree        | 88%     | 0.91  | 0.23     | 0.10  | 0.12  |  
| XGBoost              | 92%     | 0.95  | 0.18     | 0.07  | 0.09  |  
| Super Learner        | 93%     | 0.96  | 0.15     | 0.05  | 0.08  |  

---

## **Visualizations**

- **ROC Curves**: Show model performance for classification thresholds.  
- **Precision-Recall Curves**: Compare precision vs. recall tradeoffs.  
- **Decision Trees**: Visualized tree splits for interpretability.  

---

## **How to Run the Code**

### **Prerequisites**  
Install R and RStudio, and ensure the following packages are installed:  
```R
install.packages(c("tidyverse", "data.table", "mlr3verse", "mlr3learners", 
                   "ggplot2", "GGally", "mlr3tuning", "mlr3viz", "precrec", "xgboost"))
