library(tidyverse)
library("data.table")
library("mlr3verse")
library("mlr3learners")
library(ggplot2) 
library(GGally) 
library(mlr3tuning)
library(mlr3viz)
install.packages("precrec")
install.packages("xgboost")
#############################################Data visualization################################################
credit.data <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
View(credit.data)
summary(credit.data)
cor(credit.data)
skimr::skim(credit.data)
DataExplorer::plot_bar(credit.data, ncol = 3)
DataExplorer::plot_histogram(credit.data, ncol = 3)
DataExplorer::plot_boxplot(credit.data, by = "Personal.Loan", ncol = 3)
ggpairs(credit.data)
#######################################################Models fitting and pre processing#########################
credit.data$Personal.Loan <- as.factor(credit.data$Personal.Loan)
set.seed(212) # set seed for reproducibility
credit.task <- TaskClassif$new(id = "loanCredit",
                               backend = credit.data, # <- NB: no na.omit() this time
                               target = "Personal.Loan",
                               positive = "0")
cv5<- rsmp("cv", folds = 5)
cv5$instantiate(credit.task)

lrn.baseline <- lrn("classif.featureless", predict_type = "prob")
lrn.cart <- lrn("classif.rpart", predict_type = "prob")  
lrn.lda<- lrn("classif.lda", predict_type = "prob")

res.baseline <- resample(credit.task, lrn.baseline, cv5, store_models = TRUE)
res.cart <- resample(credit.task, lrn.cart, cv5, store_models = TRUE)  # Corrected variable name
res.lda<-resample(credit.task, lrn.lda, cv5, store_models = TRUE)

res.baseline$aggregate()
res.cart$aggregate()
res.lda$aggregate()

res <- benchmark(data.table(
  task       = list(credit.task),
  learner    = list(lrn.baseline,
                    lrn.cart,lrn.lda),
  resampling = list(cv5)
), store_models = TRUE)
res
res$aggregate()
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
#########################Tree plot#################################3
trees <- res$resample_result(2)

tree1 <- trees$learners[[1]]

tree1_rpart <- tree1$model
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)
plot(res$resample_result(2)$learners[[5]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[5]]$model, use.n = TRUE, cex = 0.8)

lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

res_cart_cv <- resample(credit.task, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[5]]$model)

lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.013)

res <- benchmark(data.table(
  task       = list(credit.task),
  learner    = list(lrn.baseline,
                    lrn.cart,
                    lrn_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

# Create a pipeline which encodes and then fits an XGBoost model
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)

lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob",num.trees=100)
res <- benchmark(data.table(
  task       = list(credit.task),
  learner    = list(lrn.baseline,
                    lrn.cart,lrn.lda,
                    lrn_cart_cp,
                    pl_xgb,
                    lrn_log_reg,lrn_ranger),
  resampling = list(cv5)
), store_models = TRUE)

initialstats<-res$aggregate(list(msr("classif.ce"),
                                 msr("classif.acc"),
                                 msr("classif.fpr"),
                                 msr("classif.fnr"),msr("classif.auc"),
                                 msr("classif.logloss")))
initialstats
autoplot(res)
#####################################defined to plot roc and recall-precision plots only, the answers match initial stats######## 
tasks = credit.task
# learner = list(lrn.baseline,
#                lrn.cart,lrn.lda,
#                lrn_cart_cp,
#                pl_xgb,
#                lrn_log_reg,lrn_ranger)
# resampling = rsmps("cv")
learner = lrns(c("classif.featureless", "classif.rpart", "classif.xgboost","classif.lda","classif.log_reg","classif.ranger"), predict_type = "prob")
resampling = rsmps("cv")
# bmr = benchmark(benchmark_grid(tasks, learner, resampling))
bmr = benchmark(benchmark_grid(tasks, learner, resampling))
autoplot(bmr, type = "roc")
autoplot(bmr,type="prc")
D################################################Super learner model##################################################
# cv5 <- rsmp("cv", folds = 5)
# cv5$instantiate(credit.task)

set.seed(468) # set seed for reproducibility
super.task <- TaskClassif$new(id = "loanCredit",
                              backend = credit.data, 
                              target = "Personal.Loan",
                              positive = "1")
# Define a collection of base learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob",id="featureless")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob",id="cart")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.013, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob",id="ranger")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob",id="boost")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob",id="logistic")
# lrn_lda=lrn("classif.lda",predict_type="prob")
# Define a super learner
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")
pl_factor <- po("encode")

# Now define the full pipeline
spr_lrn <- gunion(list(
  # First group of learners requiring no modification to input
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv",lrn_cart),
    po("learner_cv",lrn_cart_cp)
  )),
  gunion(list(
    po("learner_cv", lrn_ranger),
    po("learner_cv", lrn_log_reg))),
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)

# This plot shows a graph of the learning pipeline
spr_lrn$plot()
# Finally fit the base learners and super learner and evaluate
res_spr <- resample(super.task, spr_lrn, cv5, store_models = TRUE)

superstats<-res_spr$aggregate(list(msr("classif.ce"),
                                   msr("classif.acc"),
                                   msr("classif.fpr"),
                                   msr("classif.fnr"),
                                   msr("classif.auc"),
                                   msr("classif.logloss"),msr("classif.tnr"),msr("classif.tpr"),msr("classif.precision"),msr("classif.recall")))
superstats
autoplot(res_spr)
# ROC curve for resample result
autoplot(res_spr,type="roc")
autoplot(res_spr, type = "prc")
autoplot(res_spr, type = "prediction")
#as.data.table(mlr_measures)
