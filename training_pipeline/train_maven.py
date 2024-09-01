# Import packages
import os
import pandas as pd
import numpy as  np
import xgboost as xgb
from sklearn.pipeline import Pipeline
import random
import train_maven_support as PS
from ruffus import *
import random
random.seed(9)
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score,precision_recall_curve, roc_curve, auc
import logging
import sys
import pickle 
import matplotlib.pyplot as plt
from sklearn import model_selection
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from itertools import chain
optuna.logging.set_verbosity(1)
plt.rcParams["font.family"] = "Arial"

# Move back one directory
os.chdir('..')

# Define version of MAVEN and cutoff for binary labels
y_percentile = 40 
version = 'v1.0.0'

# Create directories to store results and model
if not os.path.exists('maven_' + version):
    os.mkdir('maven_' + version)
if not os.path.exists('maven_' + version + '/model'):
    os.mkdir('maven_' + version + '/model')
if not os.path.exists('maven_' + version + '/model' + '/percentile_'+str(y_percentile)):
    os.mkdir('maven_' + version + '/model'+'/percentile_'+str(y_percentile))
if not os.path.exists('maven_' + version + '/figures/'):
    os.mkdir('maven_' + version + '/figures/')
if not os.path.exists('maven_' + version + '/figures'+'/percentile_'+str(y_percentile)):
    os.mkdir('maven_' + version + '/figures'+'/percentile_'+str(y_percentile))
if not os.path.exists('maven_' + version + '/figures'+'/percentile_'+ str(y_percentile)+'/classifier/'):
    os.mkdir('maven_' + version + '/figures'+'/percentile_'+ str(y_percentile)+'/classifier/')
if not os.path.exists('maven_' + version + '/figures'+'/percentile_'+ str(y_percentile)+'/classifier/hyperparameter_tuning/'):
    os.mkdir('maven_' + version + '/figures'+'/percentile_'+str(y_percentile)+'/classifier/hyperparameter_tuning/')
if not os.path.exists('maven_' + version + '/figures'+'/percentile_'+ str(y_percentile)+'/classifier/training_data/'):
    os.mkdir('maven_' + version + '/figures'+'/percentile_'+ str(y_percentile)+'/classifier/training_data/')
if not os.path.exists('maven_' + version + '/data/'):
    os.mkdir('maven_' + version + '/data/')
if not os.path.exists('maven_' + version + '/data/'+ '/percentile_'+str(y_percentile)):
    os.mkdir('maven_' + version + '/data/'+ '/percentile_'+str(y_percentile))
if not os.path.exists('maven_' + version + '/model' + '/percentile_'+str(y_percentile)+'/ensembl/'):
    os.mkdir('maven_' + version + '/model'+'/percentile_'+str(y_percentile)+'/ensembl/')
if not os.path.exists('maven_' + version + '/model' + '/percentile_'+str(y_percentile)+'/hyperparameter_search/'):
    os.mkdir('maven_' + version + '/model'+'/percentile_'+str(y_percentile)+'/hyperparameter_search/')


################################################################################################################
################################################################################################################
################################################################################################################
################################################# Train MAVEN ##################################################
################################################################################################################
################################################################################################################
################################################################################################################

########################################################
########################################################
########################################################
#########  1. Load MAVEN Training Data 
########################################################
########################################################
########################################################

# Cutoff to use for binary label
label_col = 'MAVE_score_binary_' + str(y_percentile) + '_percentile'

# Get all features (numeric and categorical) and feature order
categorical_cols = PS.file_to_list('training_data/categorical_features.txt')
numeric_cols = PS.file_to_list('training_data/numeric_features.txt')
all_features = PS.file_to_list('training_data/maven_all_features_order.txt')

# Import training data X,y
X = pd.read_csv('training_data/X_training_data_all.csv',low_memory=False).set_index(['ProteinChange', 'gene','EnsT', 'aapos', 'aaref', 'aaalt', 'Uniprot_id','ClinicalSignificance', 'ClinVar Variant Category', 'Stars','norm_raw_score','Uniprot+aapos'])[all_features]
y = pd.read_csv('training_data/y_training_data_all.csv',low_memory=False).set_index(['ProteinChange', 'gene','EnsT', 'aapos', 'aaref', 'aaalt', 'Uniprot_id','ClinicalSignificance', 'ClinVar Variant Category', 'Stars','norm_raw_score','Uniprot+aapos'])

# Print number of training genes
training_genes = np.unique(X.index.get_level_values('gene'))
print(str(len(training_genes)) + ' training genes')

########################################################
########################################################
########################################################
#########  2. Split training data for 5-fold CV
########################################################
########################################################
########################################################
# For each gene, randomly divide protein positions into 5 folds and assign variants to each fold based on posiiton (all variants at the same position will be assigned to the same fold)
cv_splits = {}
cv_data_x = {}
cv_data_y = {}

for gene in training_genes: 

    cv_splits[gene] = []
    X_gene = X[X.index.get_level_values('gene')==gene]
    y_gene = y.loc[X_gene.index]
    cv_data_x[gene] = X_gene
    cv_data_y[gene] = y_gene
    all_pos = np.unique(X_gene.index.get_level_values('aapos'))
    random.Random(50).shuffle(all_pos)
    pos_split = np.array_split(all_pos,5)
    
    for i in range(0,len(pos_split)):
        
        other_splits = [x for x in range(0,len(pos_split)) if x != i]
        val_idx = X_gene.index.get_indexer_for((X_gene[X_gene.index.get_level_values('aapos').isin(pos_split[i])].index))

        train_split = []
        for split in other_splits:
            train_split.append(pos_split[split])
        train_split = list(chain.from_iterable(train_split))
        train_idx = X_gene.index.get_indexer_for((X_gene[X_gene.index.get_level_values('aapos').isin(train_split)].index))
        cv_splits[gene].append((train_idx, val_idx))

########################################################
########################################################
########################################################
#########  3. Hyperparameter Search with Optuna
########################################################
########################################################
########################################################
# Hyperparameter search using Optuna and 5-fold CV
def hyperparameter_search_all(trial):

    classifier = trial.suggest_categorical('classifier', ['LogisticRegression','RandomForestClassifier','SVC','XGBClassifier','GaussianNB'])


    if classifier == 'XGBClassifier':

        n_estimators = trial.suggest_int('n_estimators',100,1000,step=10)
        max_depth = trial.suggest_int('max_depth',3,10,step=1)
        min_child_weight = trial.suggest_int('min_child_weight',2, 50,step=1)
        eta = trial.suggest_categorical("eta", [1e-3, 0.01, 0.1])
        subsample = trial.suggest_float('subsample',0.2,1,step=0.1)
        colsample_bytree= trial.suggest_float('colsample_bytree',0.2,1,step=0.1)
        gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
        grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        model = xgb.XGBClassifier(n_estimators=n_estimators,grow_policy=grow_policy,max_depth=max_depth,subsample=subsample,colsample_bytree=colsample_bytree,min_child_weight=min_child_weight,gamma=gamma,eta=eta,reg_lambda=reg_lambda,reg_alpha=reg_alpha,objective='binary:logistic',eval_metric='logloss')
        pipeline = Pipeline(steps=[('model', model)])
        score = model_selection.cross_val_score(pipeline, X_gene, y_gene, cv = cv_split, n_jobs=-1,scoring = 'neg_log_loss') 
        score = score.mean()
    

    if classifier == 'RandomForestClassifier':

        n_estimators = trial.suggest_int('n_estimators',100,1000,step=10)
        max_depth = trial.suggest_int('max_depth',3,10,step=1)
        min_samples_split = trial.suggest_int('min_samples_split',2,50,step=1)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',1,50,step=1)
        max_features = trial.suggest_categorical("max_features", ['sqrt', 'log2'])
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    
        model =  RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=max_features,criterion=criterion)
        numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True))])
        categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True))])
        imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
        pipeline = Pipeline(steps=[('impute',imputation),('model', model)])
        score = model_selection.cross_val_score(pipeline, X_gene, y_gene, cv = cv_split, n_jobs=-1,scoring = 'neg_log_loss') 
        score = score.mean()
    

    if classifier == 'LogisticRegression':

        C = trial.suggest_loguniform('C', 1e-4, 1e4)
        max_iter = trial.suggest_int('max_iter', 1, 1000, log=False)
        l1_ratio = trial.suggest_float('l1_ratio', 0, 1, log=False)
        solver  = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'sag', 'saga','newton-cg','newton-cholesky'])
        if solver == 'lbfgs':
            penalty = trial.suggest_categorical('lbfgs', ['l2', None])
        elif solver == 'liblinear':
            penalty = trial.suggest_categorical('liblinear', ['l1', 'l2'])
        elif solver == 'sag':
            penalty = trial.suggest_categorical('sag', ['l2', None])
        elif solver == 'newton-cg':
            penalty = trial.suggest_categorical('newton-cg', ['l2', None])
        elif solver == 'newton-cholesky':
            penalty = trial.suggest_categorical('newton-cholesky', ['l2', None])
        else: 
            penalty = trial.suggest_categorical('saga', ['elasticnet', 'l1', 'l2', None])    

        model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter,l1_ratio=l1_ratio)
        numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True)),('scale', MinMaxScaler())])
        categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True))])
        imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
        feature_selection = SelectKBest(k=20)
        pipeline = Pipeline(steps=[('impute',imputation),('feature_selection',feature_selection),('model', model)])
        score = model_selection.cross_val_score(pipeline, X_gene, y_gene, cv = cv_split, n_jobs=-1,scoring = 'neg_log_loss') 
        score = score.mean()

    
    if classifier == 'SVC':

        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        C = trial.suggest_loguniform('C', 1e-4, 1e4)

        model = SVC(kernel=kernel, C=C, probability=True)
        numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True)),('scale', MinMaxScaler())])
        categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True))])
        imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
        feature_selection = SelectKBest(k=20)
        pipeline = Pipeline(steps=[('impute',imputation),('feature_selection',feature_selection),('model', model)])
        score = model_selection.cross_val_score(pipeline, X_gene, y_gene, cv = cv_split, n_jobs=-1,scoring = 'neg_log_loss') 
        score = score.mean()
    

    if classifier == 'GaussianNB':
        
        var_smoothing =  trial.suggest_categorical('var_smoothing', np.logspace(0,-9, num=100)) 

        model = GaussianNB(var_smoothing=var_smoothing)
        numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True)),('scale', MinMaxScaler())])
        categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True))])
        imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
        feature_selection = SelectKBest(k=20)
        pipeline = Pipeline(steps=[('impute',imputation),('feature_selection',feature_selection),('model', model)])
        score = model_selection.cross_val_score(pipeline, X_gene, y_gene, cv = cv_split, n_jobs=-1,scoring = 'neg_log_loss')
        score = score.mean()
    
    
    return(score)  

# Run Hyperparameter search 
print('Training protein-specific models...')
for i,gene in enumerate(training_genes):

    cv_split = cv_splits[gene]
    X_gene = cv_data_x[gene]
    y_gene = cv_data_y[gene]

    if not os.path.exists('maven_' + version + '/data' + '/percentile_'+str(y_percentile) + '/val_y_'+gene+'_split' + str(len(cv_split)-1)+'.csv'):
        for split in range(0,len(cv_split)):
            train_x = X_gene.iloc[cv_split[split][0]]
            val_x = X_gene.iloc[cv_split[split][1]]
            train_y = y_gene.iloc[cv_split[split][0]]
            val_y = y_gene.iloc[cv_split[split][1]]
            train_x.to_csv('maven_' + version + '/data' + '/percentile_'+str(y_percentile) +'/train_x_'+gene+'_split' + str(split)+'.csv')
            train_y.to_csv('maven_' + version + '/data' + '/percentile_'+str(y_percentile) + '/train_y_'+gene+'_split' + str(split)+'.csv')
            val_x.to_csv('maven_' + version + '/data' + '/percentile_'+str(y_percentile) + '/val_x_'+gene+'_split' + str(split)+'.csv')
            val_y.to_csv('maven_' + version + '/data' + '/percentile_'+str(y_percentile) + '/val_y_'+gene+'_split' + str(split)+'.csv')

    print('#########################################')
    print('#########################################')
    print(str(i+1) + '/' + str(len(training_genes)) + ' ' + gene)
    print('#########################################')
    print('#########################################')

    y_gene = np.array(y_gene[label_col])
    
    # Hyperparameter search 
    if not os.path.exists('maven_' + version + '/model' + '/percentile_'+str(y_percentile)+'/hyperparameter_search/'+"maven_optuna_20240610_"+gene+'.db'):
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = 'maven_' + version + '/model' + '/percentile_'+str(y_percentile)+'/hyperparameter_search/'+"maven_optuna_20240610_"+gene
        storage_name = "sqlite:///{}.db".format(study_name)
        search = optuna.create_study(direction='maximize',study_name=study_name, storage=storage_name, load_if_exists=True)
        search.optimize(hyperparameter_search_all, n_trials=500) # 500 trials for each gene

########################################################
########################################################
########################################################
#########  4. Assess performance of best model and hyperparameters using 5-fold CV
########################################################
########################################################
########################################################

# Train a model for each fold
ensembl = {}
split_train = {}
gene_2_classifier = {}

for gene in training_genes:

    ensembl[gene] = {}

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'maven_' + version + '/model' + '/percentile_'+str(y_percentile)+'/hyperparameter_search/'+"maven_optuna_20240610_"+gene
    storage_name = "sqlite:///{}.db".format(study_name)
    search = optuna.create_study(direction='maximize',study_name=study_name, storage=storage_name, load_if_exists=True)
    classifier = search.best_params['classifier']
    gene_2_classifier[gene]= classifier

    for kfold in range(0,5):

    
        train_x = pd.read_csv('maven_' + version +'/data' + '/percentile_'+str(y_percentile)+ '/train_x_'+gene+'_split'+str(kfold)+'.csv',low_memory=False).set_index(['ProteinChange', 'gene','EnsT', 'aapos', 'aaref', 'aaalt', 'Uniprot_id','ClinicalSignificance', 'ClinVar Variant Category', 'Stars','norm_raw_score','Uniprot+aapos'])[all_features]
        train_y = pd.read_csv('maven_' + version +'/data' + '/percentile_'+str(y_percentile)+ '/train_y_'+gene+'_split'+str(kfold)+'.csv',low_memory=False).set_index(['ProteinChange', 'gene','EnsT', 'aapos', 'aaref', 'aaalt', 'Uniprot_id','ClinicalSignificance', 'ClinVar Variant Category', 'Stars','norm_raw_score','Uniprot+aapos'])
        split_train[kfold]=(train_x,train_y)

        if not os.path.exists('maven_' + version + '/model' + '/percentile_'+str(y_percentile)+'/ensembl/'+'maven_'+gene+'_split'+str(kfold)+'.pkl'):
                  
            if classifier == 'RandomForestClassifier':
                
                model =  RandomForestClassifier(n_estimators=search.best_params['n_estimators'],max_depth=search.best_params['max_depth'],min_samples_split=search.best_params['min_samples_split'],min_samples_leaf=search.best_params['min_samples_leaf'],max_features=search.best_params['max_features'],criterion=search.best_params['criterion'])
                numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True))])
                categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True))])
                imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
                pipeline = Pipeline(steps=[('impute',imputation),('model', model)])
                pipeline.fit(train_x, train_y)

            if classifier == 'XGBClassifier':

                model = xgb.XGBClassifier(n_estimators=search.best_params['n_estimators'],grow_policy=search.best_params['grow_policy'],max_depth=search.best_params['max_depth'],subsample=search.best_params['subsample'],colsample_bytree=search.best_params['colsample_bytree'],min_child_weight=search.best_params['min_child_weight'],gamma=search.best_params['gamma'],eta=search.best_params['eta'],reg_lambda=search.best_params['reg_lambda'],reg_alpha=search.best_params['reg_alpha'],objective='binary:logistic',eval_metric='logloss')
                pipeline = Pipeline(steps=[('model', model)])
                pipeline.fit(train_x, train_y)

            
            if classifier == 'LogisticRegression':
                
                solver = search.best_params['solver']  

                model = LogisticRegression(penalty=search.best_params[solver], C=search.best_params['C'], solver=solver, max_iter=search.best_params['max_iter'],l1_ratio=search.best_params['l1_ratio'])
                numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True)),('scale', MinMaxScaler())])
                categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True))])
                imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
                feature_selection = SelectKBest(k=20)
                pipeline = Pipeline(steps=[('impute',imputation),('feature_selection',feature_selection),('model', model)])
                pipeline.fit(train_x, train_y)
            
            if classifier == 'SVC':

                model = SVC(kernel=search.best_params['kernel'], C=search.best_params['C'], probability=True)
                numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True)),('scale', MinMaxScaler())])
                categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True))])
                imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
                feature_selection = SelectKBest(k=20)
                pipeline = Pipeline(steps=[('impute',imputation),('feature_selection',feature_selection),('model', model)])
                pipeline.fit(train_x, train_y)
            
            if classifier == 'GaussianNB':

                model = GaussianNB(var_smoothing=search.best_params['var_smoothing'])
                numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True)),('scale', MinMaxScaler())])
                categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True))])
                imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
                feature_selection = SelectKBest(k=20)
                pipeline = Pipeline(steps=[('impute',imputation),('feature_selection',feature_selection),('model', model)])
                pipeline.fit(train_x, train_y)

            ensembl[gene][kfold] = pipeline

            with open('maven_' + version + '/model' + '/percentile_'+str(y_percentile)+'/ensembl/'+'maven_'+gene+'_split'+str(kfold)+'.pkl', "wb") as f:
                pickle.dump(pipeline, f)
        else:
            pipeline = pickle.load(open('maven_' + version + '/model' + '/percentile_'+str(y_percentile)+'/ensembl/'+'maven_'+gene+'_split'+str(kfold)+'.pkl', 'rb'))
            ensembl[gene][kfold] = pipeline


# Save the best model for each gene
best_model = pd.DataFrame.from_dict(gene_2_classifier, orient='index').reset_index()
best_model.columns = ['gene','Best Model']
best_model.to_csv('maven_' + version +'/data' + '/percentile_'+str(y_percentile)+'/maven_hyperparameter_search_best_models.csv')
print('Best model for each protein:')
print(best_model)


# Assess performance of each model on 5-fold CV
print('Assessing performance of each protein-specific model using 5-fold CV...')
predictions_lofo = {}
predictions_training = {}
pr_curves = {}
pr_curves_training = {}
auc_curves = {}
auc_curves_training = {}
mean_auc_curves = {}
mean_pr_curves = {}
mean_auc_genes = {}
mean_pr_genes = {}

colors = ['red','blue','pink','orange','green'] 

for gene in training_genes:

    fig, ax = plt.subplots(1,4, figsize=(40,8))

    predictions_lofo[gene] = {}
    predictions_training[gene] = {}
    pr_curves[gene] = {}
    pr_curves_training[gene] = {}
    auc_curves[gene] = {}
    auc_curves_training[gene]= {}
    mean_auc_curves[gene] ={}
    mean_pr_curves[gene] ={}

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    mean_precision = 0
    mean_recall = np.linspace(1, 0, 100)


    for kfold in range(0,5):

        X_train = pd.read_csv('maven_' + version + '/data' + '/percentile_'+str(y_percentile) + '/train_x_'+gene+'_split'+str(kfold)+'.csv',low_memory=False).set_index(['ProteinChange', 'gene','EnsT', 'aapos', 'aaref', 'aaalt', 'Uniprot_id','ClinicalSignificance', 'ClinVar Variant Category', 'Stars','norm_raw_score'])[all_features]
        X_test = pd.read_csv('maven_' + version + '/data' + '/percentile_'+str(y_percentile) +'/val_x_'+gene+'_split'+str(kfold)+'.csv',low_memory=False).set_index(['ProteinChange', 'gene','EnsT', 'aapos', 'aaref', 'aaalt', 'Uniprot_id','ClinicalSignificance', 'ClinVar Variant Category', 'Stars','norm_raw_score'])[all_features]
        y_train = pd.read_csv('maven_' + version + '/data' + '/percentile_'+str(y_percentile) +'/train_y_'+gene+'_split'+str(kfold)+'.csv',low_memory=False).set_index(['ProteinChange', 'gene','EnsT', 'aapos', 'aaref', 'aaalt', 'Uniprot_id','ClinicalSignificance', 'ClinVar Variant Category', 'Stars','norm_raw_score'])
        y_test = pd.read_csv('maven_' + version + '/data' + '/percentile_'+str(y_percentile) +'/val_y_'+gene+'_split'+str(kfold)+'.csv',low_memory=False).set_index(['ProteinChange', 'gene','EnsT', 'aapos', 'aaref', 'aaalt', 'Uniprot_id','ClinicalSignificance', 'ClinVar Variant Category', 'Stars','norm_raw_score'])

        model_gene = ensembl[gene][kfold]

        pred = model_gene.predict(X_test)
        pred_prob = [x[1] for x in model_gene.predict_proba(X_test)]
        pred_df = pd.DataFrame({label_col:list(y_test.loc[:,label_col]), 'Predicted Label': pred,'Predicted Probability': pred_prob})
        pred_df.index = X_test.index
        pred_df = pred_df.reset_index()
        predictions_lofo[gene] = pred_df

        
        accuracy = accuracy_score(pred_df[label_col],pred_df['Predicted Label'])
        auc_test = roc_auc_score(pred_df[label_col],pred_df['Predicted Probability'])
        precision_s = precision_score(pred_df[label_col],pred_df['Predicted Label'])
        recall_s = recall_score(pred_df[label_col],pred_df['Predicted Label'])
        precision, recall, thresholds = precision_recall_curve(pred_df[label_col],pred_df['Predicted Probability'])
        mean_precision += np.interp(mean_recall, precision, recall)
        mean_precision[0] = 1
        auc_precision_recall_test = auc(recall, precision)
        pr_curves[gene][kfold] = [precision, recall, thresholds]
        fpr, tpr, thr = roc_curve(pred_df[label_col],pred_df['Predicted Probability'], pos_label=1)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        auc_curves[gene][kfold] = [fpr, tpr, thr ]


        # Make predictions for training data 
        pred = model_gene.predict(X_train)
        pred_prob = [x[1] for x in model_gene.predict_proba(X_train)]
        pred_df = pd.DataFrame({label_col:list(y_train.loc[:,label_col]), 'Predicted Label': pred,'Predicted Probability': pred_prob})
        pred_df.index = X_train.index
        pred_df = pred_df.reset_index()
        predictions_training[gene][kfold] = pred_df

        accuracy = accuracy_score(pred_df[label_col],pred_df['Predicted Label'])
        auc_train = roc_auc_score(pred_df[label_col],pred_df['Predicted Probability'])
        precision_s = precision_score(pred_df[label_col],pred_df['Predicted Label'])
        recall_s = recall_score(pred_df[label_col],pred_df['Predicted Label'])
        precision, recall, thresholds = precision_recall_curve(pred_df[label_col],pred_df['Predicted Probability'])
        auc_precision_recall_train = auc(recall, precision)
        pr_curves_training[gene][kfold] = [precision, recall, thresholds]
        fpr, tpr, thr = roc_curve(pred_df[label_col],pred_df['Predicted Probability'], pos_label=1)
        auc_curves_training[gene][kfold] = [fpr, tpr, thr ]


        ax[0].plot(pr_curves[gene][kfold][1],pr_curves[gene][kfold][0],label='Fold ' + str(kfold+1) +' (AUC='+str(round(auc_precision_recall_test ,3))+')',color =colors[kfold])
        ax[0].set_xlabel('Recall', fontsize=25)
        ax[0].set_ylabel('Precision',fontsize=25)
        ax[0].set_title(gene + ' Validation Fold',fontsize=30)
        ax[0].tick_params(axis='both',labelsize=20)
        ax[0].legend(loc="lower left",fontsize=20)
        ax[0].set_ylim([-0.1,1.1])

        ax[1].plot(pr_curves_training[gene][kfold][1],pr_curves_training[gene][kfold][0],label='Fold ' + str(kfold+1) +' (AUC='+str(round(auc_precision_recall_train ,3))+')',color =colors[kfold]) 
        ax[1].set_xlabel('Recall', fontsize=25)
        ax[1].set_ylabel('Precision',fontsize=25)
        ax[1].set_title(gene + ' Training Folds',fontsize=30)
        ax[1].tick_params(axis='both',labelsize=20)
        ax[1].legend(loc="lower left",fontsize=20)
        ax[1].set_ylim([-0.1,1.1])

        ax[2].plot(auc_curves[gene][kfold][0],auc_curves[gene][kfold][1],label='Fold ' + str(kfold+1) +' (AUC='+str(round(auc_test,3))+')',color = colors[kfold],lw=2,linestyle='-')
        ax[2].set_xlabel('False Positive Rate', fontsize=25)
        ax[2].set_ylabel('True Positive Rate',fontsize=25)
        ax[2].set_title(gene + ' Validation Fold',fontsize=30)
        ax[2].tick_params(axis='both',labelsize=20)
        ax[2].legend(loc="lower right",fontsize=20)
        ax[2].set_ylim([-0.1,1.1])

        ax[3].plot(auc_curves_training[gene][kfold][0],auc_curves_training[gene][kfold][1],label='Fold ' + str(kfold+1) +' (AUC='+str(round(auc_train,3))+')',color = colors[kfold],lw=2,linestyle='-')
        ax[3].set_xlabel('False Positive Rate', fontsize=25)
        ax[3].set_ylabel('True Positive Rate',fontsize=25)
        ax[3].set_title(gene + ' Training Folds',fontsize=30)
        ax[3].tick_params(axis='both',labelsize=20)
        ax[3].legend(loc="lower right",fontsize=20)
        ax[3].set_ylim([-0.1,1.1])
    
    # Calc mean AUC ROC AND AUC PR
    mean_tpr /= 5
    mean_precision /= 5
    mean_tpr[-1] = 1.0
    mean_precision[-1] = 1
    mean_auc_curves[gene] = (mean_fpr, mean_tpr)
    mean_pr_curves[gene] = (mean_recall, mean_precision)
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_prauc = auc(mean_recall, mean_precision)
    mean_auc_genes[gene] = mean_auc
    mean_pr_genes[gene] = mean_prauc 


    ax[0].plot(mean_pr_curves[gene][1],mean_pr_curves[gene][0],label='Mean (AUC='+str(round(mean_prauc ,3))+')',color ='black',linestyle='--',lw=2)
    ax[0].legend(loc="lower left",fontsize=20)
    ax[2].plot(mean_auc_curves[gene][0],mean_auc_curves[gene][1],label='Mean (AUC='+str(round(mean_auc,3))+')',color = 'black',lw=2,linestyle='--')
    ax[2].legend(loc="lower right",fontsize=20)
    plt.savefig('maven_' + version + '/figures' + '/percentile_'+str(y_percentile) + '/classifier/hyperparameter_tuning/'+gene+'_5foldCV_best_hyperparameters.png',dpi=300,bbox_inches='tight')
    plt.savefig('maven_' + version + '/figures' + '/percentile_'+str(y_percentile) + '/classifier/hyperparameter_tuning/'+gene+'_5foldCV_best_hyperparameters.svg',bbox_inches='tight')
    plt.close()


# Plot all leave-one-gene-out CV results for the classification model
fig, ax = plt.subplots(1,2, figsize=(21,9))
colors = sns.hls_palette(len(np.unique(list(X.index.get_level_values('gene')))),h=0.1)

auc_df = pd.DataFrame.from_dict(mean_auc_genes, orient='index')
auc_df.columns = ['AUCROC']
pr_df = pd.DataFrame.from_dict(mean_pr_genes, orient='index')
pr_df.columns = ['AUCPR']
logo_df = auc_df.merge(pr_df, how='left',right_index=True,left_index=True)
logo_df = logo_df.reset_index()
logo_df.columns = ['Gene','ROC_AUC','PR_AUC']
logo_df = logo_df.sort_values(by=['ROC_AUC','Gene'],ascending=[False,True]).reset_index(drop=True)

for i,gene in enumerate(list(logo_df['Gene'])):
    ax[0].plot(mean_auc_curves[gene][0],mean_auc_curves[gene][1],label= gene + ' (AUC='+str(round(mean_auc_genes[gene],3))+')',color = colors[i],lw=1.5,linestyle='-')  
logo_df = logo_df.sort_values(by=['PR_AUC','Gene'],ascending=[False,True]).reset_index(drop=True)

for i,gene in enumerate(list(logo_df['Gene'])):
    ax[1].plot(mean_pr_curves[gene][1],mean_pr_curves[gene][0],label=gene + ' (AUC='+str(round(mean_pr_genes[gene] ,3))+')',color = colors[i],lw=1.5,linestyle='-')
   
ax[0].set_xlabel('False Positive Rate', fontsize=20)
ax[0].set_ylabel('True Positive Rate',fontsize=20)
ax[0].set_title('Mean AUC ROC 5-Fold Cross-Validation',fontsize=25)
ax[0].tick_params(axis='both',labelsize=15)
ax[0].legend(loc="lower right",fontsize=13,ncol=2)  

ax[1].set_xlabel('Recall', fontsize=20)
ax[1].set_ylabel('Precision',fontsize=20)
ax[1].set_title('Mean AUC PR 5-Fold Cross-Validation',fontsize=25)
ax[1].tick_params(axis='both',labelsize=15)
ax[1].set_ylim([-0.4,1.05])
ax[1].legend(loc="lower left",fontsize=13,ncol=2)
plt.subplots_adjust(wspace=0.15, hspace=0)
plt.savefig('maven_' + version + '/figures' + '/percentile_'+str(y_percentile) + '/classifier/hyperparameter_tuning/'+'All_genes_mean_5foldCV_best_hyperparameters.png',dpi=300,bbox_inches='tight')
plt.savefig('maven_' + version + '/figures' + '/percentile_'+str(y_percentile) + '/classifier/hyperparameter_tuning/'+'All_genes_mean_5foldCV_best_hyperparameters.svg',bbox_inches='tight')
plt.close()

# Save results of 5-fold CV
logo_df.to_csv('maven_' + version + '/data' +  '/percentile_'+str(y_percentile)+ '/maven_training_5foldCV_performance_by_gene.csv')


########################################################
########################################################
########################################################
#########  5. Train protein-specific models
########################################################
########################################################
########################################################
# Train a model for each protein using the best model and best selected hyperparameters
print('Training final protein-specific models...')
ensembl_final = {}

for gene in training_genes:

        train_x = cv_data_x[gene]
        train_y = cv_data_y[gene]

        val_x = X[X.index.get_level_values('gene')!=gene] # Use data from all other genes for early stopping 
        val_y = y.loc[val_x.index]

        if not os.path.exists('maven_' + version + '/model' +  '/percentile_'+str(y_percentile)+ '/ensembl/maven_'+gene+'.pkl'):

            optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            study_name = 'maven_' + version + '/model' + '/percentile_'+str(y_percentile)+'/hyperparameter_search/'+"maven_optuna_20240610_"+gene
            storage_name = "sqlite:///{}.db".format(study_name)
            search = optuna.create_study(direction='maximize',study_name=study_name, storage=storage_name, load_if_exists=True)

            classifier = search.best_params['classifier']
            
            if classifier == 'RandomForestClassifier':
                
                model =  RandomForestClassifier(n_estimators=search.best_params['n_estimators'],max_depth=search.best_params['max_depth'],min_samples_split=search.best_params['min_samples_split'],min_samples_leaf=search.best_params['min_samples_leaf'],max_features=search.best_params['max_features'],criterion=search.best_params['criterion'])
                numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True))])
                categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True,missing_values=pd.NA))])
                imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
                pipeline = Pipeline(steps=[('impute',imputation),('model', model)])
                pipeline.fit(train_x, train_y)

            if classifier == 'XGBClassifier':

                model = xgb.XGBClassifier(n_estimators=search.best_params['n_estimators'],grow_policy=search.best_params['grow_policy'],max_depth=search.best_params['max_depth'],subsample=search.best_params['subsample'],colsample_bytree=search.best_params['colsample_bytree'],min_child_weight=search.best_params['min_child_weight'],gamma=search.best_params['gamma'],eta=search.best_params['eta'],reg_lambda=search.best_params['reg_lambda'],reg_alpha=search.best_params['reg_alpha'],objective='binary:logistic',eval_metric='logloss')
                pipeline = Pipeline(steps=[('model', model)])
                pipeline.fit(train_x, train_y)

            
            if classifier == 'LogisticRegression':
                
                solver = search.best_params['solver']  

                model = LogisticRegression(penalty=search.best_params[solver], C=search.best_params['C'], solver=solver, max_iter=search.best_params['max_iter'],l1_ratio=search.best_params['l1_ratio'])
                numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True)),('scale', MinMaxScaler())])
                categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True,missing_values=pd.NA))])
                imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
                feature_selection = SelectKBest(k=20)
                pipeline = Pipeline(steps=[('impute',imputation),('feature_selection',feature_selection),('model', model)])
                pipeline.fit(train_x, train_y)
            
            if classifier == 'SVC':

                model = SVC(kernel=search.best_params['kernel'], C=search.best_params['C'], probability=True)
                numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True)),('scale', MinMaxScaler())])
                categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True,missing_values=pd.NA))])
                imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
                feature_selection = SelectKBest(k=20)
                pipeline = Pipeline(steps=[('impute',imputation),('feature_selection',feature_selection),('model', model)])
                pipeline.fit(train_x, train_y)
            
            if classifier == 'GaussianNB':

                model = GaussianNB(var_smoothing=search.best_params['var_smoothing'])
                numeric_impute = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5,keep_empty_features=True)),('scale', MinMaxScaler())])
                categorical_impute = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True,missing_values=pd.NA))])
                imputation = ColumnTransformer(transformers=[('numerical', numeric_impute, numeric_cols),('categorical', categorical_impute, categorical_cols)])
                feature_selection = SelectKBest(k=20)
                pipeline = Pipeline(steps=[('impute',imputation),('feature_selection',feature_selection),('model', model)])
                pipeline.fit(train_x, train_y)

            ensembl_final[gene] = pipeline

            with open('maven_' + version + '/model' +  '/percentile_'+str(y_percentile)+ '/ensembl/maven_'+gene+'.pkl', "wb") as f:
                pickle.dump(pipeline, f)
        else:
            pipeline = pickle.load(open('maven_' + version + '/model' +  '/percentile_'+str(y_percentile)+ '/ensembl/maven_'+gene+'.pkl', 'rb'))
            ensembl_final[gene] = pipeline

########################################################
########################################################
########################################################
#########  6. Train meta classifier
########################################################
########################################################
########################################################

y_label = 'MAVE_score_binary_'+str(y_percentile) + '_percentile'

meta = pd.read_csv('training_data/metadata.csv',sep=',',header=0)
meta = meta.dropna(axis=1, how='all').dropna(axis=0, how='all') # drop all rows and columns with all NAs
meta = meta[meta['LOF']==1]
meta = meta.sort_values(by='Approved Symbol').reset_index(drop=True)
gene_2_assay = dict(zip(list(meta['Approved Symbol']),list(meta['Type'])))

train_cv_performance = pd.read_csv('maven_' + version + '/data' +  '/percentile_'+str(y_percentile)+ '/maven_training_5foldCV_performance_by_gene.csv',index_col=0)
train_cv_performance = train_cv_performance[(train_cv_performance['ROC_AUC']>=0.9)&(train_cv_performance['PR_AUC']>=0.9)]
train_cv_performance['Assay_type'] = [gene_2_assay[x] for x in train_cv_performance['Gene']]
training_genes = np.unique(train_cv_performance['Gene'])

train_cv_performance = pd.read_csv('maven_' + version + '/data' +  '/percentile_'+str(y_percentile)+ '/maven_training_5foldCV_performance_by_gene.csv',index_col=0)
train_cv_performance = train_cv_performance[(train_cv_performance['ROC_AUC']<0.9)|(train_cv_performance['PR_AUC']<0.9)]
validation_genes = np.unique(train_cv_performance['Gene'])

X = pd.read_csv('training_data/X_training_data_all.csv',low_memory=False).set_index(['ProteinChange', 'gene','EnsT', 'aapos', 'aaref', 'aaalt', 'Uniprot_id','ClinicalSignificance', 'ClinVar Variant Category', 'Stars','norm_raw_score','Uniprot+aapos'])[all_features]
y = pd.read_csv('training_data/y_training_data_all.csv',low_memory=False).set_index(['ProteinChange', 'gene','EnsT', 'aapos', 'aaref', 'aaalt', 'Uniprot_id','ClinicalSignificance', 'ClinVar Variant Category', 'Stars','norm_raw_score','Uniprot+aapos'])

X_val = X[X.index.get_level_values('gene').isin(validation_genes)]
y_val = y.loc[X_val.index]

X = X[~X.index.get_level_values('gene').isin(validation_genes)]
y = y.loc[X.index]


# New MAVE score percentile cuttoff value to train meta classifier
y_label_new = 'MAVE_score_binary_'+str(20) + '_percentile'
y_new_cutoffs = pd.read_csv('training_data/maven_y_data.csv',sep=',',index_col=0,low_memory=False) # get file with all cutoff labels
y_new_cutoffs['Uniprot+aapos'] = y_new_cutoffs['Uniprot_id'] + '_' + y_new_cutoffs['aapos'].astype('str')
y_new_cutoffs = y_new_cutoffs.set_index(['ProteinChange', 'gene','EnsT', 'aapos', 'aaref', 'aaalt', 'Uniprot_id','ClinicalSignificance', 'ClinVar Variant Category', 'Stars','norm_raw_score','Uniprot+aapos'])
y_new_cutoffs = y_new_cutoffs.loc[y_val.index]
y_new_cutoffs = y_new_cutoffs[[y_label_new]]
y_new_cutoffs = y_new_cutoffs[y_new_cutoffs[y_label_new].isin([0,1])]

X_new_cutoffs = X_val.loc[y_new_cutoffs.index]

min_num = 2000
prot_change_keep_all = []
for gene in np.unique(X_new_cutoffs.index.get_level_values('gene')):
    y_new_cutoffs_gene = y_new_cutoffs[y_new_cutoffs.index.get_level_values('gene')==gene]
    all_prot_changes = np.unique(y_new_cutoffs_gene.index.get_level_values('ProteinChange'))
    if len(all_prot_changes)> min_num:
        np.random.seed(9)
        prot_change_keep = np.random.choice(all_prot_changes,min_num,replace=False)
        for x in prot_change_keep:
            prot_change_keep_all.append(x)
    else:
        for x in all_prot_changes:
            prot_change_keep_all.append(x)

   
X_new_cutoffs = X_new_cutoffs[X_new_cutoffs.index.get_level_values('ProteinChange').isin(prot_change_keep_all)]
y_new_cutoffs = y_new_cutoffs.loc[X_new_cutoffs.index]
all_training_genes = np.unique(list(np.unique(X_val.index.get_level_values('gene'))) + list(np.unique(X.index.get_level_values('gene'))))

PS.list_to_file(all_training_genes,'maven_'+version+'/data/percentile_'+str(y_percentile) +'/'+'ensembl_all_training_genes.txt')
PS.list_to_file(training_genes,'maven_'+version+'/data/percentile_'+str(y_percentile) +'/'+'ensembl_base_genes.txt')
PS.list_to_file(validation_genes,'maven_'+version+'/data/percentile_'+str(y_percentile) +'/'+'ensembl_stacking_genes.txt')

print('Training meta classifier...')
print(str(len(training_genes)) + ' protein-specific base models')
print(str(len(validation_genes)) + ' proteins used to train meta classifier')
print('Meta classifier labels by gene:')
print(Counter(X_new_cutoffs.index.get_level_values('gene')))
print('Meta classifier labels by label:')
print(Counter(y_new_cutoffs[y_label_new]))

# Make sure all columns are float or integers
categorical_cols = PS.file_to_list('training_data/categorical_features.txt')
numeric_cols = PS.file_to_list('training_data/numeric_features.txt')
for col in numeric_cols:
    X[col] = X[col].astype(float)
    X_val[col] = X_val[col].astype(float)
    X_new_cutoffs[col] = X_new_cutoffs[col].astype(float)
for col in categorical_cols:
    X[col] = X[col].astype('Int64')
    X_val[col] = X_val[col].astype('Int64')
    X_new_cutoffs[col] = X_new_cutoffs[col].astype('Int64')
for col in categorical_cols:
    try:
        X[col] = X[col].astype('int')
    except:
        pass
    try:
        X_val[col] = X_val[col].astype('int')
    except:
        pass
    try:
        X_new_cutoffs[col] = X_new_cutoffs[col].astype('int')
    except:
        pass

base_models =[]
for gene in training_genes:
    try:
        model = pickle.load(open('maven_' + version + '/model' +  '/percentile_'+str(y_percentile)+ '/ensembl/maven_'+gene+'.pkl', 'rb'))
        base_models.append((gene,model))
    except:
        pass

# Plot number of 1 and 0 labels for each gene in training set
n = []
t = []
g = []
for gene in np.unique(y_new_cutoffs.index.get_level_values('gene')):
    sub = y_new_cutoffs.reset_index()
    sub = sub[sub['gene']==gene]
    c = Counter(sub[y_label_new])
    g.append(gene)
    g.append(gene)
    n.append(c[0])
    t.append('Benign')
    n.append(c[1])
    t.append('Deleterious')
training_plot = pd.DataFrame({'Gene':g,'Type':t,'Count':n})

total_count_plot = training_plot[['Gene','Count']]
total_count_plot = pd.DataFrame(total_count_plot.groupby(['Gene'],dropna=False).agg({ 'Count' :'sum'})).sort_values(by=['Count','Gene'],ascending=[True,True]).reset_index()
gene_order =  np.array(total_count_plot['Gene'])

training_plot = training_plot.set_index('Gene')
training_plot = training_plot.loc[[x for x in gene_order]]
training_plot = training_plot.reset_index()

fig,ax = plt.subplots(figsize=(20,6))
sns.barplot(training_plot, x="Gene", y="Count", hue="Type",palette=['blue','red'],alpha=0.75)
plt.tick_params('both',labelsize=20)
plt.tick_params('x',labelsize=20,rotation=45)
plt.ylabel('Missense Variants',fontsize=25)
plt.xlabel('Protein',fontsize=25)
plt.legend(fontsize=20,loc='upper left')
plt.title('Meta-Classifier Training Labels',fontsize=25)
plt.savefig('maven_' + version + '/figures'+ '/percentile_'+str(y_percentile)+'/classifier/' + 'meta_classifier_training_labels_by_gene.png',dpi=300,bbox_inches='tight')
plt.savefig('maven_' + version + '/figures'+ '/percentile_'+str(y_percentile)+'/classifier/' + 'meta_classifier_training_labels_by_gene.svg',bbox_inches='tight')

# Train MAVEN stacked classifer 
if not os.path.exists('maven_' + version + '/model' +  '/percentile_'+str(y_percentile)+ '/ensembl/maven_stacked_ensembl.pkl'):
    stacked_ensembl = StackingClassifier(estimators=base_models, stack_method='predict_proba', cv = 'prefit', final_estimator=LogisticRegression(solver='lbfgs', max_iter=1000))  
    stacked_ensembl.fit(X_new_cutoffs, y_new_cutoffs.loc[:,y_label_new])
    with open('maven_' + version + '/model' +  '/percentile_'+str(y_percentile)+ '/ensembl/maven_stacked_ensembl.pkl', "wb") as f:
        pickle.dump(stacked_ensembl, f)

print('DONE')