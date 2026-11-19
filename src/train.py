#!/usr/bin/env python3
"""Tune and evaluate rainfall classifiers using temporal CV and a later-period holdout."""
from __future__ import annotations
import argparse, json, platform, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
import joblib
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning

SEED=42
TARGET="RainTomorrow"

def preprocessor(numeric_cols, categorical_cols, *, dense=False, scale=False):
    num_steps=[("impute",SimpleImputer(strategy="median",add_indicator=True))]
    if scale: num_steps.append(("scale",StandardScaler(with_mean=not dense)))
    cat_steps=[("impute",SimpleImputer(strategy="most_frequent")),
               ("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=not dense))]
    return ColumnTransformer([("num",Pipeline(num_steps),numeric_cols),
                              ("cat",Pipeline(cat_steps),categorical_cols)],remainder="drop")

def model_specs(numeric, categorical):
    sparse_prep=lambda scale=False:preprocessor(numeric,categorical,dense=False,scale=scale)
    dense_prep=lambda scale=False:preprocessor(numeric,categorical,dense=True,scale=scale)
    return {
      "K-Nearest Neighbors":(Pipeline([("prep",sparse_prep(True)),("model",KNeighborsClassifier(n_jobs=-1))]),[
        {"model__n_neighbors":[11],"model__weights":["uniform"]},
        {"model__n_neighbors":[21],"model__weights":["uniform"]},
        {"model__n_neighbors":[41],"model__weights":["distance"]},
        {"model__n_neighbors":[61],"model__weights":["distance"]},
        {"model__n_neighbors":[81],"model__weights":["distance"]}]),
      "CatBoost":(Pipeline([("prep",dense_prep()),("model",CatBoostClassifier(verbose=False,loss_function="Logloss",eval_metric="AUC",random_seed=SEED,allow_writing_files=False,thread_count=4,auto_class_weights="Balanced"))]),[
        {"model__iterations":[120],"model__depth":[5],"model__learning_rate":[0.05],"model__l2_leaf_reg":[3]},
        {"model__iterations":[160],"model__depth":[6],"model__learning_rate":[0.05],"model__l2_leaf_reg":[5]},
        {"model__iterations":[180],"model__depth":[7],"model__learning_rate":[0.04],"model__l2_leaf_reg":[5]},
        {"model__iterations":[150],"model__depth":[6],"model__learning_rate":[0.08],"model__l2_leaf_reg":[8]},
        {"model__iterations":[220],"model__depth":[5],"model__learning_rate":[0.07],"model__l2_leaf_reg":[7]}]),
      "Random Forest":(Pipeline([("prep",sparse_prep()),("model",RandomForestClassifier(class_weight="balanced_subsample",n_jobs=-1,random_state=SEED))]),[
        {"model__n_estimators":[150],"model__max_depth":[None],"model__min_samples_leaf":[2],"model__max_features":["sqrt"]},
        {"model__n_estimators":[220],"model__max_depth":[None],"model__min_samples_leaf":[4],"model__max_features":["sqrt"]},
        {"model__n_estimators":[180],"model__max_depth":[18],"model__min_samples_leaf":[2],"model__max_features":[0.7]},
        {"model__n_estimators":[220],"model__max_depth":[24],"model__min_samples_leaf":[5],"model__max_features":["log2"]},
        {"model__n_estimators":[250],"model__max_depth":[None],"model__min_samples_leaf":[6],"model__max_features":[0.7]}]),
      "Decision Tree":(Pipeline([("prep",sparse_prep()),("model",DecisionTreeClassifier(class_weight="balanced",random_state=SEED))]),[
        {"model__criterion":["gini"],"model__max_depth":[5],"model__min_samples_leaf":[20],"model__ccp_alpha":[0.0]},
        {"model__criterion":["entropy"],"model__max_depth":[8],"model__min_samples_leaf":[30],"model__ccp_alpha":[0.0]},
        {"model__criterion":["gini"],"model__max_depth":[12],"model__min_samples_leaf":[50],"model__ccp_alpha":[0.0001]},
        {"model__criterion":["entropy"],"model__max_depth":[None],"model__min_samples_leaf":[60],"model__ccp_alpha":[0.0003]},
        {"model__criterion":["log_loss"],"model__max_depth":[16],"model__min_samples_leaf":[40],"model__ccp_alpha":[0.0005]}]),
      "Linear SVM":(Pipeline([("prep",sparse_prep(True)),("model",LinearSVC(class_weight="balanced",max_iter=6000,random_state=SEED,dual="auto"))]),[
        {"model__C":[0.03]}, {"model__C":[0.1]}, {"model__C":[0.3]}, {"model__C":[1.0]}, {"model__C":[3.0]}]),
      "Logistic Regression":(Pipeline([("prep",sparse_prep(True)),("model",LogisticRegression(class_weight="balanced",max_iter=2500,solver="liblinear",random_state=SEED))]),[
        {"model__C":[0.03],"model__penalty":["l1"]},
        {"model__C":[0.1],"model__penalty":["l2"]},
        {"model__C":[0.3],"model__penalty":["l1"]},
        {"model__C":[1.0],"model__penalty":["l2"]},
        {"model__C":[3.0],"model__penalty":["l2"]}]),
      "Histogram Gradient Boosting":(Pipeline([("prep",dense_prep()),("model",HistGradientBoostingClassifier(random_state=SEED,class_weight="balanced"))]),[
        {"model__learning_rate":[0.05],"model__max_iter":[100],"model__max_leaf_nodes":[15],"model__l2_regularization":[1.0]},
        {"model__learning_rate":[0.08],"model__max_iter":[150],"model__max_leaf_nodes":[31],"model__l2_regularization":[1.0]},
        {"model__learning_rate":[0.05],"model__max_iter":[180],"model__max_leaf_nodes":[31],"model__l2_regularization":[3.0]},
        {"model__learning_rate":[0.1],"model__max_iter":[120],"model__max_leaf_nodes":[23],"model__l2_regularization":[2.0]},
        {"model__learning_rate":[0.07],"model__max_iter":[200],"model__max_leaf_nodes":[15],"model__l2_regularization":[5.0]}])
    }

def score_vector(estimator,X):
    if hasattr(estimator,"predict_proba"): return estimator.predict_proba(X)[:,1]
    return estimator.decision_function(X)

def make_date_grouped_cv(dates, n_splits=5):
    """Expanding-window folds with no calendar date shared across partitions."""
    values=pd.to_datetime(pd.Series(dates)).drop_duplicates().sort_values().to_numpy()
    blocks=np.array_split(values,n_splits+1)
    folds=[]
    all_dates=pd.to_datetime(pd.Series(dates)).to_numpy()
    for i in range(1,n_splits+1):
        train_dates=np.concatenate(blocks[:i])
        valid_dates=blocks[i]
        train_idx=np.flatnonzero(np.isin(all_dates,train_dates))
        valid_idx=np.flatnonzero(np.isin(all_dates,valid_dates))
        if not len(train_idx) or not len(valid_idx):
            raise ValueError("Date-grouped CV generated an empty fold")
        folds.append((train_idx,valid_idx))
    return folds

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",default="cleaned_data/weatherAUS_clean.csv")
    ap.add_argument("--output-dir",default="outputs/modeling")
    ap.add_argument("--test-fraction",type=float,default=.20)
    ap.add_argument("--cv-rows",type=int,default=8000,help="Maximum number of regularly spaced chronological training rows used for CV tuning.")
    ap.add_argument("--knn-fit-rows",type=int,default=20000,help="Maximum chronological training sample size for final KNN refit.")
    args=ap.parse_args(); data_path=Path(args.data); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(data_path,parse_dates=["Date"])
    if TARGET not in df: raise ValueError(f"Missing {TARGET}")
    df[TARGET]=df[TARGET].map({"No":0,"Yes":1})
    df=df.dropna(subset=["Date",TARGET]).sort_values(["Date","Location"],kind="stable").reset_index(drop=True)
    feature_cols=[c for c in df.columns if c not in ["Date",TARGET]]
    numeric=[c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical=[c for c in feature_cols if c not in numeric]
    dates=np.array(sorted(df.Date.unique())); cutpos=int(np.floor((1-args.test_fraction)*len(dates))); cutpos=min(max(cutpos,1),len(dates)-1); cutoff=pd.Timestamp(dates[cutpos-1])
    train_mask=df.Date<=cutoff; test_mask=df.Date>cutoff
    train,test=df.loc[train_mask].copy(),df.loc[test_mask].copy()
    y_train=train[TARGET].astype(int); y_test=test[TARGET].astype(int)
    X_train=train[feature_cols]; X_test=test[feature_cols]
    # CV uses a deterministic evenly spaced subset, kept in chronological order.
    take=min(args.cv_rows,len(train)); idx=np.linspace(0,len(train)-1,take,dtype=int)
    cv_df=train.iloc[idx]; X_cv=cv_df[feature_cols]; y_cv=cv_df[TARGET].astype(int)
    if len(cv_df)<10: raise ValueError("Not enough CV rows")
    specs=model_specs(numeric,categorical)
    cv_rows=[]; comparison=[]; fitted={}; warnings.filterwarnings("ignore",category=ConvergenceWarning)
    for model_name,(pipe,param_grid) in specs.items():
        cv=make_date_grouped_cv(cv_df.Date,n_splits=5)
        search=GridSearchCV(pipe,param_grid=param_grid,scoring="average_precision",cv=cv,n_jobs=1,refit=True,return_train_score=False,error_score="raise",verbose=0)
        search.fit(X_cv,y_cv)
        for r in pd.DataFrame(search.cv_results_).to_dict(orient="records"):
            cv_rows.append({"model":model_name,"config_rank":int(r["rank_test_score"]),"params":json.dumps(r["params"],sort_keys=True,default=str),"mean_cv_average_precision":float(r["mean_test_score"]),"std_cv_average_precision":float(r["std_test_score"]),"mean_fit_time_seconds":float(r["mean_fit_time"]),"mean_score_time_seconds":float(r["mean_score_time"]),"folds":5,"cv_rows":len(cv_df)})
        best_params=search.best_params_; final_pipe=search.best_estimator_
        # Refit on the full training era except KNN, whose exact neighbor search is
        # quadratic at prediction time; use a deterministic, date-spanning sample.
        if model_name=="K-Nearest Neighbors":
            fit_n=min(args.knn_fit_rows,len(train)); fit_idx=np.linspace(0,len(train)-1,fit_n,dtype=int)
            X_fit=X_train.iloc[fit_idx]; y_fit=y_train.iloc[fit_idx]
        else:
            X_fit=X_train; y_fit=y_train
        final_pipe.fit(X_fit,y_fit)
        scores=score_vector(final_pipe,X_test); pred=final_pipe.predict(X_test).astype(int)
        row={"model":model_name,"best_params":json.dumps(best_params,sort_keys=True,default=str),"cv_best_average_precision":float(search.best_score_),"final_fit_rows":len(X_fit),"accuracy":accuracy_score(y_test,pred),"balanced_accuracy":balanced_accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),"recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),"roc_auc":roc_auc_score(y_test,scores),"average_precision":average_precision_score(y_test,scores),"test_rows":len(test),"test_positive_prevalence":float(y_test.mean())}
        comparison.append(row); fitted[model_name]=(final_pipe,scores,pred)
        print(f"Completed {model_name}: CV AP={search.best_score_:.4f}; test AP={row['average_precision']:.4f}",flush=True)
    pd.DataFrame(cv_rows).to_csv(out/"hyperparameter_search_results.csv",index=False)
    results=pd.DataFrame(comparison).sort_values("cv_best_average_precision",ascending=False).reset_index(drop=True)
    results.insert(0,"cv_rank",np.arange(1,len(results)+1)); results.to_csv(out/"model_comparison.csv",index=False); results.to_csv(out/"metrics.csv",index=False)
    best_name=str(results.iloc[0].model); best_pipe,best_scores,best_pred=fitted[best_name]
    joblib.dump(best_pipe,out/"best_model.joblib",compress=3)
    report=classification_report(y_test,best_pred,target_names=["No","Yes"],output_dict=True,zero_division=0)
    pd.DataFrame(report).T.to_csv(out/"final_classification_report.csv")
    cm=confusion_matrix(y_test,best_pred,labels=[0,1]); pd.DataFrame(cm,index=["actual_No","actual_Yes"],columns=["predicted_No","predicted_Yes"]).to_csv(out/"final_confusion_matrix.csv")
    fpr,tpr,thresholds=roc_curve(y_test,best_scores); pd.DataFrame({"false_positive_rate":fpr,"true_positive_rate":tpr,"threshold":thresholds}).to_csv(out/"final_roc_curve.csv",index=False)
    pd.DataFrame({"actual":y_test.to_numpy(),"score":best_scores,"predicted":best_pred}).to_csv(out/"final_test_predictions.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    im=axes[0].imshow(cm,cmap="Blues"); axes[0].set(xticks=[0,1],xticklabels=["No","Yes"],yticks=[0,1],yticklabels=["No","Yes"],xlabel="Predicted",ylabel="Actual",title=f"{best_name} confusion matrix")
    for (i,j),v in np.ndenumerate(cm): axes[0].text(j,i,f"{v:,}",ha="center",va="center",color="black",fontsize=11)
    fig.colorbar(im,ax=axes[0],fraction=.046,pad=.04)
    axes[1].plot(fpr,tpr,color="#176B87",lw=2,label=f"AUC={roc_auc_score(y_test,best_scores):.3f}"); axes[1].plot([0,1],[0,1],"--",color="#888",lw=1)
    axes[1].set(xlabel="False positive rate",ylabel="True positive rate",title="ROC curve — later-period holdout",xlim=(0,1),ylim=(0,1)); axes[1].legend(loc="lower right")
    fig.suptitle(f"Final holdout evaluation: {best_name}",fontsize=14); fig.tight_layout(); fig.savefig(out/"final_evaluation.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5)); r=results.sort_values("average_precision")
    ax.barh(r.model,r.average_precision,color="#3978A8"); ax.set(xlabel="Test average precision",title="Tuned model comparison on chronological holdout",xlim=(0,max(r.average_precision)*1.15))
    for y,v in enumerate(r.average_precision): ax.text(v+.005,y,f"{v:.3f}",va="center")
    fig.tight_layout(); fig.savefig(out/"model_comparison.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    metadata={"dataset_file":data_path.name,"rows_after_cleaning":len(df),"features_before_encoding":len(feature_cols),"numeric_features":numeric,"categorical_features":categorical,"training_rows":len(train),"test_rows":len(test),"training_start":str(train.Date.min().date()),"training_end":str(train.Date.max().date()),"test_start":str(test.Date.min().date()),"test_end":str(test.Date.max().date()),"cutoff_date":str(cutoff.date()),"selection_metric":"five-fold chronological CV average precision, training era only","cv_strategy":"TimeSeriesSplit(n_splits=5), chronological; deterministic evenly spaced subset of training era","cv_rows":len(cv_df),"cv_positive_prevalence":float(y_cv.mean()),"holdout_positive_prevalence":float(y_test.mean()),"seed":SEED,"selected_model":best_name,"selected_model_reason":"Highest mean average precision over 5 chronological CV folds within the training period; test set was not used for model selection.","software":{"python":platform.python_version(),"pandas":pd.__version__,"scikit_learn":sklearn.__version__}}
    metadata["selection_metric"]="five-fold date-grouped expanding chronological CV average precision, training era only"
    metadata["cv_strategy"]="Five expanding-window folds partitioned by whole calendar dates on an evenly spaced training-era subset"
    metadata["cv_rows_limit"]=args.cv_rows
    metadata["knn_fit_rows_limit"]=args.knn_fit_rows
    metadata["selected_model_final_fit_rows"]=int(results.loc[0,"final_fit_rows"])
    metadata["selected_model_reason"]="Highest mean average precision across five date-grouped chronological CV folds within the training period; holdout was not used for model selection."
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("\nMODEL COMPARISON (held-out metrics, ranked by training-only CV AP)")
    print(results[["model","cv_best_average_precision","accuracy","precision","recall","f1","roc_auc","average_precision"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nSelected model:",best_name,"; test classification report:\n",pd.DataFrame(report).T.to_string())
    print("Saved to",out.resolve())

if __name__=="__main__": main()
