"""Create a reconciled calibrated-results copy using primary selected candidates.
Only recalculates rows where saved calibrated C/gamma differs from primary results.
Original results files are not overwritten.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, f1_score
from src.data_loader import get_dataset
from src.evaluation.nested_cv_experiment import RANDOM_STATE, OUTER_FOLDS, INNER_FOLDS, PREPROCESSING_CONFIGS, create_model, RESULTS_DIR
from src.evaluation.threshold_calibration_experiment import choose_threshold
OUT=RESULTS_DIR/'additional_analysis'; OUT.mkdir(parents=True,exist_ok=True)

def main():
    primary=pd.read_csv(RESULTS_DIR/'nested_cv_all_datasets_preprocessing_comparison.csv')
    old=pd.read_csv(RESULTS_DIR/'nested_cv_all_datasets_threshold_calibrated_comparison.csv')
    keys=['dataset','preprocessing_config','outer_fold','method']
    m=old.merge(primary[keys+['C','gamma','inner_mean_loss','inner_std_loss','selection_score']],on=keys,suffixes=('_old','_primary'))
    mismatch=((m.C_old-m.C_primary).abs()>1e-8)|((m.gamma_old-m.gamma_primary).abs()>1e-8)
    print('Rows with C/gamma mismatch:',int(mismatch.sum()))
    corrected=old.copy()
    for _,r in m.loc[mismatch].iterrows():
        ds=r.dataset; cfg=next(c for c in PREPROCESSING_CONFIGS if c['name']==r.preprocessing_config)
        X,y,categorical=get_dataset(ds); numeric=[c for c in X.columns if c not in categorical]
        splits=list(StratifiedKFold(n_splits=OUTER_FOLDS,shuffle=True,random_state=RANDOM_STATE).split(X,y))
        fold=int(r.outer_fold); tri,tei=splits[fold-1]
        Xtr,Xte=X.iloc[tri],X.iloc[tei]; ytr,yte=y.iloc[tri],y.iloc[tei]
        inn=list(StratifiedKFold(n_splits=INNER_FOLDS,shuffle=True,random_state=RANDOM_STATE+fold).split(Xtr,ytr))
        args=dict(C=float(r.C_primary),gamma=float(r.gamma_primary),numeric_features=numeric,categorical_features=categorical,numeric_imputation=cfg['numeric_imputation'],dimensionality_reduction=cfg['dimensionality_reduction'])
        oof=cross_val_predict(create_model(**args),Xtr,ytr,cv=inn,method='decision_function',n_jobs=1)
        threshold,inner_ba=choose_threshold(ytr,oof)
        model=create_model(**args); model.fit(Xtr,ytr); scores=model.decision_function(Xte); pred=(scores>=threshold).astype(int)
        yy=np.asarray(yte,dtype=int); tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel(); auc=float(roc_auc_score(yy,scores)); loss=1-auc
        ix=(corrected.dataset==ds)&(corrected.preprocessing_config==r.preprocessing_config)&(corrected.outer_fold==fold)&(corrected.method==r.method)
        vals={'C':args['C'],'gamma':args['gamma'],'inner_mean_loss':float(r.inner_mean_loss_primary),'inner_std_loss':float(r.inner_std_loss_primary),'selection_score':float(r.selection_score_primary),'threshold':float(threshold),'inner_oof_balanced_accuracy_at_threshold':float(inner_ba),'outer_roc_auc':auc,'outer_loss':loss,'overtuning_gap':loss-float(r.inner_mean_loss_primary),'balanced_accuracy':float(balanced_accuracy_score(yy,pred)),'sensitivity':float(tp/(tp+fn)) if tp+fn else 0.0,'specificity':float(tn/(tn+fp)) if tn+fp else 0.0,'f1_score':float(f1_score(yy,pred,zero_division=0)),'predicted_negative_count':int((pred==0).sum()),'predicted_positive_count':int((pred==1).sum()),'decision_score_min':float(scores.min()),'decision_score_max':float(scores.max())}
        for col,val in vals.items(): corrected.loc[ix,col]=val
        print('Reconciled:',ds,r.preprocessing_config,fold,r.method)
    out=OUT/'threshold_calibrated_consistency_checked_outer_folds.csv'; corrected.to_csv(out,index=False)
    metrics=['outer_roc_auc','balanced_accuracy','sensitivity','specificity','f1_score','overtuning_gap']
    summary=corrected.groupby(['dataset','preprocessing_config','method'])[metrics].agg(['mean','std']); summary.columns=[f'{a}_{b}' for a,b in summary.columns]; summary=summary.reset_index()
    summary.to_csv(OUT/'threshold_calibrated_consistency_checked_summary.csv',index=False)
    print('Saved',out)
    print('Saved summary; corrected rows:',int(mismatch.sum()))
if __name__=='__main__': main()
