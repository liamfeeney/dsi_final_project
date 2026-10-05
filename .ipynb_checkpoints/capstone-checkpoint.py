import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, accuracy_score, roc_auc_score
def plot_roc(y_probs, ys, labels=None):
    plt.figure(figsize=(7,5))
    print(len(y_probs))
    if (len(y_probs) != len(ys)):
        return()
    for i, y_prob in enumerate(y_probs):
        fpr, tpr, thresholds = roc_curve(ys[i], y_prob)
        auc = roc_auc_score(ys[i], y_prob)
        if len(labels) > i:
            label = f"{labels[i]} (AUC = {auc:.3f})"
        else:
            label = f"(AUC = {auc:.3f})"
        plt.plot(fpr, tpr, label=label)
        # Reference diagonal
    
    plt.plot([0,1],[0,1],'k--')
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
def get_confusion(y_prob, y, threshold):
    # Predict based on threshold
    y_pred_opt = (y_prob >= threshold).astype(int)
    accuracy = accuracy_score(y, y_pred_opt)
    print(f"Accuracy with optimum threshold: {accuracy:.3f}")
    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y, y_pred_opt))
    
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y, y_pred_opt))
def find_optimal_threshold(y_prob,y):
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    i = np.arange(len(tpr))

    # Build ROC dataframe
    roc = pd.DataFrame({
        'fpr'        : pd.Series(fpr, index=i),
        'tpr'        : pd.Series(tpr, index=i),
        '1-fpr'      : pd.Series(1-fpr, index=i),
        'tf'         : pd.Series(tpr - (1-fpr), index=i),  # Youden's J statistic
        'thresholds' : pd.Series(thresholds, index=i)
    })
    
    # Find optimal threshold (maximizing sensitivity-specificity difference)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = roc.iloc[optimal_idx]['thresholds']
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    return(optimal_threshold)