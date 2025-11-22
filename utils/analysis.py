import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def generate_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    return fig

def report_text(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=None)
