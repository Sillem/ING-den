import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score

def get_predictions_proba_train(model):
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    return y_pred_proba_train


def get_predictions_proba_test(model):
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    return y_pred_proba_test


def get_predictions_test(model, threshold=0.5):
    y_pred_test_proba = model.predict_proba(X_test)[:, 1]
    y_pred_test_proba = np.array([1 if p > threshold else 0 for p in y_pred_test_proba])
    return y_pred_test_proba


def evaluate_model(model, X_train, y_train, X_test, y_test, threshold=0.5):
    model_name = type(model).__name__
    print(f"Evaluating model: {model_name}")

    # Train set
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred_proba_train)
    y_pred_train = np.array([1 if p > threshold else 0 for p in y_pred_proba_train])
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)

    # Test set
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred_proba_test)
    y_pred_test = np.array([1 if p > threshold else 0 for p in y_pred_proba_test])
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)

    # Create DataFrame
    data = {
        'Model': [model_name, model_name],
        'Dataset': ['Train', 'Test'],
        'AUC': [auc_train, auc_test],
        'Accuracy': [accuracy_train, accuracy_test],
        'Precision': [precision_train, precision_test],
        'Recall': [recall_train, recall_test],
        'F1-score': [f1_train, f1_test]
    }
    df = pd.DataFrame(data)

    return df


def specificity_score(y_true, y_pred):
    tn = ((y_true == 0) & (y_pred == 0)).sum()  # True negatives
    fp = ((y_true == 0) & (y_pred == 1)).sum()  # False positives

    score = tn / (tn + fp)
    return score


def find_best_threshold(y_true, y_proba):
    tresholds = np.array([])
    best_threshold = 0

    roc_scores = np.array([])
    best_roc = 0

    recalls = np.array([])
    specificities = np.array([])

    for threshold in np.arange(0, 1, 0.001):
        y_pred = np.array([1 if p > threshold else 0 for p in y_proba])

        roc_score = roc_auc_score(y_true, y_pred)
        if roc_score > best_roc:
            best_roc = roc_score
            best_threshold = threshold

        recall = recall_score(y_true, y_pred)
        specificity = specificity_score(y_true, y_pred)

        roc_scores = np.append(roc_scores, roc_score)
        tresholds = np.append(tresholds, threshold)
        recalls = np.append(recalls, recall)
        specificities = np.append(specificities, specificity)

    plt.subplot(1, 2, 1)
    plt.plot(tresholds, roc_scores)
    plt.xlabel('treshold')
    plt.ylabel('roc AUC value')

    plt.subplot(1, 2, 2)
    plt.plot(tresholds, recalls)
    plt.plot(tresholds, specificities)
    plt.xlabel('treshold')
    plt.ylabel('recall and specifcity value')
    plt.tight_layout()

    return best_threshold, best_roc