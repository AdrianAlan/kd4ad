import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import wasserstein_distance as emd
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def plot_loss(loss, path):
    """Plot loss function"""
    plt.figure(figsize=(5, 4))
    plt.semilogy(loss["train"], label="Train")
    plt.semilogy(loss["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss per batch")
    plt.grid()
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()


def plot_roc(y_true, y_pred_student, y_pred_teacher, save_path):
    teacher_fpr, teacher_tpr, _ = roc_curve(y_true + 0.0, -y_pred_teacher)
    student_fpr, student_tpr, _ = roc_curve(y_true + 0.0, -y_pred_student)
    teacher_auc = np.round(roc_auc_score(y_true + 0.0, -y_pred_teacher), 3)
    student_auc = np.round(roc_auc_score(y_true + 0.0, -y_pred_student), 3)

    plt.plot(
        teacher_fpr,
        teacher_tpr,
        label="Teacher, AUC={0:.3f}".format(teacher_auc),
        zorder=4,
    )
    plt.plot(
        student_fpr,
        student_tpr,
        label="Student, AUC={0:.3f}".format(student_auc),
        zorder=4,
    )
    plt.plot(
        [0, 0.2, 0.5, 0.7, 1],
        [0, 0.2, 0.5, 0.7, 1],
        linestyle="--",
        color="gray",
        zorder=4,
    )
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(zorder=1)
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()


def plot_prc(y_true, y_pred_student, y_pred_teacher, save_path):
    teacher_precision, teacher_recall, _ = precision_recall_curve(
        y_true + 0.0, -y_pred_teacher, pos_label=1
    )
    teacher_ap = average_precision_score(y_true + 0.0, -y_pred_teacher)
    student_precision, student_recall, _ = precision_recall_curve(
        y_true + 0.0, -y_pred_student, pos_label=1
    )
    student_ap = average_precision_score(y_true + 0.0, -y_pred_student)

    plt.plot(
        teacher_recall,
        teacher_precision,
        label="Teacher, AP={0:.3f}".format(teacher_ap),
        zorder=4,
    )
    plt.plot(
        student_recall,
        student_precision,
        label="Student, AP={0:.3f}".format(student_ap),
        zorder=4,
    )
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(zorder=1)
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()


def plot_dist(y_true, y_pred_student, y_pred_teacher, save_path):
    # Print score distance
    teacher_anomaly_score = np.array(y_pred_teacher[~y_true])
    student_anomaly_score = np.array(y_pred_student[~y_true])
    teacher_normal_score = np.array(y_pred_teacher[y_true])
    student_normal_score = np.array(y_pred_student[y_true])
    distance_anomaly = emd(student_anomaly_score, teacher_anomaly_score)
    distance_normal = emd(student_normal_score, teacher_normal_score)

    maximum = np.max([np.max(y_pred_teacher), np.max(y_pred_student)])
    bins = np.arange(0, maximum, maximum / 50)

    plt.hist(
        np.array(y_pred_teacher[~y_true]),
        bins=bins,
        density=True,
        edgecolor="#B71C1C",
        facecolor="none",
        histtype="step",
        label="Teacher, Anomaly",
    )
    plt.hist(
        np.array(y_pred_student[~y_true]),
        bins=bins,
        density=True,
        edgecolor="#EF5350",
        facecolor="none",
        histtype="step",
        label="Student, EMD: {0:.3f}".format(distance_anomaly),
    )
    plt.hist(
        np.array(y_pred_teacher[y_true]),
        bins=bins,
        density=True,
        edgecolor="#66BB6A",
        facecolor="none",
        histtype="step",
        label="Teacher, Normal",
    )
    plt.hist(
        np.array(y_pred_student[y_true]),
        bins=bins,
        density=True,
        edgecolor="#1B5E20",
        facecolor="none",
        histtype="step",
        label="Student, EMD: {0:.3f}".format(distance_normal),
    )
    plt.xlabel("Anomaly score")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()
