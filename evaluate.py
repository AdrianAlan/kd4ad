import numpy as np
import torch
from models import (
    Teacher,
    StudentA,
    StudentB,
    StudentC,
    StudentD,
    StudentE,
    StudentF,
    StudentG,
)
from plotting import plot_dist, plot_roc, plot_prc


def evaluation(device, test_loader, y_true, dataset, digit, save_path):
    # Evaluation
    teacher = Teacher().to(device)
    teacher.load_state_dict(
        torch.load("{}/{}-{}-Teacher".format(save_path, dataset, digit))
    )
    teacher.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        img_out = []
        img_in = []
        for image_batch in test_loader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Decode data
            output_teacher = teacher(image_batch)
            # Append the network output and the original image to the lists
            img_out.append(output_teacher.cpu())
            img_in.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        img_out = torch.cat(img_out)
        img_in = torch.cat(img_in)
    y_pred_teacher = torch.mean((img_out - img_in) ** 2, axis=(1, 2, 3))
    y_pred_teacher = y_pred_teacher.cpu().numpy()

    for x, student in enumerate(
        [
            StudentA(),
            StudentB(),
            StudentC(),
            StudentD(),
            StudentE(),
            StudentF(),
            StudentG(),
        ]
    ):
        student = student.to(device)
        student_prefix = "{}/{}-{}".format(save_path, dataset, digit)
        student.load_state_dict(
            torch.load("{}-Student{}".format(student_prefix, chr(65 + x)))
        )
        student.eval()

        with torch.no_grad():  # No need to track the gradients
            student_score = []
            for image_batch in test_loader:
                # Move tensor to the proper device
                image_batch = image_batch.to(device)
                # Decode data
                output_student = student(image_batch)
                # Append the network output and the original image to the lists
                student_score.append(output_student)
        y_pred_student = np.squeeze(torch.cat(student_score).cpu().numpy(), 1)

        plot_dist(
            y_true,
            y_pred_student,
            y_pred_teacher,
            "{}-Distribution-Student{}".format(student_prefix, chr(65 + x)),
        )
        plot_roc(
            y_true,
            y_pred_student,
            y_pred_teacher,
            "{}-ROC-Student{}".format(student_prefix, chr(65 + x)),
        )
        plot_prc(
            y_true,
            y_pred_student,
            y_pred_teacher,
            "{}-PR-Student{}".format(student_prefix, chr(65 + x)),
        )
