import torch

from dataset import generate
from evaluate import evaluation
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
from plotting import plot_loss

from torch import nn
from tqdm import tqdm


def epoch(
    net, device, dataloader, loss_fn, optimizer=None, teacher=None, teacher_loss=None
):
    total_loss = torch.tensor(0.0, dtype=torch.float).to(device)
    scaler = torch.cuda.amp.GradScaler()
    for image_batch in dataloader:
        # Move tensor to the proper device
        target = image_batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Decode data
            decoded_data = net(target)
            if teacher:
                with torch.no_grad():
                    teacher_output = teacher(target)
                    target = teacher_loss(teacher_output, target)
                    target = torch.log(torch.mean(target, (2, 3)) + 1)
            # Evaluate loss
            loss = loss_fn(decoded_data, target)
        # Backward pass
        if optimizer:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # Print batch loss
        total_loss += loss.item()
    return total_loss.detach().cpu().numpy()


def run_experiment(dataset, digit, epochs, batch_size, save_path):

    # Define the loss function
    loss_fn_teacher = torch.nn.MSELoss()
    loss_fn_student = torch.nn.L1Loss()

    # Set the random seed for reproducible results
    torch.manual_seed(0)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # Initialize the teacher
    teacher = Teacher().to(device).apply(init_weights)

    # Define an optimizer
    lr = 1e-3
    optim_teacher = torch.optim.Adam(teacher.parameters(), lr=lr, weight_decay=1e-05)

    # Get the dataset
    train_loader, val_loader, test_loader, y_true = generate(digit, batch_size, dataset)

    loss = {"train": [], "val": []}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_teacher, "min")
    best_loss = float("inf")
    for _ in tqdm(range(epochs), desc="Progress Teacher"):
        teacher.train()
        train_loss = epoch(
            teacher, device, train_loader, loss_fn_teacher, optim_teacher
        ) / (len(train_loader) * batch_size)
        teacher.eval()
        with torch.no_grad():
            val_loss = epoch(teacher, device, val_loader, loss_fn_teacher) / (
                len(val_loader) * batch_size
            )
            if val_loss < best_loss:
                torch.save(
                    teacher.state_dict(),
                    "{}/{}-{}-Teacher".format(save_path, dataset, digit),
                )
        scheduler.step(val_loss)
        loss["train"].append(train_loss)
        loss["val"].append(val_loss)

    plot_loss(loss, "{}/{}-{}-Teacher-Loss".format(save_path, dataset, digit))

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
        student.to(device).apply(init_weights)
        optim_student = torch.optim.Adam(
            student.parameters(), lr=lr, weight_decay=1e-05
        )
        loss = {"train": [], "val": []}
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_student, "min")
        teacher.eval()
        score = torch.nn.MSELoss(reduction="none")
        best_loss = float("inf")
        for _ in tqdm(range(epochs), desc="Progress Student{}".format(chr(65 + x))):
            student.train()
            train_loss = epoch(
                student,
                device,
                train_loader,
                loss_fn_student,
                optimizer=optim_student,
                teacher=teacher,
                teacher_loss=score,
            ) / (len(train_loader) * batch_size)
            student.eval()
            with torch.no_grad():
                val_loss = epoch(
                    student,
                    device,
                    val_loader,
                    loss_fn_student,
                    teacher=teacher,
                    teacher_loss=score,
                ) / (len(val_loader) * batch_size)
                if val_loss < best_loss:
                    torch.save(
                        student.state_dict(),
                        "{}/{}-{}-Student{}".format(
                            save_path, dataset, digit, chr(65 + x)
                        ),
                    )
            scheduler.step(val_loss)
            loss["train"].append(train_loss)
            loss["val"].append(val_loss)

        plot_loss(
            loss,
            "{}/{}-{}-Student{}-Loss".format(save_path, dataset, digit, chr(65 + x)),
        )

    evaluation(device, test_loader, y_true, dataset, digit, save_path)


def main():
    # Perform experiments for both datasetets and all digits
    epochs = 300
    batch_size = 100
    save_path = "./results-baseline"
    for dataset in ["MNIST", "FMNIST"]:
        for digit in range(10):
            torch.cuda.empty_cache()
            print("Testing digit {}".format(digit))
            run_experiment(dataset, digit, epochs, batch_size, save_path)


if __name__ == "__main__":
    main()
