import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model import CNN_DigitClassifier
import random
from tqdm import tqdm
import os
import yaml

# prepare dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_data = datasets.MNIST(
    "mnist_data", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    "mnist_data", train=False, download=True, transform=transform
)


# global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_DigitClassifier().to(device)

learning_rate = 0.01
momentum = 0.5
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def getdatasetstate(args={}):
    return {k: k for k in range(70000)}


def train(args, labeled, resume_from, ckpt_file):
    labled_data = torch.utils.data.Subset(train_data, labeled)
    train_dataloader = torch.utils.data.DataLoader(
        labled_data, batch_size=args["BATCH_SIZE"], shuffle=True
    )

    if resume_from is not None and not args["weightsclear"]:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    # loss function
    cec = nn.CrossEntropyLoss()
    current_loss = float("inf")
    for epoch in range(3):
        for batch_idex, (images, labels) in enumerate(
            tqdm(train_dataloader, leave=False)
        ):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        print(f"Ending Epoch: {epoch+1} with Loss: {loss.item()}")
        if loss.item() < current_loss:
            current_loss = loss.item()
            if not os.path.isdir(args["EXPT_DIR"]):
                os.mkdir(args["EXPT_DIR"])

            print(f"Saving Model to {ckpt_file}")
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))
    return


def test(args, ckpt_file):

    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))
    model.load_state_dict(ckpt["model"])
    model.eval()
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=args["BATCH_SIZE"], shuffle=False
    )
    predictions, targets = [], []

    correct, total = 0, 0
    for images, labels in tqdm(test_dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)
        _, predicted = torch.max(pred.data, 1)
        predictions.extend(predicted.cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())

        # calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Accuracy: {correct / len(test_data) * 100} %",)
    return {"predictions": predictions, "labels": targets}


def infer(args, unlabeled, ckpt_file):
    unlabeled_data = torch.utils.data.Subset(train_data, unlabeled)
    unlabeled_dataloader = torch.utils.data.DataLoader(
        unlabeled_data, batch_size=args["BATCH_SIZE"], shuffle=False
    )

    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))
    model.load_state_dict(ckpt["model"])
    model.eval()
    correct, total, k = 0, 0, 0
    outputs_fin = {}
    for i, data in tqdm(enumerate(unlabeled_dataloader), desc="Inferring"):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).data

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for j in range(len(outputs)):
            outputs_fin[k] = {}
            outputs_fin[k]["prediction"] = predicted[j].item()
            outputs_fin[k]["pre_softmax"] = outputs[j].cpu().numpy().tolist()
            k += 1

    return {"outputs": outputs_fin}


if __name__ == "__main__":
    with open("./config.yaml", "r") as stream:
        args = yaml.safe_load(stream)

    resume_from = None
    ckpt_file = "ckpt_0"

    train(args, random.sample(range(60000), 10000), resume_from, ckpt_file)
    test_results = test(args, ckpt_file)
    infer(args, random.sample(range(60000), 1000), ckpt_file)
