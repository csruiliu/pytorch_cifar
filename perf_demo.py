import torch
import torchvision
import torchvision.transforms as transforms

from models.lenet import LeNet
from models.alexnet import AlexNet


def main():
    #################################
    # hyperparameter
    #################################
    batch_size = 32
    learn_rate = 0.005
    train_epoch = 10

    #################################
    # prepare the training data
    #################################
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./dataset',
                                            train=True,
                                            download=True,
                                            transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)

    #################################
    # build model
    #################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = LeNet()
    model = AlexNet()

    # put the model on GPU
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4)

    #################################
    # train model
    #################################

    for epoch in range(train_epoch):
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # send the inputs
            inputs, labels = data[0].to(device), data[1].to(device)

            # reset gradient to zero
            optimizer.zero_grad()

            # make forward computation and get loss
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            # backward computation according to loss
            loss.backward()

            # update gradient according to backward computation
            optimizer.step()

            # get the prediction
            _, prediction = outputs.max(1)

            correct_prediction += prediction.eq(labels).sum().item()
            total_prediction += labels.size(0)

        print(f'Accuracy after epoch {epoch+1}: {correct_prediction / total_prediction}')


if __name__ == "__main__":
    main()
