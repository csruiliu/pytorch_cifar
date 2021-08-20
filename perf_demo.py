import torch
import torchvision
import torchvision.transforms as transforms

from timeit import default_timer as timer

from models.lenet import LeNet
from models.alexnet import AlexNet
from models.vgg import VGG
from models.mobilenet import MobileNet
from models.mobilenet_v2 import MobileNetV2
from models.resnet import ResNet
from models.zfnet import ZFNet
from models.densenet import DenseNet


def main():
    #################################
    # hyper-parameter
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

    # model = LeNet()
    # model = AlexNet()
    # model = VGG(conv_layer=11)
    # model = MobileNet()
    # model = MobileNetV2()
    # model = ResNet(residual_layer=18)
    # model = ZFNet()
    model = DenseNet(residual_layer=121)

    # put the model on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4)

    #################################
    # train model
    #################################

    print('start training model')

    start_time = timer()

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

    end_time = timer()
    proc_time = end_time - start_time

    print(f'finish training, training time:{proc_time}')


if __name__ == "__main__":
    main()
