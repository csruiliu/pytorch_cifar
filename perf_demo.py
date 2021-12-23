import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
import argparse

from timeit import default_timer as timer

from models.lenet import LeNet
from models.alexnet import AlexNet
from models.vgg import VGG
from models.mobilenet import MobileNet
from models.mobilenet_v2 import MobileNetV2
from models.resnet import ResNet
from models.zfnet import ZFNet
from models.densenet import DenseNet
from models.efficientnet import EfficientNet
from models.resnext import ResNext
from models.inception import Inception
from models.shufflenet import ShuffleNet
from models.shufflenet_v2 import ShuffleNetV2
from models.squeezenet import SqueezeNet
from models.xception import Xception


def main():
    #################################
    # hyper-parameter
    #################################
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--modelname', action='store', type=str, default='resnet',
                        choices=['lenet', 'alexnet', 'vgg', 'mobilenet', 'mobilenetv2', 'resnet', 
                                 'zfnet', 'densenet', 'efficientnet', 'resnext', 'inception', 
                                 'shufflenet', 'shufflenetv2', 'squeezenet', 'xception'],
                        help='set the model arch')
    parser.add_argument('-b', '--batchsize', action='store', type=int, default=32,
                        help='set the training batch size')
    parser.add_argument('-r', '--lr', action='store', type=float, default=0.005,
                        help='set the learning rate')
    parser.add_argument('-e', '--epoch', action='store', type=int, default=1,
                        help='set the number of training epoch')

    args = parser.parse_args()
    
    model_name = args.modelname
    batch_size = args.batchsize
    learn_rate = args.lr
    train_epoch = args.epoch
    
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

    if model_name == 'lenet':
        model = LeNet()
    elif model_name == 'alexnet':
        model = AlexNet()
    elif model_name == 'vgg':
        model = VGG(conv_layer=11)
    elif model_name == 'mobilenet':
        model = MobileNet()
    elif model_name == 'mobilenetv2':
        model = MobileNetV2()
    elif model_name == 'resnet':
        model = ResNet(residual_layer=18)
    elif model_name == 'zfnet':
        model = ZFNet()
    elif model_name == 'densenet':
        model = DenseNet(residual_layer=121)
    elif model_name == 'efficientnet':
        model = EfficientNet()
    elif model_name == 'resnext':
        model = ResNext(cardinality=2)
    elif model_name == 'inception':
        model = Inception()
    elif model_name == 'shufflenet':
        model = ShuffleNet(num_groups=2)
    elif model_name == 'shufflenetv2':
        model = ShuffleNetV2(complexity=0.5)
    elif model_name == 'squeezenet':
        model = SqueezeNet()
    elif model_name == 'xception':
        model = Xception()
    else:
        raise ValueError('not supported model name')

    # put the model on GPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
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
    model_stats = summary(model, input_size=(batch_size, 3, 32, 32))
    total_parameters = model_stats.total_params
    total_memory = (model_stats.to_megabytes(model_stats.total_input) +
                    model_stats.float_to_megabytes(model_stats.total_output + 
                    model_stats.total_params))
    print(f'Total Parameters: {total_parameters}')
    print(f'Total Memory (MB): {total_memory}')


if __name__ == "__main__":
    main()
