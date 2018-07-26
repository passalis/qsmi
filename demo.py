import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from hashing.qmi_hashing import train_model_qmi, extract_network_representation
from hashing.evaluation import evaluate_database
import torch.nn as nn
import torch.nn.functional as F


def load_fashion_mnist(batch_size=128, dataset_path="/home/nick/Data/Datasets/torch"):
    """
    Fashion MNIST loader
    :param batch_size:
    :param dataset_path:
    :return:
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = dsets.FashionMNIST(root=dataset_path, train=True, transform=transform)
    test_dataset = dsets.FashionMNIST(root=dataset_path, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


class FashionNet(nn.Module):

    def __init__(self, bits=24):
        super(FashionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, bits)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d((self.conv2(x)), 2))
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return x



def train_evaluate(bits):
    train_loader, test_loader = load_fashion_mnist()

    # Define the model
    model = FashionNet(bits=bits)
    model.cuda()

    train_model_qmi(model, train_loader, learning_rate=0.001, epochs=50, alpha=0.01, M=0)

    # In domain evaluation
    train_data, train_labels = extract_network_representation(model, train_loader)
    test_data, test_labels = extract_network_representation(model, test_loader)

    results = evaluate_database(train_data, train_labels, test_data, test_labels, metric='hamming', batch_size=128)
    print("mAP:", results[0])
    print("Precision < 2 bits:", results[-1])


if __name__ == '__main__':

    for bits in [12, 24, 36, 48]:
        train_evaluate(bits)


