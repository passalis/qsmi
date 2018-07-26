import torch.optim as optim
from time import time
import numpy as np
import torch
from torch.autograd import Variable


def train_model_qmi(model, train_loader, learning_rate=0.001, epochs=10, sigma=0, alpha=0.001,
                    M=0, use_cosine=True, use_square_clamp=True, ):
    """
    Trains a model using the proposed QSMIH method
    :param model: the model to be trained
    :param train_loader: a dataset loader
    :param learning_rate: learning rate to be used for the optimization
    :param epochs: number of epochs to run the optimization
    :param sigma: scaling factor for the Gaussian kernel (if Parzen window estimation is used)
    :param alpha: weight for the hashing regularizer
    :param M: number of information needs (assuming that each one is equiprobable)
    :param use_cosine: Set to true to use QSMI, otherwise QMI is used
    :param use_square_clamp: Set to true to used the square clamping trick
    :return:
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # For each optimization epoch
    for cur_epoch in range(epochs):

        start_time = time()
        avg_loss, counter = 0, 0

        # For each batch
        for documents, targets in train_loader:
            # Get the data and sample some queries
            documents_raw, targets = Variable(documents.cuda()), Variable(targets.cuda())

            # Feed forward the network
            optimizer.zero_grad()
            documents = model(documents_raw)

            # Calculate the QSMI loss
            qsmi_loss = qmi_loss(documents, targets, sigma=sigma, M=M,
                                 use_square_clamp=use_square_clamp, use_cosine=use_cosine)
            # Hashing regularizer
            regularizer = torch.sum(torch.abs(torch.abs(documents) - 1))

            # Calculate the total loss
            loss = qsmi_loss + alpha * regularizer
            loss.backward()

            # Optimize the model
            optimizer.step()

            # Measure loss
            avg_loss += loss.data.item()
            counter += documents.size(0)

        avg_loss = avg_loss / float(counter)
        end_time = time()
        elapsed_time = (end_time - start_time)
        time_left = elapsed_time * (epochs - cur_epoch - 1)

        print('Train Epoch: {} \tLoss: {:.6f} \t Time left: {:.2f}s ({:.2f}s per epoch)'.
              format(cur_epoch, avg_loss, time_left, elapsed_time))


def qmi_loss(documents, targets, sigma=3, M=10, eps=1e-8, use_cosine=True, use_square_clamp=True):
    """
    Implements the QSMI loss
    :param documents: the documents representation
    :param targets: the documents labels
    :param sigma: scaling factor for the Gaussian kernel (if used)
    :param eps: a small number used to ensure the stability of the cosine similarity
    :param M: number of information needs (assuming that each one is equiprobable)
    :param use_cosine: Set to true to use QSMI, otherwise QMI is used
    :param use_square_clamp: Set to true to used the square clamping method
    :return: the QMI/QSMI loss
    """

    if use_cosine:
        documents = documents / (torch.sqrt(torch.sum(documents ** 2, dim=1, keepdim=True)) + eps)
        Y = torch.mm(documents, documents.t())
        Y = 0.5 * (Y + 1)
    else:
        Y = squared_pairwise_distances(documents)
        Y = torch.exp(-Y / (2 * sigma ** 2))

    # Get the indicator matrix \Delta
    D = (targets.view(targets.shape[0], 1) == targets.view(1, targets.shape[0]))
    D = D.type(torch.cuda.FloatTensor)

    if M == 0:
        M = D.size(1) ** 2 / torch.sum(D)

    if use_square_clamp:
        Q_in = (D * Y - 1) ** 2
        Q_btw = (1.0 / M) * Y ** 2
        # Minimize clamped loss
        loss = torch.sum(Q_in + Q_btw)
    else:
        Q_in = D * Y
        Q_btw = (1.0 / M) * Y
        # Maximize QMI/QSMI
        loss = -torch.sum(Q_in - Q_btw)

    return loss


def squared_pairwise_distances(a, b=None):
    """
    Calculates the pairwise distances between matrices a and b (or a and a, if b is not set)
    :param a:
    :param b:
    :return:
    """
    if b is None:
        b = a

    aa = torch.sum(a ** 2, dim=1)
    bb = torch.sum(b ** 2, dim=1)

    aa = aa.expand(bb.size(0), aa.size(0)).t()
    bb = bb.expand(aa.size(0), bb.size(0))

    AB = torch.mm(a, b.transpose(0, 1))

    dists = aa + bb - 2 * AB
    dists = torch.clamp(dists, min=0, max=np.inf)

    return dists


def extract_network_representation(net, loader):
    """
    Extracts the hash codes
    :param net: the network to feed-forward
    :param loader: the loader to use for extracting the representation
    :return:
    """

    net.eval()

    features = []
    labels = []

    for (inputs, targets) in loader:
        inputs = Variable(inputs.cuda(), volatile=True)
        outputs = net(inputs)
        outputs = torch.sign(outputs)
        features.append(outputs.cpu().data.numpy().astype('int8'))
        labels.append(targets.cpu().numpy())

    return np.concatenate(features), np.concatenate(labels)
