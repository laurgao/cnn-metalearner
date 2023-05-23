import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# main function: weights (tensor) + data (tensor) -> 16 pairs of scalars (representing classification probabilities)

N_IN = 1  # number of input channels
N_BATCH = 16
N_SIZE = 28


def generate_random_input_data(batchsize=N_BATCH):
    # torch.rand creates random values uniformly [0, 1]
    # image data is usually scaled to [0, 1] from the [0, 255] range when inputted to ml models.
    return torch.rand((batchsize, N_IN, N_SIZE, N_SIZE), requires_grad=True)
    # minibatch, in channels, height, width


def generate_random_weights():
    # now we want to generate random weights
    # 3 conv layers 4 filters per layer
    conv1 = generate_random_weights_for_layer(4, N_IN, 3, 3)
    conv2 = generate_random_weights_for_layer(4, 4, 3, 3)
    conv3 = generate_random_weights_for_layer(4, 4, 3, 3)

    linear_transformation = torch.empty(64, 2)
    linear_transformation = nn.init.xavier_uniform_(
        linear_transformation, gain=nn.init.calculate_gain('relu'))
    linear_transformation.requires_grad = True

    return conv1, conv2, conv3, linear_transformation


def generate_random_weights_for_layer(out_channels, in_channels, w, h):
    # return torch.rand((out_channels, in_channels, w, h), requires_grad=True)
    w = torch.empty(out_channels, in_channels, w, h)
    w = nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    w.requires_grad = True
    return w


def my_relu(tensor):
    return torch.max(tensor, torch.zeros_like(tensor))


class WeirdCNN(nn.Module):
    def __init__(self, conv1, conv2, conv3, lt):
        super().__init__()
        # self.fuckyweights = generate_random_weights()
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.linear_transformation = lt
        self.max_pool = nn.MaxPool2d(2, 2)
        # for odd number so edge doesn't get cut off
        self.max_pool_odd = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, input):
        feature_maps_1 = F.conv2d(input, self.conv1)  # 26
        # apply relu
        feature_maps_1 = torch.relu(feature_maps_1)
        feature_maps_1 = self.max_pool(feature_maps_1)  # 13

        # layer 2
        # does this fn involve adding shit up too?
        feature_maps_2 = F.conv2d(feature_maps_1, self.conv2)  # 11
        feature_maps_2 = torch.relu(feature_maps_2)
        feature_maps_2 = self.max_pool_odd(feature_maps_2)  # 6

        # layer 3
        feature_maps_3 = F.conv2d(feature_maps_2, self.conv3)  # 4
        feature_maps_3 = torch.relu(feature_maps_3)

        y3 = feature_maps_3.flatten(1)  # should be 16x(4x4x4)

        # you want this y before softmax to be in the range of like -5 to 5

        y = torch.matmul(y3, self.linear_transformation)

        # softmax y
        return torch.softmax(y, dim=1)


def cursed_predictor(input_data, weights_sequence):
    model = WeirdCNN(*weights_sequence)
    # .forward is like the .__call__ ???
    return model(input_data)


def cursed_loss_function(probabilities):
    # probabilities: output of cursed_predictor
    probabilities_top = probabilities[:, 0]

    # sort
    sorted_tensor, indices = torch.sort(probabilities_top, descending=True)
    # these are sorted from highest prob of top to lowest prob of top

    # split into 2 groups
    t8 = sorted_tensor[:N_BATCH // 2]
    b8 = sorted_tensor[N_BATCH // 2:]

    # cross entropy loss
    top_loss = torch.sum(-torch.log(t8))
    bottom_loss = torch.sum(-torch.log(1 - b8))

    return top_loss + bottom_loss


def cursed_train():
    image_tensor = generate_random_input_data()

    weights = generate_random_weights()
    params = [image_tensor, *weights]
    optimizer = torch.optim.SGD(params, lr=0.01)

    for i in range(100):
        for p in params:
            p.grad = None
        probabilities = cursed_predictor(image_tensor, weights)
        loss = cursed_loss_function(probabilities)
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            print(i, loss.item())

    # now we want to print out/store the images

    # ooh what if we store it in order
    _, indices = torch.sort(probabilities[:, 0], descending=True)
    sorted_tensor = image_tensor[indices]
    # these are sorted from highest prob of top to lowest prob of top

    # numpy_data = data.detach().numpy() # convert to numpy array
    from torchvision.utils import save_image
    sorted_tensor = sorted_tensor.cpu()
    save_image(sorted_tensor, 'images.png', nrow=4)

    # i should also like store image data along with their labels.
    # eh this is easy enough to implement, just pickle or something, so i'll do it later


if __name__ == "__main__":
    cursed_train()
