import torch
import torch.nn as nn
import torch.nn.functional as F

# main function: weights (tensor) + data (tensor) -> 16 pairs of scalars (representing classification probabilities)

# so you want random image of 28x28 pixels and 16 of them.
# 16x28x28


def generate_random_input_data(batchsize=16):
    # assume grayscale for now
    # data should be normalized to like 0 mean stdev 1
    w = torch.rand((batchsize, 1, 28, 28))
    # w = w * 255
    w.requires_grad = True
    return w
    # minibatch, in channels, height, width

# now we want to generate random weights
# i pulled these numbers out of my ass, let's do 2 conv layers with 4 and 8 filters
# we're going to store them as tensors lol?
# the kernel will be 3x3
# layer 1: 1 input channel, 8 output channels
# so dim is 8x3x3?
# layer 2: 8 input channels, hm idk how many output
# this function returns a sequence (?) of tensors


def generate_random_weights():
    conv1 = generate_random_weights_for_layer(4, 1, 3, 3)
    conv2 = generate_random_weights_for_layer(8, 4, 3, 3)
    linear_transformation = torch.empty(24*24*8, 2)
    linear_transformation = nn.init.xavier_uniform_(
        linear_transformation, gain=nn.init.calculate_gain('relu'))
    linear_transformation.requires_grad = True

    return conv1, conv2, linear_transformation


def generate_random_weights_for_layer(out_channels, in_channels, w, h):
    # return torch.rand((out_channels, in_channels, w, h), requires_grad=True)
    w = torch.empty(out_channels, in_channels, w, h)
    w = nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    w.requires_grad = True
    return w


def my_relu(tensor):
    return torch.max(tensor, torch.zeros_like(tensor))


class WeirdCNN(nn.Module):
    def __init__(self, conv1, conv2, lt):
        super().__init__()
        # self.fuckyweights = generate_random_weights()
        self.conv1 = conv1
        self.conv2 = conv2
        self.linear_transformation = lt
        #  first dim is almost always batch
        # 2nd dim is sequence wtv that is
        # x y channel

    def forward(self, input):
        feature_maps_1 = F.conv2d(input, self.conv1)  # 26
        # apply relu
        feature_maps_1_1 = torch.relu(feature_maps_1)

        # layer 2
        feature_maps_2 = F.conv2d(feature_maps_1_1, self.conv2)  # 24
        feature_maps_2_1 = torch.relu(feature_maps_2)

        y3 = feature_maps_2_1.flatten(1)  # should be 16x(24x24x8)

        # you want this to be in the range of like -5 to 5
        y = torch.matmul(y3, self.linear_transformation)

        # softmax y
        y = torch.softmax(y, dim=1)
        return y  # LOL


def cursed_predictor(input_data, weights_sequence):
    model = WeirdCNN(*weights_sequence)
    # .forward is like the .__call__ ???
    return model(input_data)


def cursed_loss_function(input_data, weights_sequence):
    probabilities = cursed_predictor(input_data, weights_sequence)
    probabilities_top = probabilities[:, 0]

    # sort
    sorted_tensor, indices = torch.sort(probabilities_top, descending=True)
    # these are sorted from highest prob of top to lowest prob of top

    # split into 2 groups
    # top 8
    t8 = sorted_tensor[:8]
    # bottom 8
    b8 = sorted_tensor[8:]

    # compute loss (log things?? squares? maybe not direct difference? idfk)
    # simple cursed loss is
    # for top 8, you take their diff from 1
    # for bottom 8, you take their diff from 0
    # and you sum them

    # top 8
    top_loss = torch.sum(torch.abs(t8 - 1.0))
    # bottom 8
    bottom_loss = torch.sum(torch.abs(b8 - 0.0))

    return top_loss + bottom_loss


# now we want to train this thing
# we need to generate random weights
# we need to generate random input data
# we need to compute the loss
# we need to compute the gradient of the loss wrt the weights AND INPUT DATA
# we need to update the weights
# we need to repeat this process
# we need to do this for a lot of iterations (4 iterations ?!?!)


def cursed_train():
    data = generate_random_input_data()

    weights = generate_random_weights()
    params = [data, *weights]
    optimizer = torch.optim.SGD(params, lr=0.01)

    for i in range(100):
        for p in params:
            p.grad = None
        loss = cursed_loss_function(data, weights)
        loss.backward()
        optimizer.step()
        print(i, loss)


if __name__ == "__main__":
    cursed_train()
