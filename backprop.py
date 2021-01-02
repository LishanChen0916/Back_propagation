import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))

def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return y * (1 - y)

def cost(y, y_hat):
    return y - y_hat

def der_cost(y, y_hat):
    return y - y_hat

class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        # Flatten the 'labels' to nx1 and return
        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        # Flatten the 'labels' to nx1 and return
        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor
        }[mode]

        return data_gen_func(n)

class layer:
    def __init__(self, n_in, n_out):
        '''random initalization of weights for this layer 
            inputs: 
                n_in: type: int
                n_out: type: int
        '''

        # Model parameters initialization
        # Set the 'w' and 'b' be a random small number with mean '0' and standard deviation '1'
        # b is set at dimension n_in+1
        self.w = np.random.normal(0, 1, (n_in + 1, n_out))

    def forward(self, x):
        # add 1 to last dimension is to calculate 'w', 'b' simultaneously
        x = np.append(x, np.ones((x.shape[0], 1)),axis=1)
        # can be seen as weight sum
        self.forward_gradient = x
        # y = sigmoid(w^T * x + b)
        self.y = sigmoid(np.matmul(x, self.w))
        return self.y

    def backward(self, derivative_C):
        self.deltas = np.multiply(der_sigmoid(self.y), derivative_C)
        return np.matmul(self.deltas, self.w[:-1].T)

    def update(self, eta):
        self.gradient = np.matmul(self.forward_gradient.T, self.deltas)
        self.w -= eta * self.gradient
        return self.gradient


class SimpleNet:
    def __init__(self, sizes, learning_rate=0.1, num_step=2000, print_interval=100):
        """ A hand-crafted implementation of simple network.

        Args:
            sizes:    the sizes of neurons per layer.
            num_step (optional):    the total number of training steps.
            print_interval (optional):  the number of steps between each reported number.
        """
        self.num_step = num_step
        self.print_interval = print_interval
        
        self.eta = learning_rate
        self.convergence = False

        # add 0 to last dimension as terminate condition
        hidden_size = sizes[1:] + [0]
        self.layers = []
        for n_in, n_out in zip(sizes, hidden_size):
            if (n_in+1)*n_out == 0:
                continue
            self.layers += [layer(n_in, n_out)]

    def feedforward(self, x):
        activ = x
        for l in self.layers:
            activ = l.forward(activ)
        return activ

    def backpropagate(self, dC):
        _dC = dC
        for l in self.layers[::-1]:
            _dC = l.backward(_dC)
    
    def update(self):
        for l in self.layers:
            l.update(self.eta)

    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        assert data.shape[0] == gt_y.shape[0]
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()

    def train(self, inputs, labels):
        """ The training routine that runs and update the model.

        Args:
            inputs: the training (and testing) data used in the model.
            labels: the ground truth of correspond to input data.
        """
        # make sure that the amount of data and label is match
        assert inputs.shape[0] == labels.shape[0]

        n = inputs.shape[0]

        for epochs in range(self.num_step):
            for idx in range(n):
                # operation in each training step:
                #   1. forward passing
                #   2. compute loss
                #   3. propagate gradient backward to the front
                self.output = self.feedforward(inputs[idx:idx+1, :])
                self.error = cost(self.output ,labels[idx:idx+1, :])
                self.backpropagate(der_cost(self.output, labels[idx:idx+1, :]))
                self.update()

            if epochs % self.print_interval == 0:
                print('Epochs {}: '.format(epochs))
                self.test(inputs, labels)

            if self.convergence:
                print('Network is convergence.')
                break

        print('Training finished')
        self.test(inputs, labels)

    def test(self, inputs, labels):
        """ The testing routine that run forward pass and report the accuracy.

        Args:
            inputs: the testing data. One or several data samples are both okay.
                The shape is expected to be [BatchSize, 2].
            labels: the ground truth correspond to the inputs.
        """
        n = inputs.shape[0]

        error = 0.0
        error_threshold = 0.005
        for idx in range(n):
            result = self.feedforward(inputs[idx:idx+1, :])
            error += abs(result - labels[idx:idx+1, :])

        error /= n
        if error < error_threshold:
            self.convergence = True

        print(" Loss : %.10f" % error)
        print(' Accuracy : %.2f' % ((1 - error)*100) + '%')
        print('')


if __name__ == '__main__':
    data, label = GenData.fetch_data('Linear', 70)
    
    net = SimpleNet(sizes=[2, 4, 4, 1], 
                    num_step=10000,
                    print_interval=500,
                    learning_rate=0.1)

    net.train(data, label)

    pred_result = np.round(net.feedforward(data))

    print("Label : ")
    print(label)
    print("Prediction : ")
    print(pred_result)

    SimpleNet.plot_result(data, label, pred_result)