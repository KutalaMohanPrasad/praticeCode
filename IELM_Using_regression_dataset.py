import numpy as np
from time import process_time
import math
from matplotlib import pyplot as plt
import PSwarm

b = np.random.normal()  # keeping the global random value of b


def sigmoid(x):
    calculate = 1 / (1 + np.exp(-x))
    return calculate


def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def test_Regression_data(test_data, test_labels, beta, w):
    temp_h = np.dot(test_data, w)
    b = np.random.normal()
    temp_h = temp_h + b
    h = sigmoid(temp_h)

    y = np.dot(h, beta)
    # Calculate testing accuracy (RMSE) for regression case
    rmse = np.square(np.subtract(test_labels, y).mean())
    tot_test_rmse = math.sqrt(rmse)

    print("Total testing RMSE is: ", tot_test_rmse)


def train_regression_data(train_data, train_labels):
    hidden_neurons = 10
    actual_weight = []
    beta = []
    tot_train_rmse = 0
    total_rmse = np.zeros(hidden_neurons)
    sample = 0
    w_range = (3.0, 8.0)
    lr_range = (4.0, 9.0)
    iw_range = (1.9, 8.9)
    c = (0.5, 0.3)
    no_solution = 100
    no_dim = 1000
    for l in range(1, hidden_neurons + 1):
        sample = l
        # w=random input data
        w = np.random.normal(size=(train_data.shape[1], l))
        actual_weight = w
        # temp_h=a*w+b
        temp_h = np.dot(train_data, w) + b
        # applying sigmoid or sin activation fucntion for temp_h
        h = sigmoid(temp_h)
        # updating the beta values
        # beta=h'*train_labels
        beta = np.dot(np.linalg.pinv(h), train_labels)
        # predicted output y=h*beta
        y = np.dot(h, beta)
        # calculating the RMSE for each neuron of train data
        rmse = np.square(np.subtract(train_labels, y).mean())
        # removing that error form initial data to avoid overlapping
        train_labels = train_labels - y
        tot_train_rmse = math.sqrt(rmse)

        total_rmse[l - 1] = tot_train_rmse
        # printing the RMSE for each hidden neuron
        print("RMSE for hidden neuron : ", l, "is: ", tot_train_rmse)

        obj = PSwarm.PSwarm(no_solution, no_dim, w_range, lr_range, iw_range, c)

        tot_pso = obj.optimize(l, y)
        tot_pso=math.sqrt(tot_pso)

    if not sample < hidden_neurons:
        print("\nTotal training RMSE : ", tot_train_rmse)
        print("\n Total global best : ", tot_pso , "%")

    print("Plotting the learning curve ...... \n")
    plt.title("Learning Curve For Stock dataset 2017-2018")
    plt.xlabel("Number of Neurons")
    plt.ylabel("RMSE")
    plt.plot(total_rmse)
    plt.show()

    return beta, actual_weight


if __name__ == "__main__":
    print("Reading the regression file")
    # Using the delta elevators dataset for regression
    data = np.genfromtxt('delta_elevators.data')
    data = data[:]
    #print(data)
    data = norm_data(data)
    input_data = data[:, :-1]
    sample_output = data[:, -1]
    output_data = np.array([list([row]) for row in sample_output])
    # as per previous assignment feed back.. removing the overlapping between train and testing data
    # 50% for training and remaining 50% for testing
    train_percent = 50

    test_percent = 100 - train_percent
    train_data = int((output_data.shape[0] * train_percent) / 100)
    test_data = int((output_data.shape[0] * test_percent) / 100)
    TrainData = input_data[:train_data, :]
    TrainLabel = output_data[:train_data, :]
    train_start = process_time()
    beta, a = train_regression_data(TrainData, TrainLabel)
    train_end = process_time()
    print("Training time for Regression :", (train_end - train_start), " seconds")

    # testing the regression data set
    TestData = input_data[test_data:, :]
    TestLabel = output_data[test_data:, :]
    test_start = process_time()
    test_Regression_data(TestData, TestLabel, beta, a)
    test_end = process_time()

    print("Testing time for Regression:", (test_end - test_start), " seconds")


