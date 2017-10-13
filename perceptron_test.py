#!/usr/bin/python

from argparse import ArgumentParser
from random import random

from perceptron import *

if __name__ == "__main__":
    print("Hello!")

    parser = ArgumentParser()
    parser.add_argument("epoch_count", type=int, help="Number of training epochs")
    parser.add_argument("learning_rate", type=float, help="Learning rate")
    args = parser.parse_args()

    epoch_count = args.epoch_count
    learning_rate = args.learning_rate

    print("Building network")
    input_count = 3
    hidden_count = 10
    output_count = 1

    input_layer = Layer()
    hidden_layer = Layer()
    output_layer = Layer()
    
    input_layer.index = 0
    hidden_layer.index = 1
    output_layer.index = 2

    input_layer.add_nodes(input_count)
    hidden_layer.add_nodes(hidden_count)
    output_layer.add_nodes(output_count)

    hidden_layer.connect_nodes_from_layer(input_layer)
    output_layer.connect_nodes_from_layer(hidden_layer)

    input_layer.nodes[0].value = 0.5
    input_layer.nodes[1].value = 0.5
    input_layer.nodes[2].value = 0.5

    hidden_layer.randomize_weights()
    output_layer.randomize_weights()

    print("Training network: {0} epochs, {1} learning rate".format(
            epoch_count, learning_rate))

    last_percentage = 0
    for epoch in range(epoch_count):
        percentage = (epoch * 100) / epoch_count

        if percentage != last_percentage and percentage % 10 == 0:
            print("{0}% complete".format(percentage))

        last_percentage = percentage

        inputs = [random(), random(), random()]
        expected_output = (1.0 - inputs[0]) * (inputs[1]) * (1.0 - inputs[2])

        input_layer.nodes[0].value = inputs[0]
        input_layer.nodes[1].value = inputs[1]
        input_layer.nodes[2].value = inputs[2]

        hidden_layer.forward()
        output_layer.forward()

        predicted_output = output_layer.nodes[0].value
        errors = [expected_output - predicted_output]

        output_layer.backward(errors, learning_rate)
    
    test_count = 1000
    average_error = 0
    max_error = 0.0
    min_error = 1.0

    print("Testing network")
    for test in range(test_count):
        #print("Running test #{0}".format(test + 1))

        inputs = [random(), random(), random()]
        expected_output = (1.0 - inputs[0]) * (inputs[1]) * (1.0 - inputs[2])

        #print("Input / Expected Output: {0} / {1}".format(inputs, expected_output))

        input_layer.nodes[0].value = inputs[0]
        input_layer.nodes[1].value = inputs[1]
        input_layer.nodes[2].value = inputs[2]

        hidden_layer.forward()
        output_layer.forward()

        predicted_output = output_layer.nodes[0].value
        output_error = expected_output - predicted_output

        if test % 100 == 0:
            print("Predicted / Expected output: {0} / {1} ({2})".format(
                    predicted_output, expected_output, output_error))

        average_error += abs(output_error)
        max_error = max(max_error, abs(output_error))
        min_error = min(min_error, abs(output_error))

    average_error /= (test_count * 1.0)
    print("Min / Max error: {0} / {1}".format(min_error, max_error))
    print("Average error on test: {0}".format(average_error))

    print("Goodbye!")

