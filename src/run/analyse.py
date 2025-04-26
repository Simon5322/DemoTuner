import matplotlib.pyplot as plt


def draw(data):
    epochs = list(range(1, len(data)+1))

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, data)

    plt.xlabel('Epochs')
    plt.ylabel('Throughput')


class Analyse():
    def __init__(self):
        self.action = {}
        self.resultArray = []

