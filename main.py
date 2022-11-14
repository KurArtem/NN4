from network import PANetwork
from datasets import ImagesDataset

network = PANetwork(1, 16, 3, 16, dbg=True)
dataset = ImagesDataset("D:\\DATASETS\\Fruits\\", 0.6, 8, reversed_colors=True, dbg=True)
network.train(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, 0.8, ages=1000)
hit_rate, loss = network.test(dataset.x_test, dataset.y_test)
print("Test split\n hit rate: ", str(hit_rate), ";\n loss: ", str(loss))
