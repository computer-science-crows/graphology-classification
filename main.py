from Dataset.create_dataset import build_dataset
from Algorithms.KMeans import k_means

dataset = build_dataset()
print(dataset[0][1])
k_means(dataset)
