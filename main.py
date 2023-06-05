from Features.FeatureExtractor import FeaturesInfo
from Dataset.create_dataset import data
import numpy as np

ds = np.load('Dataset/dataset_feat.npy', allow_pickle=True)

print(ds[0].features)
print(ds[0].big_five)
