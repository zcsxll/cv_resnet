import sys
sys.path.append("../")

from dataset import create_dataset

def create_data_loader(name, mode, mean_var = None, batch_size = 32, shuffle = True, num_workers = 32):
    dataset = create_dataset(name)
    