from data_loader import PicaiLoader
import jax.numpy as np








def train():
    
    data_set = PicaiLoader()
    epochs = 20

    model = None

    batch = data_set.get_batch(6)
    pass



if __name__ == '__main__':
    train()