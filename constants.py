from enum import Enum

SHAPE = (768, 768)

MIN_SHAPE_SIZE = 12

CACHE_PATH = "./cache/"

DATA_PATH = "./data/"

IMAGE_PATH = f"{DATA_PATH}/images/"

class DataSplit(Enum): 
    TEST = f"{DATA_PATH}/test.csv"
    TRAIN = f"{DATA_PATH}/train.csv"
    VAL = f"{DATA_PATH}/val.csv"