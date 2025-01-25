import numpy as np
import pandas as pd
from surprise import KNNWithMeans, Dataset, accuracy, Reader, SVD, SVDpp, NMF
from surprise.model_selection import KFold

def collaborative_filtering(trainset):
    df = pd.DataFrame(trainset, columns=['user_id', 'item_id', 'rating'])
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df, reader)
    trainset = dataset.build_full_trainset()
    model = SVDpp(random_state=42)
    model.fit(trainset)
    return model
