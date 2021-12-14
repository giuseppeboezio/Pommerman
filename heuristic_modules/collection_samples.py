import pandas as pd
import os
import numpy as np


def convert_to_images(X, y, pth_images, pth_annotations):

    for counter, value in enumerate(X):

        # save the input
        np.savetxt(os.path.join(pth_images, f'{counter}.csv'), value.reshape((5,-1)), delimiter=',')

    names = [f"{i}.csv" for i in range(len(y))]

    # creation of the dataframe
    df = pd.DataFrame({"name": names, "label": y})
    df.to_csv(pth_annotations, index=False)
