from PIL import Image
import pandas as pd
import os


def convert_to_images(X, y, pth_images, pth_annotations):

    for counter, value in enumerate(X):

        # save the image
        im = Image.fromarray(value)
        im.save(os.path.join(pth_images, f'{counter}.png'))

    names = [f"{i}.png" for i in range(len(y))]

    # creation of the dataframe
    df = pd.DataFrame({"name": names, "label": y})
    df.to_csv(pth_annotations)
