import os
import time
from random import shuffle

from keras.models import load_model
from PIL import Image
import tensorflow as tf
import numpy as np

model = load_model("model.h5")

base_dir = os.path.join("DataSet", "fit")


def make_square(img, min_size=224, fill_color=(255, 255, 255, 0)) -> Image:
    """
    Делает картинку квадратной с сохранением пропорций
    """
    output_size = min_size, min_size
    x, y = img.size
    max_size = max(x, y)
    coeff = min_size / max_size
    x, y = round(x * coeff), round(y * coeff)
    img = img.resize((x, y), 1)
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    new_im.thumbnail(output_size, Image.ANTIALIAS)
    return new_im


def img2vec_that_saves_proportions(image_path):
    return tf.keras.preprocessing.image.img_to_array(img=make_square(Image.open(image_path)))


all_labels = os.listdir(base_dir) # укажи пожалуйста в base_dir дерикторию в которой лежат фотки. не придется руками забивать лейблы
# SELECT * from labels

input_tensor = np.expand_dims(img2vec_that_saves_proportions("image.jpg"), 0) #надо сделать из фотки тензор


# next_step_point это минимальное количество фоток для следующего шага
def optimized_output(input_tensor, next_step_accuracy_percent=0.5, next_step_point=5, comparing_percent=0.8):
    matching_items = []
    for label in all_labels:
        currently_accepted_items = []
        comparing_list = []
        all_vectors = Vectors.select().where(Vectors.type == label)
        shuffle(all_vectors)
        for v in all_vectors:
            vec = np.frombuffer(v.vector, dtype=='float32').reshape((224,224,3))
            comparing_tensor = np.expand_dims(vec, 0)
            compare_result = model.predict([input_tensor, comparing_tensor])[0][0]
            comparing_list.append(compare_result)
            if compare_result > comparing_percent:
                currently_accepted_items.append({"url": v.url, "type": label})
            if len(comparing_list) > next_step_point and sum(comparing_list) / len(comparing_list) < next_step_accuracy_percent:
                break
        matching_items.extend(currently_accepted_items)
    return matching_items
