import requests
import numpy as np
import pickle
from src.text import translate_table
import tensorflow as tf

METHOD = "predict"
VERSION = "1"
URL = f"http://localhost:8501/v{VERSION}/models/category_classifier:{METHOD}"

instances = ["nintendo 64", "tv lg c2",
            "samsung s5",
         "5200mAh Auto Detect Portable Charger External Battery Power Bank with LED light For iPhone 6 Plus 5S iPad Mini Samsung Galaxy S6 edge S5 S4 S3 Note 4 3 HTC Sony Most 5V Smart phones and Tablet (Blue)"]

response = requests.post(URL, json={"instances": instances})

if response.status_code == 200:
    predictions = response.json()["predictions"]
    for sample in zip(instances, predictions):
        title, logits = sample
        pred_index = tf.argmax([logits], axis=1).numpy()
        print("Title: ", title)
        print("category prediction: ", translate_table.lookup(tf.constant([pred_index])))
        print("---------------------------------------------------------------------------")
else:
    print(response.status_code)
