from src.constants import CLASSES
import tensorflow as tf
import re
import string

def standardize_method(inputs):
    strip_chars = re.escape(string.punctuation)

    lowercase = tf.raw_ops.StringLower(input=inputs, encoding="utf-8")
    escaped_chars = tf.raw_ops.StaticRegexReplace(input=lowercase, pattern=f"[{strip_chars}]", rewrite="")
    return tf.raw_ops.StaticRegexReplace(input=escaped_chars, pattern="[0-9]+", rewrite="[NUMBER]")

classes_table = tf.lookup.StaticHashTable(
                            tf.lookup.KeyValueTensorInitializer(CLASSES, range(len(CLASSES))),
                            default_value=-1
                        )

translate_table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=range(len(CLASSES)),
        values=CLASSES,
    ),
    default_value=tf.constant("N/A"),
    name="main_categories"
)