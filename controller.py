
import data_feed
import json
import lstm
import tensorflow as tf
import sys
from copy import deepcopy

class data_stream:
    def __init__(self, dataset):
        self.dataset = dataset
    def iterator(self):
        for elem in self.dataset:
            yield elem 

def get_dataset(mode, origin_config):
    config = deepcopy(origin_config)
    if mode == "dev":
        config["epoch"] = 100000
    if mode == "test":
        config["epoch"] = 1
    examples = data_feed.get_train_examples(config[mode+"_file"], config["max_seq_len"])
    data_feed.file_based_convert_examples_to_features(examples, config, output_file=mode+".record")
    if mode == "test":
        is_training = False
    else:
        is_training = True
    input_dataset = data_feed.generate_input_dataset(config, is_training=is_training, input_file=mode+".record")
    return input_dataset

def eager_train(config):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = lstm.lstm(config)
    train_dataset = get_dataset("train", config)
    test_dataset = iter(get_dataset("dev", config))
    test_data_iterator = data_stream(test_dataset).iterator()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    #加载预训练模型
    embedding_table = tf.train.load_variable("./chinese_L-12_H-768_A-12", "bert/embeddings/word_embeddings")
    model.embedding_table.assign(embedding_table)
    step = 0
    writer = tf.summary.create_file_writer("tf.log")
    for batch_data in train_dataset:
        with tf.GradientTape() as tape:
            prob, loss = model.predict(batch_data)
        train_variables = model.trainable_variables
        gradients = tape.gradient(target=loss, sources=train_variables)
        optimizer.apply_gradients(zip(gradients, train_variables))
        step += 1
        if step % 1000 == 0:
            print(loss)
        if step % 20000 == 0:
            checkpoint.save("./output/lstm-%d" % step)
        test_data = next(test_dataset)
        _, test_loss = model.predict(test_data)
        with writer.as_default():
            tf.summary.scalar("loss", loss, step)
            tf.summary.scalar("test_loss", test_loss, step)
        writer.flush()
    checkpoint.save("./output/final") 

def predict(config):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = lstm.lstm(config)
    dataset = get_dataset("test", config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    status = checkpoint.restore(tf.train.latest_checkpoint("./output/"))
    status.assert_existing_objects_matched()
    writer = open('result', 'w')
    for item in dataset:
        prob, _ = model.predict(item, False)
        res = tf.math.argmax(prob, -1)
        for label in list(res.numpy()):
            writer.write(str(list(label))+"\n")
    writer.close()
def test(config):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = lstm.lstm(config)
    train_dataset = get_dataset("test", config)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore()
        
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def test():
    config = load_config("lstm.conf")
    model = lstm.LSTM(config)
    get_loss_from_file("test", config, model) 

max_seq_len=50

if __name__ == "__main__":
    config = load_config("lstm.conf")
    max_seq_len = config["max_seq_len"]
    if sys.argv[1] == "train":
        eager_train(config)
    elif sys.argv[1] == "predict":
        predict(config)
