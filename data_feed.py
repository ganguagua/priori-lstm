
import collections
import tokenization
import max_match
import tensorflow as tf
import numpy as np

#转成一个二分类，是否是一个词的结束，就不用crf做校验了

def generate_input_dataset(config, input_file="train.record", is_training=True):
    seq_len = config["max_seq_len"]
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_len], tf.int64),
        "prior_info": tf.io.FixedLenFeature([seq_len], tf.int64),
        "labels": tf.io.FixedLenFeature([seq_len], tf.int64)
    }
    batch_size = config["batch_size"]
    epoch = config["epoch"]
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.shuffle(buffer_size=20000, reshuffle_each_iteration=True)

    print("epoch: %d, batch_size: %d" % (epoch, batch_size))
    d = d.repeat(count=epoch)\
         .map(lambda record: tf.io.parse_single_example(record, name_to_features))\
         .batch(batch_size=batch_size, drop_remainder=True)

    return d

def one_hot(records):
    batch_label = []
    batch_input_ids = []
    batch_prior_info = []
    for record in records:
        features = eval(record)
        label = features['labels']
        prior_info = features['prior_info']
        input_ids = features["input_ids"]
        label = tf.one_hot(label, 4)
        prior_info = tf.cast(tf.one_hot(prior_info, 4), tf.float32)
        batch_label.append(label)
        batch_input_ids.append(input_ids)
        batch_prior_info.append(prior_info)
    return batch_input_ids, batch_prior_info, batch_label
        

def get_label(words):
    label = ["out"]
    for word in words:
        if len(word) == 0:
            continue
        if len(word) == 1:
            label.append("end")
        else:
            for ch in word[:-1]:
                label.append("not-end")
            label.append("end")
    label.append("out")
    return label

def truncate(words, max_seq_len):
    length = 0
    for index in range(len(words)):
        # placeholder for CLS&SEP
        if length + len(words[index]) <= max_seq_len-2:
            length += len(words[index])
        else:
            return words[:index]
    return words

def get_train_examples(file_name, max_seq_len):
    examples = []
    for line in open(file_name, 'r').readlines():
        line = line.strip("\ufeff")
        line = tokenization.convert_to_unicode(line)
        fields = line.strip().split()
        fields = truncate(fields, max_seq_len)
        text = "".join(fields)
        examples.append({"sentence": text, "words": fields})
    return examples

label2id = {"out":0, "end":1, "not-end":2}
def convert_label_to_id(label_list):
    label_ids = []
    for label in label_list:
        label_ids.append(label2id[label])
    return label_ids

def one_hot(values, depth):
    result = np.zeros([len(values), depth], dtype=int)
    for index in range(len(values)):
        result[values[index]] = 1
    return result

def load_vocab(vocab_file):
    dictionary = []
    for line in open(vocab_file, 'r').readlines():
        dictionary.append(line.strip())
    return dictionary

def convert_sentence_to_ids(tokenizer, sentence):
    tokens =  tokenizer.tokenize(sentence)
    tokens_with_border = ["[CLS]"]
    for token in tokens:
        tokens_with_border.append(token)
    tokens_with_border.append("[SEP]")
    input_ids = tokenizer.convert_tokens_to_ids(tokens_with_border)
    return input_ids

def file_based_convert_examples_to_features(examples, config, output_file="train.record", words_file="words"):
    '''
    init search tree
    '''
    max_seq_len = config["max_seq_len"]
    dictionary = load_vocab(words_file)
    search_tree = max_match.trie()
    search_tree.batch_insert(dictionary)
    writer = tf.compat.v1.python_io.TFRecordWriter(output_file)
    tokenizer = tokenization.FullTokenizer(vocab_file=config["bert_vocab"], do_lower_case=True)
    for example in examples:
        input_ids = convert_sentence_to_ids(tokenizer, example["sentence"])

        label = get_label(example['words'])
        match_words = search_tree.split_to_words(example["sentence"])
        match_words_label = get_label(match_words)
        while len(label) < max_seq_len:
            label.append("out")
            match_words_label.append("out")
            input_ids.append(0)

        assert len(label) == max_seq_len
        assert len(match_words_label) == max_seq_len
        assert len(input_ids) == max_seq_len

        features = collections.OrderedDict()
        features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids))
        features["labels"] = tf.train.Feature(int64_list=tf.train.Int64List(value=convert_label_to_id(label)))      
        features["prior_info"] = tf.train.Feature(int64_list=tf.train.Int64List(value=convert_label_to_id(match_words_label)))
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def test():
    test_data = "扬帆  远东  做  与  中国  合作  的  先行"
    test_data = "".join(test_data.split())
    dictionary = load_vocab("words")
    search_tree = max_match.trie()
    search_tree.batch_insert(dictionary)
    match_words = search_tree.split_to_words(test_data)
    print(match_words)

if __name__ == "__main__":
    test()   
