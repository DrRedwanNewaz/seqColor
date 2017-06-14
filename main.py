import numpy as np
import model
import train
import tensorflow as tf
import random
from data_processing import data_base

EMB_DIM = 20
HIDDEN_DIM = 25
SEQ_LENGTH = 3
START_TOKEN = 0

EPOCH_ITER = 3000
CURRICULUM_RATE = 0.02  # how quickly to move from supervised training to unsupervised
TRAIN_ITER = 100000  # generator/discriminator alternating
D_STEPS = 3  # how many times to train the discriminator per generator step
LEARNING_RATE = 0.01 * SEQ_LENGTH
SEED = 88



class BookGRU(model.GRU):

    def d_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate


def get_trainable_model(num_emb):
    return BookGRU(
        num_emb, EMB_DIM, HIDDEN_DIM,
        SEQ_LENGTH, START_TOKEN)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    d_train = data_base([1, 2, 3, 4])
    d_test = data_base([5])
    # print train.data_size()


    assert START_TOKEN == 0
    words = d_test.read_files()
    word2idx = []
    for word in words:
        word2idx.append(d_train.color2ind(word))
    num_words = len(words)
    # three_grams = dict((tuple(word2idx[w] for w in token_stream[i:i + 3]), True)
    #                    for i in xrange(len(token_stream) - 3))
    # print 'num words', num_words
    # print 'num indx', len(word2idx)
    # print 'distinct 3-grams', len(three_grams)

#     # saving model
#
    trainable_model = get_trainable_model(num_words)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    for epoch in range(TRAIN_ITER // EPOCH_ITER):
        print('epoch= %d'% epoch)
        proportion_supervised = max(0.0, 1.0 - CURRICULUM_RATE * epoch)
        train.train_epoch(
            sess, trainable_model, EPOCH_ITER,
            proportion_supervised=proportion_supervised,
            g_steps=1, d_steps=D_STEPS,
            next_sequence=lambda :np.random.rand(3),
            verify_sequence=lambda seq:d_train.is_valid(seq),
            words=words)

    save_path = saver.save(sess, "code_demo_model.ckpt")
    print('model saved in file {}'.format(save_path))





if __name__ == '__main__':
    main()
