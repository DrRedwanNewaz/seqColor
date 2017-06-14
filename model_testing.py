import tensorflow as tf
import model
from data_processing import data_base
import numpy as np
from update_cmap import DataBase


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
        SEQ_LENGTH, START_TOKEN,learning_rate=LEARNING_RATE)


def main():
    # get testing set
    id =5
    d_test = data_base([id])
    words = np.array(d_test.read_files())
    num_words = len(words)
    pred = get_trainable_model(num_words)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    print ("restoring session ...")
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "../gru_model/code_demo_model.ckpt")
        #     Tesing model
        print ("Testing model ...")
        # seq = np.random.rand(3)


        # np.savetxt("result.txt",prediction)
        color=[]
        seq = np.array(words[1,:]).transpose()
        total_g_loss = 0
        for i in range(num_words):
            _, g_loss, g_pred = pred.pretrain_step(sess, seq)
            _, d_loss = pred.train_d_real_step(sess, seq)
            g_pred =np.array(g_pred).transpose() # convert into np array
            # g_pred =g_pred[1,:]
            g_pred =g_pred[np.random.randint(0,len(g_pred)),:]
            # g_pred = np.mean(g_pred,axis=0) #does not work
            if abs(d_loss-g_loss)>0.1:
                i=i-1
                continue
            color.append(g_pred)
            # use the same color as a input
            seq = g_pred.transpose()
            total_g_loss =total_g_loss+g_loss
            if(i%100==0):
                print("iteration {} out of {} g_loss {}".format(i,num_words,total_g_loss))
        np.savetxt("result/prediction.txt",color)
        obj = DataBase(data=np.array(color), driver_id=id)
        obj.show()
        obj.view()

    # show original figure
    obj2=DataBase(data=np.loadtxt("output2/driver_%d_map.txt"%id,delimiter=","),driver_id=id)
    obj2.show()
    obj2.view()


    # obj =rgb_converter(np.loadtxt("result.txt"))
    # obj.show()




if __name__ == '__main__':
    main()
