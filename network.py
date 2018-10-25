import os
import tensorflow as tf
import shutil
import numpy as np
import random
from utils.data_reader import H5DataLoader
from utils import ops


class VGG16(object):

    def __init__(self, conf):
        self.conf = conf
        config = tf.ConfigProto()
        config.inter_op_parallelism_threads=self.conf.num_threads
        self.sess = tf.Session(config=config)
        latest_checkpoint_path = tf.train.latest_checkpoint(self.conf.modeldir)
        print('Lastest checkpoint find: ',  latest_checkpoint_path)
        if latest_checkpoint_path is not None:
            self.global_step = tf.Variable(int(latest_checkpoint_path.rsplit('-')[1]), name='global_step', trainable=False)
        else:
            self.global_step = tf.Variable(0,name='global_step',trainable=False)

        self.def_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        shutil.copy2('./main.py',os.path.join(self.conf.modeldir,'main.py'))
        shutil.copy2('./network.py', os.path.join(self.conf.modeldir,'network.py'))
        shutil.copy2('./utils/ops.py',os.path.join(self.conf.modeldir,'ops.py'))
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')
        if latest_checkpoint_path is not None:
            self.saver.restore(self.sess, latest_checkpoint_path)
        print('global step:',tf.train.global_step(self.sess,self.global_step))



    def def_params(self):
        self.data_format = 'NHWC'
        self.conv_size = (3, 3)
        self.pool_size = (2, 2)
        self.axis, self.channel_axis = (1, 2), 3
        self.input_shape = [
            self.conf.batch, self.conf.height, self.conf.width, self.conf.channel]
        self.output_shape = [self.conf.batch]

    def configure_networks(self):
        self.build_network()
        self.cal_loss()
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_placeholder)
        self.train_op = self.optimizer.minimize(self.loss_op, name='train_op',global_step=self.global_step)
        self.applied_learning_rate = self.conf.learning_rate
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def build_network(self):
        self.inputs = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.labels = tf.placeholder(
            tf.int64, self.output_shape, name='labels')
        self.preds = self.inference(self.inputs)

    def cal_loss(self):
        self.loss_op = tf.losses.sparse_softmax_cross_entropy(
            logits=self.preds, labels=self.labels, scope='loss/loss_op')
        self.loss_op += 0.0003*tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        with tf.variable_scope('accuracy'):
            self.decoded_preds = tf.argmax(self.preds, -1)
            self.accuracy_op = tf.reduce_mean(
                tf.cast(tf.equal(self.labels, self.decoded_preds), tf.float32))

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summarys.append(tf.summary.scalar(name+'/learning_rate',self.optimizer._lr))
        summary = tf.summary.merge(summarys)
        return summary

    def inference(self, inputs):
        outputs = inputs
        for layer_index in range(self.conf.network_depth):
            is_first = True if not layer_index else False
            is_last = True if layer_index == self.conf.network_depth-1 else False
            name = 'down%s' % layer_index
            outputs = self.build_down_block(
                outputs, name, layer_index,is_first, is_last)
        outputs = self.build_bottom_block(outputs, 'bottom')
        return outputs

    def build_down_block(self, inputs, name, layer_index=0, first=False, last=False):
        if first:
            num_outputs = self.conf.start_channel_num
        elif last:
            num_outputs = inputs.shape[self.channel_axis].value
        else:
            num_outputs = 2 * inputs.shape[self.channel_axis].value
        print("drop out keep probabilities on layer %d: "%(layer_index),self.conf.drop_outs[layer_index])
        conv1 = ops.conv2d(
            inputs, num_outputs, self.conv_size, name+'/conv1')
        conv1 = ops.dropout(conv1,self.conf.drop_outs[layer_index][0],name+'/dropout1')
        conv2 = ops.conv2d(
            conv1, num_outputs, self.conv_size, name+'/conv2')
        conv2 = ops.dropout(conv2,self.conf.drop_outs[layer_index][1],name+'/dropout2')
        if layer_index > 1:
            conv3 = ops.conv2d(conv2, num_outputs, self.conv_size, name+'/conv3')
            conv3 = ops.dropout(conv3,self.conf.drop_outs[layer_index][2],name+'/dropout3')
#             conv4 = ops.conv2d(conv3, num_outputs, self.conv_size, name+'/conv4',self.regularizer)
#             conv4 = ops.dropout(conv4,self.conf.drop_outs[layer_index][3],name+'/dropout4')
            pool = ops.pool2d(conv3, self.pool_size, name+'/pool')
            pool = ops.dropout(pool,self.conf.drop_outs[layer_index][3],name+'/dropout_pool')
        else:
            pool = ops.pool2d(conv2, self.pool_size, name+'/pool')
            pool = ops.dropout(pool,self.conf.drop_outs[layer_index][2],name+'/dropout_pool')
        return pool

    def build_bottom_block(self, inputs, name):
        outs = tf.contrib.layers.flatten(inputs, scope=name+'/flat')
        outs = ops.dense(outs, 512, name+'/dense1',activation_fn=tf.nn.relu)
        outs = ops.dropout(outs,0.5,name+'/dropout1')
        outs = ops.dense(outs, 512, name+'/dense2',activation_fn=tf.nn.relu)
        outs = ops.dropout(outs,0.5,name+'/dropout2')
        outs = ops.dense(outs, self.conf.class_num, name+'/dense_output',activation_fn=tf.nn.softmax)
        return outs

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def learning_rate_schedule(self,step):
        if step < 25000:
            return 1e-3
        elif step < 50000:
            return 1e-4
        elif step < 75000:
            return 1e-5
        else:
            return 1e-6

    def train(self):
        def random_flip(image):
            return random.choice([image,np.fliplr(image)])
        def random_crop(image):
            pad_width = ((4,4),(4,4),(0,0))
            image = np.lib.pad(image, pad_width=pad_width, mode='constant', constant_values=0)
            start_h = random.randint(0, 8)
            start_w = random.randint(0, 8)
            return image[start_h:start_h+32,start_w:start_w+32]
        train_reader = H5DataLoader(self.conf.data_dir+self.conf.train_data)
        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data)

        for epoch_num in range(tf.train.global_step(self.sess,self.global_step),self.conf.max_step+1):
            if epoch_num and epoch_num % self.conf.test_interval == 0:
                inputs, labels = valid_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.labels: labels,self.learning_rate_placeholder:self.learning_rate_schedule(epoch_num)}
                loss, summary = self.sess.run(
                    [self.loss_op, self.valid_summary], feed_dict=feed_dict)
                self.save_summary(summary,epoch_num)
                print('global step: %d; training loss %f'% (epoch_num,loss))
            if epoch_num and epoch_num % self.conf.summary_interval == 0:
                inputs, labels = train_reader.next_batch(self.conf.batch)
                inputs = np.array(list(map(random_flip,inputs)))
                inputs = np.array(list(map(random_crop,inputs)))
                feed_dict = {self.inputs: inputs, self.labels: labels,self.learning_rate_placeholder:self.learning_rate_schedule(epoch_num)}
                loss, _, summary = self.sess.run(
                    [self.loss_op, self.train_op, self.train_summary],
                    feed_dict=feed_dict)
                self.save_summary(summary, epoch_num)
            else:
                inputs, labels = train_reader.next_batch(self.conf.batch)
                inputs = np.array(list(map(random_flip,inputs)))
                inputs = np.array(list(map(random_crop,inputs)))
                feed_dict = {self.inputs: inputs, self.labels: labels,self.learning_rate_placeholder:self.learning_rate_schedule(epoch_num)}
                loss, _ = self.sess.run(
                    [self.loss_op, self.train_op], feed_dict=feed_dict)
                print('global step: %d; training loss %f'%( epoch_num,loss))
            if epoch_num and epoch_num % self.conf.save_interval == 0:
                self.save(epoch_num)


    def test(self):
        if self.conf.test_all:
            model_name = os.path.basename(self.conf.modeldir)
            if not os.path.exists('./test'):
                os.makedirs('./test')
            f = open(os.path.join('./test', model_name + '.csv'),"w+")
            print('testing all')
            latest_checkpoint_path = tf.train.latest_checkpoint(self.conf.modeldir)
            latest_checkpoint = int(latest_checkpoint_path.rsplit('-')[1])
            checkpoint = 1 + self.conf.save_interval
            while checkpoint <=latest_checkpoint:
                self.reload(checkpoint)
                accuracies = []
                test_reader = H5DataLoader(self.conf.data_dir+self.conf.test_data, False)
                while True:
                    inputs, labels = test_reader.next_batch(self.conf.batch)
                    if inputs is None or inputs.shape[0] < self.conf.batch:
                        break
                    feed_dict = {self.inputs: inputs, self.labels: labels}
                    accur = self.sess.run(self.accuracy_op, feed_dict=feed_dict)
                    accuracies.append(accur)
                f.write('%d, \t %f'%(checkpoint, sum(accuracies)/len(accuracies)))
                print('%d, \t %f'%(checkpoint, sum(accuracies)/len(accuracies)))
                checkpoint += self.conf.save_interval
            f.close()

        else:
            print('---->testing ', self.conf.test_step)
            if self.conf.test_step > 0:
                self.reload(self.conf.test_step)
            else:
                print("please set a reasonable test_step")
                return
            test_reader = H5DataLoader(
                self.conf.data_dir+self.conf.test_data, False)
            accuracies = []
            model_name = os.path.basename(self.conf.modeldir)
            test_dir = './test/'+model_name
            if not os.path.exists(test_dir):
                    os.makedirs(test_dir)
            index = 0
            wrong_preds = []
            debug_stat = [[0 for i in range (10)] for i in range(10)]
            while True:
                inputs, labels = test_reader.next_batch(self.conf.batch)
                if inputs is None or inputs.shape[0] < self.conf.batch:
                    break
                feed_dict = {self.inputs: inputs, self.labels: labels}
                accur,preds = self.sess.run([self.accuracy_op,self.decoded_preds], feed_dict=feed_dict)
                for i in range(len(preds)):
                    if preds[i] != labels[i]:
                        img = inputs[i]
                        img[:,:,0] = (img[:,:,0]*0.24703233 + 0.49139968)*255
                        img[:,:,1] = (img[:,:,1]*0.24348505+ 0.48215827)*255
                        img[:,:,2] = (img[:,:,2]*0.26158768 + 0.44653118)*255
                        img = np.array(np.uint8(img))
                        pil_image = Image.fromarray(img)
                        pil_image.save(os.path.join(test_dir,str(index) + '.png'))
                        debug_stat[int(labels[i])][int(preds[i])]+= 1
                        wrong_preds.append({index: {'label':int(labels[i]), 'pred':int(preds[i])}})
                        index += 1
                accuracies.append(accur)
            label_list = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            print('\t',end='')
            for item in label_list:
                print(item + '\t',end=' ')
            print("total \t percentage")
            for i in range(10):
                print(label_list[i],end='\t')
                for j in range(10):
                    print( debug_stat[i][j],end='\t')
                print("%d \t %f" %(sum(debug_stat[i]),sum(debug_stat[i])/sum([sum(i) for i in debug_stat]  )))

            json.dump(wrong_preds, open(os.path.join(test_dir, str(index) +'.json'),'w+'),sort_keys=True,indent=4)
            json.dump(debug_stat, open(os.path.join(test_dir, str(index) +'.json'),'w+'),sort_keys=True,indent=4)
            print('step: %d, accuracy %f'%(self.conf.test_step, sum(accuracies)/len(accuracies)))

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=self.global_step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)
