import os
import tensorflow as tf
from utils.data_reader import H5DataLoader
from utils import ops


class VGG16(object):

    def __init__(self, conf):
        self.conf = conf
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.conf.gpu_fraction
        config.inter_op_parallelism_threads=self.conf.num_threads
        self.sess = tf.Session(config=config)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.def_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')
        latest_checkpoint_path = tf.train.latest_checkpoint(self.conf.modeldir)
        print(latest_checkpoint_path)
        if latest_checkpoint_path is not None:
            self.saver.restore(self.sess, latest_checkpoint_path)
            self.global_step = self.global_step + int(latest_checkpoint_path.rsplit('-')[1])
        print(tf.train.global_step(self.sess,self.global_step))
            
        

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
        with tf.variable_scope('accuracy'):
            self.decoded_preds = tf.argmax(self.preds, -1)
            self.accuracy_op = tf.reduce_mean(
                tf.cast(tf.equal(self.labels, self.decoded_preds), tf.float32))

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
#         summarys.append(tf.summary.scalar(name+'/learning_rate',self.optimizer._lr))
        summarys.append(tf.summary.scalar(name+'/learning_rate',self.learning_rate_placeholder))
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
        print("drop outs: ",self.conf.drop_outs[layer_index])
        conv1 = ops.conv2d(
            inputs, num_outputs, self.conv_size, name+'/conv1',)
        conv1 = ops.dropout(conv1,self.conf.drop_outs[layer_index][0],name+'/dropout1')
        conv2 = ops.conv2d(
            conv1, num_outputs, self.conv_size, name+'/conv2',)
        conv2 = ops.dropout(conv2,self.conf.drop_outs[layer_index][1],name+'/dropout2')
        if layer_index > 1:
            conv3 = ops.conv2d(conv2, num_outputs, self.conv_size, name+'/conv3',)
            conv3 = ops.dropout(conv3,self.conf.drop_outs[layer_index][2],name+'/dropout3')
            pool = ops.pool2d(conv3, self.pool_size, name+'/pool')
            pool = ops.dropout(pool,self.conf.drop_outs[layer_index][3],name+'/dropout_pool')
        else:
            pool = ops.pool2d(conv2, self.pool_size, name+'/pool')
            pool = ops.dropout(pool,self.conf.drop_outs[layer_index][2],name+'/dropout_pool')
        return pool

    def build_bottom_block(self, inputs, name):
        outs = tf.contrib.layers.flatten(inputs, scope=name+'/flat')
        outs = ops.dense(outs, 1024, name+'/dense1',activation_fn=tf.nn.relu)
        outs = ops.dropout(outs,0.5,name+'/dropout1')
#         outs = ops.dense(outs, 4096, name+'/dense2',activation_fn=tf.nn.relu)
#         outs = ops.dropout(outs,0.5,name+'/dropout2')
        outs = ops.dense(outs, self.conf.class_num, name+'/dense_output',activation_fn=tf.nn.softmax)
        return outs

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
#         print("reload step %d" %(self.conf.reload_step))
#         if self.conf.reload_step > 0:
#             self.reload(self.conf.reload_step)
#             starting_epoch = self.conf.reload_step 
#         else:
#             starting_epoch = 0
        train_reader = H5DataLoader(self.conf.data_dir+self.conf.train_data)
        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data)
        for epoch_num in range(tf.train.global_step(self.sess,self.global_step),self.conf.max_step+1):  
            if epoch_num % 5000 is 0 and epoch_num:
                self.applied_learning_rate *= 0.5
                print(self.applied_learning_rate)
            if epoch_num and epoch_num % self.conf.test_interval == 0:
                inputs, labels = valid_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.labels: labels,self.learning_rate_placeholder: self.applied_learning_rate}
                loss, summary = self.sess.run(
                    [self.loss_op, self.valid_summary], feed_dict=feed_dict)
                self.save_summary(summary,epoch_num)
                print('global step: %d; training loss %f'% (epoch_num,loss))
            if epoch_num and epoch_num % self.conf.summary_interval == 0:
                inputs, labels = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.labels: labels,self.learning_rate_placeholder:self.applied_learning_rate}
                loss, _, summary = self.sess.run(
                    [self.loss_op, self.train_op, self.train_summary],
                    feed_dict=feed_dict)
                self.save_summary(summary, epoch_num)
            else:
                inputs, labels = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.labels: labels,self.learning_rate_placeholder:self.applied_learning_rate}
                loss, _ = self.sess.run(
                    [self.loss_op, self.train_op], feed_dict=feed_dict)
                print('global step: %d; training loss %f'%( epoch_num,loss))
            if epoch_num and epoch_num % self.conf.save_interval == 0:
                self.save(epoch_num)
  

    def test(self):
        print('---->testing ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        test_reader = H5DataLoader(
            self.conf.data_dir+self.conf.test_data, False)
        accuracies = []
        while True:
            inputs, labels = test_reader.next_batch(self.conf.batch)
            if inputs is None or inputs.shape[0] < self.conf.batch:
                break
            feed_dict = {self.inputs: inputs, self.labels: labels}
            accur = self.sess.run(self.accuracy_op, feed_dict=feed_dict)
            accuracies.append(accur)
        print('accuracy is ', sum(accuracies)/len(accuracies))

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
