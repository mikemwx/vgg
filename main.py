import os
import time
import argparse
import tensorflow as tf
from network import VGG16


def configure(model_id):
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', 10000, '# of step for training')
    flags.DEFINE_integer('test_interval', 100, '# of interval to test a model')
    flags.DEFINE_integer('save_interval', 10, '# of interval to save model')
    flags.DEFINE_integer('summary_interval', 100, '# of step to save summary')
    flags.DEFINE_float('learning_rate', 1e-2, 'learning rate')
    # data
    flags.DEFINE_string('data_dir', './dataset/', 'Name of data directory')
    flags.DEFINE_string('train_data', 'cifar10train.h5', 'Training data')
    flags.DEFINE_string('valid_data', 'cifar10valid.h5', 'Validation data')
    flags.DEFINE_string('test_data', 'cifar10valid.h5', 'Testing data')
    flags.DEFINE_integer('batch', 16, 'batch size')
    flags.DEFINE_integer('channel', 3, 'channel size')
    flags.DEFINE_integer('height', 32, 'height size')
    flags.DEFINE_integer('width', 32, 'width size')
    # Debug
    flags.DEFINE_string('logdir', './logdir/'+model_id, 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir/'+model_id, 'Model dir')
    flags.DEFINE_string('sampledir', './samples/'+model_id, 'Sample directory')
    flags.DEFINE_string('model_name', 'model_'+model_id, 'Model file name')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('test_step', 0, 'Test or predict model at this step')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    # network architecture
    flags.DEFINE_integer('network_depth', 5, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 10, 'output class number')
    flags.DEFINE_integer('start_channel_num', 64,
                         'start number of outputs for the first conv layer')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, test')
    parser.add_argument('--model_id',type=str)
    args = parser.parse_args()
    if args.action not in ['train', 'test']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test")
    else:
        model = VGG16(configure(args.model_id))
        getattr(model, args.action)()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    tf.app.run()
