
import tensorflow as tf
import argparse

'''
#使用方法示例：python .\test_FLAGS.py --train_dir '/amui/haha' --max_step 10 就会将的参数值改变

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/home/ccx/AmuiData/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")



if __name__ == '__main__':
    print(FLAGS.train_dir)
    print(FLAGS.max_steps)

    with tf.variable_scope(""):
        images = tf.get_variable("images", [1,224,224,3], initializer = tf.constant_initializer(2.0))
        weight = tf.get_variable("kernel", [3,3,3,10], initializer = tf.constant_initializer(3.0))
        conv1 = tf.nn.conv2d(images, weight, strides=[1,2,2,1], padding='VALID')
        conv2 = tf.nn.conv2d(images, weight, strides=[1,2,2,1], padding='SAME')

    print(conv1.get_shape())
    print(conv2.get_shape())
    
'''

'''
a = (3,3,2)
shape = [1]
shape.extend(a)
b = tf.placeholder(tf.float32, shape=shape, name='input_images')

print (b.get_shape()[-1])

w1 = tf.get_variable(
            name='w1',
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w1))
'''
'''
with tf.variable_scope(""):
    images = tf.get_variable("images", [1, 8, 8, 40], initializer=tf.constant_initializer(2.0))
    weight = tf.get_variable("kernel", [1, 1, 40, 20], initializer=tf.constant_initializer(3.0))
    conv1 = tf.nn.conv2d(images, weight, strides=[1, 1, 1, 1], padding='VALID')
    #conv2 = tf.nn.conv2d(images, weight, strides=[1, 1, 1, 1], padding='SAME')

    ksize = [1, 8, 8, 1]
    strides = [1, 8, 8, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(images, ksize, strides, padding)
    features_total = int(output.get_shape()[-1])
    output2 = tf.reshape(output, [-1, features_total])
print(conv1.get_shape())
#print(conv2.get_shape())
#print(output.get_shape())
#print(output2.get_shape())
'''


class A:
    def __init__(self, f, b=0, a=0, **x):
        self.a = a
        b = b
        f = f
        # self.e = e # wrong
        self.e = x['e']
        self.fun1(3)

        print('a', self.a)
        print('b', b)
        print('c', self.c)
        print('e', self.e)

        print('f', f)

    def fun1(self,c = 0):
        self.c = c

parser = argparse.ArgumentParser()

parser.add_argument(
        '--e', type=int, metavar='G', default=4,
        help='1x1 group convolution (default: 4)')

parser.add_argument(
        '--f', type=int, metavar='G', default=8,
        help='1x1 group convolution (default: 4)')

args = parser.parse_args()
args1 = vars(args)
a = A(a=1,b=2,**args1)
#print(a.a)
'''
epoch=1

print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')

growth = "3-6-5-8"
growth = list(map(int, growth.split('-')))
num = len(growth)
print(num)
'''