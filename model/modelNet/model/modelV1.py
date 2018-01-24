import os
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf


TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class CondenseNet:
    def __init__(self, data_provider, growth, depth,
                 total_blocks,stages, keep_prob,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 **kwargs):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            data_provider: Class, that have all required data sets
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            total_blocks: `int`, paper value == 3
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            model_type: `str`, 'DenseNet' or 'DenseNet-BC'. Should model use
                bottle neck connections or not.
            dataset: `str`, dataset name
            should_save_logs: `bool`, should logs be saved or not
            should_save_model: `bool`, should model be saved or not
            renew_logs: `bool`, remove previous logs for current model
            reduction: `float`, reduction Theta at transition layer for
                DenseNets with bottleneck layers. See paragraph 'Compression'
                https://arxiv.org/pdf/1608.06993v3.pdf#4
            bc_mode: `bool`, should we use bottleneck layers and features
                reduction or not.
        """
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape # (W,H,C)
        self.n_classes = data_provider.n_classes
        self.depth = depth

        #self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.growth = growth
        self.first_output_features = growth[0] * 2
        self.total_blocks = total_blocks
        self.stages = stages
        self.group_1x1 = kwargs['group_1x1']
        self.group_3x3 = kwargs['group_3x3']
        self.condense_factor = kwargs['condense_factor']
        self.bottleneck = kwargs['bottleneck']
        self.group_lasso_lambda= kwargs['group_lasso_lambda']

        #self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        '''
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        '''
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0

        self._stage = 0
        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logswriter = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
            logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.logs_path, graph=self.sess.graph) # change by ccx, add the graph_def

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}_dataset_{}".format(
            self.model_type, self.growth[0], self.depth, self.dataset_name)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path + 'something')
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix, should_print=True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape) #[None, W, H, C]
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(output, out_features=out_features, kernel_size=kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(output, out_features=inter_features, kernel_size=1, padding='VALID')
            output = self.dropout(output)
        return output


    def learn_group_cov(self, _input, out_features, groups):
        '''add by ccx'''
        assert (_input.get_shape()[-1]) % groups == 0, "group number can not be divided by input channels"
        assert (_input.get_shape()[-1]) % self.condense_factor == 0, "condensation factor can not be divided by input channels"
        assert out_features % groups == 0, "group number can not be divided by output channels"
        with tf.variable_scope("learn_Group_Conv"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            output = self.dropout(output)
            output = self.conv2d_learn_group(output, out_features=out_features, kernel_size=1, groups=groups)
        return output

    def standard_group_cov(self, _input, out_features, groups, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("standard_group_conv"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d_standard_group(output, out_features=out_features, kernel_size=kernel_size, groups=groups)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        lgc_out = self.learn_group_cov(_input, out_features=growth_rate*self.bottleneck, groups=self.group_1x1)
        comp_out = self.standard_group_cov(lgc_out, out_features=growth_rate*self.bottleneck, kernel_size=3, groups=self.group_3x3)
        # concatenate _input with out from composite function

        output = tf.concat(axis=3, values=(_input, comp_out))

        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        # run average pooling changed by ccx
        output = self.avg_pool(_input, k=2)
        return output

    def transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self.weight_variable_xavier(
            [features_total, self.n_classes], name='W')
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits

    def conv2d(self, _input, out_features, kernel_size, strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1]) # get the last dimension, channel
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def conv2d_learn_group(self, _input, out_features, kernel_size,groups, strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1]) # get the last dimension, channel
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='weight')
        print('learn_group_kernel', kernel)
        mask = tf.get_variable('mask', initializer=tf.constant(1.0, shape=kernel.get_shape()), trainable=False)
        output = tf.nn.conv2d(_input, (kernel*mask), strides, padding)
        return output

    def conv2d_standard_group(self, _input, out_features, kernel_size, groups, strides=[1, 1, 1, 1], padding='SAME'):
        print(_input.get_shape())
        in_features = int(_input.get_shape()[-1]) # get the last dimension, channel
        d_in = in_features // groups
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, d_in, out_features],
            name='weight')
        print('standard_group_kernel',kernel)
        d_out = out_features // groups
        for i in range(groups):
            group_output = tf.nn.conv2d(_input[:,:,:,i*d_in:(i+1)*d_in], kernel[:,:,:,i*d_out:(i+1)*d_out], strides, padding)
            if not i == 0:
                output = tf.concat(axis=3, values=(output, group_output))
            else:
                output = group_output
        print(output.get_shape())
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def _build_graph(self):
        #growth_rate = self.growth_rate
        #layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv and 1x1 conv ? to first_output_features --changed by ccx
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(self.images, out_features=self.first_output_features, kernel_size=3)
            #output = self.conv2d(output, out_features=self.first_output_features, kernel_size=1) #???

        # add N DenseBlock
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, self.growth[block], self.stages[block])
                print('after Block_%d' % block, output)
            # last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)

        print('after all block----',output)

        with tf.variable_scope("Transition_to_classes"):
            logits = self.transition_layer_to_classes(output)
        prediction = tf.nn.softmax(logits)

        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels))
        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        lasso_loss = self.lasso_loss()

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(cross_entropy + l2_loss * self.weight_decay + lasso_loss * self.group_lasso_lambda)

        correct_prediction = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def lasso_loss(self):
        loss = [0.0]
        for i in range(self.total_blocks):
            for j in range(self.stages[i]):
                with tf.variable_scope("Block_%d/layer_%d/learn_Group_Conv" % (i,j), reuse=True):
                    weight = tf.get_variable("weight")
                in_channels = int(weight.get_shape()[-2])
                out_channels = int(weight.get_shape()[-1])
                d_out = out_channels // self.group_1x1
                assert weight.get_shape()[0] == 1
                weight = tf.squeeze(weight)
                weight = tf.square(weight)
                tf.reshape(weight, [in_channels, d_out, self.group_1x1])
                weight = tf.sqrt(tf.reduce_sum(weight, axis=1))
                weight = tf.reduce_sum(weight)
                loss = tf.add(loss, weight)
        return loss


    def stage(self, epoch, epochs):
        self._at_stage = True
        for ci in range(self.condense_factor - 1):
            if epoch < (epochs/(2*(self.condense_factor - 1)))*(ci+1):
                stage = ci
                break
        else:
            stage = self.condense_factor - 1
        if not self._stage == stage:
            self._stage = stage
            self._at_stage = False

    def droping(self):
        print("stage%d prunning...." % self._stage)
        for i in range(self.total_blocks):
            for j in range(self.stages[i]):
                print("pruning the Block_%d/layer_%d/learn_Group_Conv" % (i, j))
                with tf.variable_scope("Block_%d/layer_%d/learn_Group_Conv" % (i, j), reuse=True):
                    kernel = tf.get_variable("weight")
                    mask = tf.get_variable("mask")
                in_features = kernel.get_shape()[-2]
                out_features = kernel.get_shape()[-1]
                weight = tf.squeeze(abs(kernel))
                assert weight.get_shape()[0] == in_features
                assert weight.get_shape()[1] == out_features
                delta = in_features // self.condense_factor  # the num need to prune
                d_out = out_features // self.group_1x1  # the num of filters(feature maps) of each group
                weight = weight.reshape(in_features, d_out, self.group_1x1)
                weight = weight.transpose(0,2,1)
                weight = weight.reshape(in_features, out_features)
                for i in range(self.group_1x1):
                    wi = weight[:, i*d_out:(i+1)*d_out]
                    di = np.argsort(wi.sum(1))[(self._stage-1) * delta : self._stage * delta]
                    mask_tmp = self.sess.run(mask)
                    mask_tmp[:, :, di, i::self.group_1x1] = 0
                    self.sess.run(tf.assign(mask, mask_tmp))

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            self.stage(epoch, n_epochs)
            if not self._at_stage:
                self.droping()
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                print("Decrease learning rate, new lr = %f" % learning_rate)

            print("Training...")
            loss, acc = self.train_one_epoch(self.data_provider.train, batch_size, learning_rate)
            if self.should_save_logs:
                self.log_loss_accuracy(loss, acc, epoch, prefix='train')

            if train_params.get('validation_set', False):
                print("Validation...")
                loss, acc = self.test(self.data_provider.validation, batch_size)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, acc, epoch, prefix='valid')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                self.save_model()

        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(seconds=total_training_time)))

    def train_one_epoch(self, data, batch_size, learning_rate):
        #num_examples = data.num_examples
        num_examples = 50
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            images, labels = batch
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }
            fetches = [self.train_step, self.cross_entropy, self.accuracy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy = result
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(
                    loss, accuracy, self.batches_step, prefix='per_batch',
                    should_print=False)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def test(self, data, batch_size):
        #num_examples = data.num_examples
        num_examples = 10
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            feed_dict = {
                self.images: batch[0],
                self.labels: batch[1],
                self.is_training: False,
            }
            fetches = [self.cross_entropy, self.accuracy]
            loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
            total_loss.append(loss)
            total_accuracy.append(accuracy)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy
