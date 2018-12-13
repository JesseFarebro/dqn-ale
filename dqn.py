from functools import partial

import tensorflow as tf

from config import cfg


class DQN:
    def __init__(self, input_height, input_width, num_actions):
        """
        input_height: Downsized input height
        input_width: Downsized input width
        num_actions: Minimal action set length
        """
        self.summary = []
        # Placeholder for current state S
        self.S = tf.placeholder(
            tf.float32,
            shape=[None, input_height, input_width, cfg.frame_history_size],
            name="dqn/inputs",
        )
        # Placeholder of action value
        self.actions = tf.placeholder(tf.uint8, shape=[None], name="dqn/actions")
        # Placeholder for reward value
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="dqn/rewards")
        # Placeholder for next state S'
        self.S_p = tf.placeholder(
            tf.float32,
            shape=[None, input_height, input_width, cfg.frame_history_size],
            name="dqn/target_inputs",
        )
        # Placeholder for terminal state boolean
        self.terminals = tf.placeholder(tf.float32, shape=[None], name="dqn/terminals")

        self.image = tf.placeholder(
            tf.uint8, shape=[None, None, 1], name="dqn/raw_image"
        )

        optimize_scope = (
            "%s/" % cfg.q_scope if cfg.optimize_scope is None else cfg.optimize_scope
        )

        # Build TensorFlow operations
        self.Q = self.build_q_network(self.S, num_actions, cfg.q_scope)
        self.Q_target = self.build_q_network(self.S_p, num_actions, cfg.q_target_scope)

        self.copy_to_target = self.build_copy(cfg.q_scope, cfg.q_target_scope)
        self.action = self.build_action()
        self.train = self.build_train(num_actions, optimize_scope)
        self.process_frame = self.build_process_frame(input_height, input_width)

        tf.logging.info("Built DQN graph")

    def build_q_network(self, X, num_actions, scope):
        """
        Builds architechture for our Q-network.

        Notes: Same arch as DQN Nature paper.

        Parameters:
          num_actions: number of actions possible. Usually used as size of output
          scope: Scope for operations. Please wrap all operations inside variable_scope. See DQN.build_copy

        Returns: Final output layer which can be fed X as input
        """
        with tf.variable_scope(scope):
            conv_input = tf.divide(X, 255.0)  # Normalize inputs
            # DeepMind pads the first convolutional layer with zeros on width and height
            # Making the image now (BATCH_SIZE, 86, 86, FRAME_HISTORY_LEN)
            if cfg.pad_first_conv_layer:
                conv_input = tf.pad(conv_input, [[0, 0], [1, 1], [1, 1], [0, 0]])

            initializer = tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG", uniform=True
            )
            regularizer = tf.contrib.layers.l2_regularizer(
                scale=cfg.weight_decay_rate if cfg.use_l2 else 0.0
            )
            conv2d = partial(
                tf.layers.conv2d,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                bias_initializer=initializer,
                bias_regularizer=regularizer,
            )
            dense = partial(
                tf.layers.dense,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                bias_initializer=initializer,
                bias_regularizer=regularizer,
            )
            with tf.variable_scope("conv"):
                conv1 = conv2d(
                    conv_input,
                    filters=32,
                    kernel_size=8,
                    strides=4,
                    activation=tf.nn.relu,
                )
                if cfg.use_dropout:
                    conv1 = tf.layers.dropout(
                        conv1, rate=cfg.conv_dropout_rate, training=(not cfg.evaluate)
                    )
                conv2 = conv2d(
                    conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu
                )
                if cfg.use_dropout:
                    conv2 = tf.layers.dropout(
                        conv2, rate=cfg.conv_dropout_rate, training=(not cfg.evaluate)
                    )
                conv3 = conv2d(
                    conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu
                )
                if cfg.use_dropout:
                    conv3 = tf.layers.dropout(
                        conv3, rate=cfg.conv_dropout_rate, training=(not cfg.evaluate)
                    )
                conv_output = tf.contrib.layers.flatten(conv3)
            with tf.variable_scope("action_values"):
                fully_connected = dense(conv_output, units=512, activation=tf.nn.relu)
                if cfg.use_dropout:
                    fully_connected = tf.layers.dropout(
                        fully_connected,
                        rate=cfg.fc_dropout_rate,
                        training=(not cfg.evaluate),
                    )
                actions = dense(fully_connected, units=num_actions, activation=None)
            return actions

    def build_action(self):
        """
        Builds greedy action op.

        This will return the best possible action for state S.

        Returns: Operation which can be fed self.S
        """
        assertions = [
            tf.assert_equal(tf.shape(self.S)[0], tf.constant(1, dtype=tf.int32))
        ]
        with tf.control_dependencies(assertions):
            deterministic_action = tf.argmax(self.Q, axis=1)
            return tf.squeeze(deterministic_action)

    def build_train(self, num_actions, optimize_scope):
        """
        Builds training operation.

        Parameters:
          num_actions: Number of actions which can be taken in the environment.

        Returns:
          Training operation which should be fed:
            self.S - Input image
            self.S_p - Target input image
            self.actions - Actions taken in batch
            self.rewards - Rewards received in batch
            self.terminals - Whether state was terminal or not
        """
        q_vals = self.Q
        q_target_vals = self.Q_target

        # Compute Q(s, a)
        q_selected = tf.reduce_sum(
            q_vals * tf.one_hot(self.actions, num_actions), axis=1
        )
        self.summary.append(
            tf.contrib.summary.scalar("train/average_q", tf.reduce_mean(q_selected))
        )

        # Compute r + \gamma max_a(Q(s', a))
        # Q(s', a) = 0 for terminal s
        q_target_selected = tf.stop_gradient(
            self.rewards
            + cfg.gamma * (1.0 - self.terminals) * tf.reduce_max(q_target_vals, axis=1)
        )

        # TD Errors: r + \gamma * max_a(Q(s', a)) - Q(s, a)
        td_errors = q_target_selected - q_selected
        td_errors = tf.verify_tensor_all_finite(
            td_errors, "TD error tensor is NOT finite"
        )
        self.summary.append(
            tf.contrib.summary.scalar("train/td_error", tf.reduce_mean(td_errors))
        )

        # Compute huber loss of TD errors (clipping beyond [-1, 1])
        delta = 1.0
        errors = tf.where(
            tf.abs(td_errors) < delta,
            tf.square(td_errors) * 0.5,
            delta * (tf.abs(td_errors) - 0.5 * delta),
        )
        errors = tf.verify_tensor_all_finite(errors, "DQN training loss is NOT finite")

        loss = tf.reduce_sum(errors)
        loss += tf.reduce_sum(tf.losses.get_regularization_losses())
        self.summary.append(tf.contrib.summary.scalar("train/total_loss", loss))

        optimize_vars = []
        for scope in optimize_scope.split(","):
            optimize_vars.extend(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            )
        tf.logging.info("Optimization Scope")
        tf.logging.info(optimize_vars)

        return tf.train.RMSPropOptimizer(
            cfg.learning_rate,
            decay=cfg.rmsprop_decay,
            epsilon=cfg.rmsprop_epsilon,
            centered=True,
        ).minimize(loss, var_list=optimize_vars)

    def build_copy(self, q_scope, q_target_scope):
        """
        Builds copy operation to copy all trainable variables from
        our online network to our target network.
        """
        q_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope
        )
        target_q_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_target_scope
        )

        copy_ops = [
            target_var.assign(q_var) for target_var, q_var in zip(target_q_vars, q_vars)
        ]

        return tf.group(*copy_ops)

    def build_process_frame(self, input_height, input_width):
        """
        Builds frame processing operation.

        Crops the image to (input_height, input_width) and returns image as uint8
        """
        cropped = tf.image.resize_images(
            self.image, (input_height, input_width), tf.image.ResizeMethod.AREA
        )
        return tf.cast(cropped, tf.uint8)
