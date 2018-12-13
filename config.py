import tensorflow as tf

flags = tf.app.flags

# Global params
flags.DEFINE_integer('random_seed', 0, 'random seed')
flags.DEFINE_boolean('evaluate', False, 'evaluating agent')

# Environment specific
flags.DEFINE_string('rom', '', 'Rom file')
flags.DEFINE_float('sticky_prob', 0.25, 'sticky action probability')
flags.DEFINE_integer('mode', 0, 'mode of rom to play')
flags.DEFINE_integer('difficulty', 0, 'difficulty of rom mode')
flags.DEFINE_integer('frame_skip', 5, 'frame skip')
flags.DEFINE_integer('frame_buffer_size', 2, 'frames to perform max over on')
flags.DEFINE_boolean('display_screen', False, 'display screen')

# RL params
flags.DEFINE_float('gamma', 0.99, 'discount rate')
flags.DEFINE_float('eps_initial', 1.0, 'initial epsilon')
flags.DEFINE_float('eps_final', 0.01, 'final epsilon')
flags.DEFINE_integer('eps_anneal_over', 1000000, 'linearly anneal epsilon over')

# Dqn Params
flags.DEFINE_integer('replay_buffer_size', 1000000, 'replay buffer size')
flags.DEFINE_integer('learning_freq', 4, 'learning frequency')
flags.DEFINE_integer('frame_history_size', 4, 'frame history length given as input')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('target_update_every', 40000, 'target network update frequency')
flags.DEFINE_integer('seed_frames', 50000, 'frames to execute random policy to seed replay buffer')

flags.DEFINE_boolean('pad_first_conv_layer', False, 'add padding to first conv layer as per DeepMind')
flags.DEFINE_string('q_scope', 'q', 'TensorFlow scoping for Q network')
flags.DEFINE_string('q_target_scope', 'target', 'TensorFlow scoping for target Q network')

# Optimizer params (RMSProp)
flags.DEFINE_float('learning_rate', 0.00025, 'learning rate')
flags.DEFINE_float('rmsprop_decay', 0.95, 'rmsprop gradient momentum')
flags.DEFINE_float('rmsprop_epsilon', 0.01, 'rmsprop denominator epsilon')

# Experiment params
flags.DEFINE_integer('num_frames', 50000000, 'number of observed frames to run for')
flags.DEFINE_integer('max_episode_count', -1, 'Max number of episodes to run for')
flags.DEFINE_string('restore_dir', None, 'directory to restore weights from, takes the most recent checkpoint unless restore_file is specified')
flags.DEFINE_string('restore_file', None, 'file to restore weights from, the file must be located in restore_dir')
flags.DEFINE_string('save_dir', 'results', 'save directory')
flags.DEFINE_integer('model_chkpt_every', 1000000, 'number of frames to save the model')
flags.DEFINE_string('log_dir', 'logdir', 'log directory')
flags.DEFINE_integer('log_summary_every', 32, 'How frequently to write summary (must be a multiple of learning_freq)')
flags.DEFINE_boolean('disable_progress', False, 'Disable progress bar output')
flags.DEFINE_boolean('log_summary', True, 'logs tensorboard summaries')

# Freezing and Optimization
flags.DEFINE_string('load_scope', 'q', 'Save and restore scope for tf.Saver')
flags.DEFINE_string('optimize_scope', 'q', 'Scope to optimize variables over')

# Regularization
flags.DEFINE_boolean('use_dropout', False, 'Use dropout')
flags.DEFINE_boolean('use_l2', False, 'Use l2 regularization')

flags.DEFINE_float('conv_dropout_rate', 0.05, 'Dropout rate for conv layers')
flags.DEFINE_float('fc_dropout_rate', 0.1, 'Dropout rate for fully connected layers')
flags.DEFINE_float('weight_decay_rate', 0.0001, 'Weight decay rate')

tf.logging.set_verbosity(tf.logging.INFO)
cfg = tf.app.flags.FLAGS
