#!/usr/bin/env python3

import time
import numpy as np
import tensorflow as tf
import seq2seq_model
from format import read_pair

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "Context vocabulary size")
tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "tactic vocabulary size.")
tf.app.flags.DEFINE_string("train_data", None, "Training data.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("predict", False,
                            "Set to True for predictions.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def read_data(data_path, max_size=None):
    data_set = [[] for _ in _bucketes]
    with tf.gfile.GFile(source_path, mode="r") as data_file:
        pair = read_pair(data_file)
        counter = 0
        while pair and (not max_size or counter < max_size):
            context, tactic = pair
            counter += 1
            source_ids = [ord(x) for x in source]
            target_ids = [ord(x) for x in target]
            target_ids.append(seq2seq_model.EOS_ID)

            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(source_ids) < source_size and len(target_ids) < target_Size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
            pair = read_pair(data_file)
    return data_set
def create_model(session, forward_only):
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from {}".format(ckpt.model.checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables.initializer())
    return model
def train():
    with tf.Session() as sess:
        model = create_model(sess, False)
        train_set = read_data(FLAGS.train_data)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i+1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step {} learning rate {:.4f} step-time {:.2f} perplexity "
                      "{:.2f}".format(model.global_step.eval(),
                                      model.learning_rate.eval(),
                                      step_time, perplexity))
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                checkpoint_path = os.path.join(os.path.dirname(FLAGS.from_train_data,
                                                               "tactic.ckpt"))

                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                sys.stdout.flush()
        print("Done!")
    pass
def predict():
    pass

def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()
