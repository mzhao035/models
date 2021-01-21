import os
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('eval_logdir', '', 'Evaluation result save path')
flags.DEFINE_string('train_logdir', '', 'Train checkpoints save path')
flags.DEFINE_integer('num_classes', 71, 'Num of labels')
flags.DEFINE_string('dataset', '', 'dataset name')
flags.DEFINE_string('dataset_dir', '', 'tfrecord of dataset save path')
flags.DEFINE_string('eval_crop_size', '', 'eval_crop_size')
flags.DEFINE_string('model_variant', '', 'model_variant')


eval_logdir          = FLAGS.eval_logdir
train_logdir         = FLAGS.train_logdir
num_classes          = FLAGS.num_classes
dataset              = FLAGS.dataset
dataset_dir          = FLAGS.dataset_dir
eval_crop_size       = FLAGS.eval_crop_size
model_variant        = FLAGS.model_variant

checkpoint_dir       = FLAGS.train_logdir + "/checkpoint"

ckfiles = []
with open(checkpoint_dir, 'r') as cf:
    lines = cf.readlines()
    for line in lines:
        textline = line.split(':')
        if textline[0] == 'all_model_checkpoint_paths':
            ckfiles.append(textline[1])

for k in range(len(ckfiles)):
    with open(checkpoint_dir, 'w') as cf:
        lines[0] = 'model_checkpoint_path:' + ckfiles[k]
        cf.writelines(lines)
    exit_status = os.system("bash eval.sh" + " " + model_variant + " " + eval_crop_size + " " + dataset + " " + checkpoint_dir + " " + train_logdir + " " + eval_logdir + " " + dataset_dir )
    if exit_status == 2:
        break

eventfiles = os.listdir(eval_logdir)

steps = []
mious = {}

tag_list = ["eval/miou_1.0_overall"]
for ind in range(1, num_classes):
    tag_list.append("eval/miou_1.0_class_" + str(ind))

for tag in tag_list:
    mious[tag] = []

for filename in eventfiles:
    if filename.startswith('events.out.tfevents.'):
        eventfilepath = eval_logdir + '/' +filename
        print("event filename:", eventfilepath)
        for e in tf.train.summary_iterator(eventfilepath):
            append_step = False
            for v in e.summary.value:
                if v.tag in tag_list:
                    if not np.isnan(v.simple_value):
                        mious[v.tag].append({'step':e.step, "value":v.simple_value})
                        steps.append(e.step)
                
        

filename = eval_logdir + "/evaluation_{0}_{1}".format(np.amin(steps), np.amax(steps))
np.savez(filename, mious=mious, num_classes=num_classes)


