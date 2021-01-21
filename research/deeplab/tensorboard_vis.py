import tensorflow as tf
from tensorflow.python.platform import gfile

# 这是从二进制格式的pb文件加载模型
graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
graphdef.ParseFromString(gfile.FastGFile("./datasets/trifo_shoe_79_val_9/init_models/deeplabv3_mnv2_ade20k_train/frozen_inference_graph.pb", "rb").read())
_ = tf.import_graph_def(graphdef, name="")



#这是从文件格式的meta文件加载模型

#_ = tf.train.import_meta_graph("model.ckpt.meta")



summary_write = tf.summary.FileWriter("./datasets/trifo_shoe_79_val_9/init_models/deeplabv3_mnv2_ade20k_train/tensorboard_vis" , graph)



#然后再启动tensorboard：

#tensorboard --logdir ./datasets/trifo_shoe_79_val_9/init_models/deeplabv3_mnv2_ade20k_train/tensorboard_vis

