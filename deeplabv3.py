import tensorflow as tf
import numpy as np
import os
import cv2
import utils

batch_size=1
hw=(224,224)
class_num=22

x=tf.placeholder(tf.float32,shape=[1,hw[0],hw[1],3])
y=tf.placeholder(tf.float32)

def bottleneck(inputs,kernel_size=7,kernel_num=16,strides=2,trainable=True):
  output=tf.layers.conv2d(inputs,filters=kernel_num,kernel_size=kernel_size,strides=strides,padding="same",use_bias=True,trainable=trainable,name="conv_7x7")
  output=tf.layers.batch_normalization(output,trainable=trainable)
  output=tf.nn.relu(output)
  output=tf.layers.max_pooling2d(output,pool_size=3,strides=strides)
  shape=tf.shape(inputs)
  return output

def resnet_block(inputs,kernel_size=3,strides=2,kernel_num=64,is_training=True,use_bias=True,block_name="block1"):
  # No bias here since we want a tight and simple structure as possible
  # strides means the first convolution layer's strides
  # 3 layers bottleneck design
  # identical shortcut and zero padding for increasing dimension
  with tf.variable_scope(block_name) as scope:
    shape=tf.shape(inputs)
    # zero padding for increasing dim (A)
    """
    x=tf.concat([inputs,tf.zeros([shape[0],shape[1],shape[2],kernel_num*4-shape[3]])],axis=3)
    x=tf.reshape(x,shape=[shape[0],shape[1],shape[2],kernel_num*4])
    """
    # projection for increasing dim (B)
    """
    to be added
    """
    # projection for all shortcuts (C)
    x=tf.layers.conv2d(inputs,filters=kernel_num*4,strides=strides,kernel_size=kernel_size,padding="same",trainable=is_training,name="projection")
    x=tf.layers.batch_normalization(x,trainable=is_training)
    x=tf.nn.relu(x)
    
    conv1=tf.layers.conv2d(inputs,filters=kernel_num,strides=strides,kernel_size=1,padding="same",use_bias=use_bias,trainable=is_training,name="conv1")
    conv1=tf.layers.batch_normalization(conv1,trainable=is_training)
    conv1=tf.nn.relu(conv1)
    
    conv2=tf.layers.conv2d(conv1,filters=kernel_num,strides=1,kernel_size=kernel_size,padding="same",use_bias=use_bias,trainable=is_training,name="conv2")
    conv2=tf.layers.batch_normalization(conv2,trainable=is_training)
    conv2=tf.nn.relu(conv2)
    
    conv3=tf.layers.conv2d(conv2,filters=kernel_num*4,strides=1,kernel_size=1,padding="same",use_bias=use_bias,trainable=is_training,name="conv3")
    conv3=tf.layers.batch_normalization(conv3,trainable=is_training)
    conv3=tf.nn.relu(conv3)
    
    output=x+conv3
    return output

def resnet_stage(inputs,block_num=3,kernel_size=3,kernel_num=64,is_training=True,use_bias=True,stage_name="stage1"):
  # the only difference between blocks in the same stage is that the first has strides=2 while others' strides=1
  print(stage_name)
  with tf.variable_scope(stage_name) as scope:
    bottle=resnet_block(inputs,kernel_size=kernel_size,strides=2,kernel_num=kernel_num,is_training=is_training,use_bias=use_bias,block_name="block1")
    for i in range(2,block_num+1):
      bottle=resnet_block(bottle,kernel_size=kernel_size,strides=1,kernel_num=kernel_num,is_training=is_training,use_bias=use_bias,block_name="block"+str(i))
    return bottle

def deeplab_additional_graph(inputs,filters=256,strides=16,kernel_size=3,trainable=True):
  # atrous spacial pyramid pooling; exactly the same as what the author(https://github.com/tensorflow/models/tree/master/research/deeplab) did
  # the atrous rates are doubled when strides becomes 8
  with tf.variable_scope("additional_graph") as scope:
    rates=[6,12,18]
    inputs_size = tf.shape(inputs)[1:3]
    conv_1x1=tf.layers.conv2d(inputs,filters=filters,kernel_size=kernel_size,trainable=trainable,padding="same",activation=tf.nn.relu,name="conv_1x1")
    conv_1x1=tf.layers.batch_normalization(conv_1x1,trainable=trainable)
    conv_1x1=tf.nn.relu(conv_1x1)
    
    conv_3x3_1=tf.layers.conv2d(inputs,filters=filters,kernel_size=kernel_size,dilation_rate=rates[0],trainable=trainable,padding="same",activation=tf.nn.relu,name="conv_3x3_1")
    conv_3x3_1=tf.layers.batch_normalization(conv_3x3_1,trainable=trainable)
    conv_3x3_1=tf.nn.relu(conv_3x3_1)
    
    conv_3x3_2=tf.layers.conv2d(inputs,filters=filters,kernel_size=kernel_size,dilation_rate=rates[1],trainable=trainable,padding="same",activation=tf.nn.relu,name="conv_3x3_2")
    conv_3x3_2=tf.layers.batch_normalization(conv_3x3_2,trainable=trainable)
    conv_3x3_2=tf.nn.relu(conv_3x3_2)
    
    conv_3x3_3=tf.layers.conv2d(inputs,filters=filters,kernel_size=kernel_size,dilation_rate=rates[2],trainable=trainable,padding="same",activation=tf.nn.relu,name="conv_3x3_3")
    conv_3x3_3=tf.layers.batch_normalization(conv_3x3_3,trainable=trainable)
    conv_3x3_3=tf.nn.relu(conv_3x3_3)

    # global average pooling
    image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
    # 1x1 convolution with 256 filters( and batch normalization)
    image_level_features = tf.layers.conv2d(image_level_features, filters=filters, kernel_size=1, strides=1, name='conv_1x1_global')
    image_level_features = tf.layers.batch_normalization(image_level_features,trainable=trainable)
    image_level_features = tf.nn.relu(image_level_features)
    
    # bilinearly upsample features
    image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
    outputs=tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
    outputs=tf.layers.conv2d(outputs,filters=filters,kernel_size=1,strides=1,trainable=trainable)
    outputs=tf.layers.batch_normalization(outputs,trainable=trainable)
    outputs=tf.nn.relu(outputs)
    
    return outputs


#{Architecture}
# if we add some train time regularizers, we may need to feed an additional 0/1 to contral train/test phase 

bottle=bottleneck(x,kernel_size=7,kernel_num=16,strides=2)
stage1=resnet_stage(bottle,block_num=3,kernel_size=3,kernel_num=64,is_training=True,use_bias=True,stage_name="stage1")
stage2=resnet_stage(stage1,block_num=4,kernel_size=3,kernel_num=128,is_training=True,use_bias=True,stage_name="stage2")
stage3=resnet_stage(stage2,block_num=6,kernel_size=3,kernel_num=256,is_training=True,use_bias=True,stage_name="stage3")
stage4=resnet_stage(stage3,block_num=3,kernel_size=3,kernel_num=512,is_training=True,use_bias=True,stage_name="stage4")
aspp=deeplab_additional_graph(stage4,filters=256,strides=16,kernel_size=3,trainable=True)
with tf.variable_scope("upsampling") as scope:
  aspp=tf.layers.conv2d(aspp,filters=class_num,kernel_size=1,strides=1,padding="SAME",trainable=True,name="conv_1x1")
  aspp=tf.image.resize_bilinear(aspp,hw)

loss=-tf.reduce_mean(tf.reduce_sum(tf.multiply(y,tf.log(aspp+1e-16)),axis=1))
train_step=tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(loss)

#{End of Architecture}
print("model successfully built!")

#{Prepare dataset}
# pascal VOC 2012
dataset_dir="../VOCdevkit/VOC2012/"
rawimage_dir="JPEGImages/"
label_dir="SegmentationClass/"

# convert label to one hot and convert data types 
# USING GIVEN TRAIN AND VAL FILE NUMBER LIST 
txt_dir=dataset_dir+"ImageSets/Segmentation/"

ftrain=open(txt_dir+"train.txt")
fval=open(txt_dir+"val.txt")

train_nums=ftrain.read().splitlines()
ftrain.close()

val_nums=fval.read().splitlines()
fval.close()

def label_convertion(file_nums):
  # input the label image numbers and return converted one-hot label np.array 
  tool0=utils.toolbox()
  converted_labels=[]
  for label_num in file_nums:
    label=cv2.imread(dataset_dir+label_dir+label_num+".png")
    label=cv2.resize(label,hw)
    label=tool0.convert_label(label,22)
    converted_labels.append(label)
  return converted_labels

ConvertOrNot=True
if ConvertOrNot:
  train_labels=label_convertion(train_nums)
  np.save("./VOC_train_labels.npy",train_labels)
  val_labels=label_convertion(val_nums)
  np.save("./VOC_val_labels.npy",val_labels)
else:
  train_labels=np.load("./VOC_train_labels.npy")
  val_labels=np.load("./VOC_val_labels.npy")
# end of convertion
print(np.shape(train_labels))

# Loading raw images and resize
def load_raw(file_nums):
  # input the raw image file numbers and return the resized one
  arr=[]
  for num in file_nums:
    img=cv2.imread(dataset_dir+rawimage_dir+num+".jpg")
    img=cv2.resize(img,hw)
    arr.append(img)
  return arr

resize_img=True
if resize_img:
  train_imgs=load_raw(train_nums)
  val_imgs=load_raw(val_nums)
  np.save("VOC_train_imgs.npy",train_imgs)
  np.save("VOC_val_imgs.npy",val_imgs)
else:
  train_imgs=np.load("VOC_train_imgs.npy")
  val_imgs=np.load("VOC_val_imgs.npy")
# end of loading

#{End of dataset preparation}


# Judging functions
def judge(prob,label):
  # Given predicted probabilities and segmentation label, return IoU for each class.
  # one image in [H,W,C] format
  pred=np.argmax(prob,axis=-1)
  intersection=np.zeros([np.shape(label)[-1]])
  union=np.sum(label,axis=[0,1])+1e-20
  for i in range(np.shape(prob)[0]):
    for j in range(np.shape(prob)[1]):
      if label[pred[i,j]]==1:
        intersection[pred[i,j]]+=1
      else:
        union[pred[i,j]]+=1
  IoU=intersection/union
  return IoU
# end of judging functions


# Save and restore
restore_iter=0
model_dir="./classic_models/"
saver=tf.train.Saver()
# end of saver


init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

save_iter=20
test_iter=20
max_epoch=10000
seq=np.arange(np.shape(train_imgs)[0])

print("OK")
if restore_iter!=0:
  print("restoring from iter "+str(restore_iter+1))
  saver.restore(sess,model_dir+"classic_iter_"+str(restore_iter+1)+".ckpt")
for e in range(restore_iter,max_epoch):
  np.random.shuffle(seq)
  img_batch=[]
  label_batch=[]
  loss_avg=0
  batch_loss=0
  batch_counter=0
  if (e+1)%test_iter==0:
    print("testing")
    test_prob=sess.run(aspp,feed_dict={x:test_imgs})
    for prob in test_prob:
      IoU=judge(test_prob,test_labels)
    mean_accuracy=np.sum(correct)/np.shape(test_prob)[0]
    print("test accuracy: "+str(mean_accuracy))
  for i in range(np.shape(train_imgs)[0]):
    img_batch.append(train_imgs[seq[i]])
    label_batch.append(train_labels[seq[i]])
    if (i+1)%batch_size==0:
      batch_counter+=1
      sess.run(train_step,feed_dict={x:img_batch,y:label_batch})
      batch_loss+=sess.run(loss,feed_dict={x:img_batch,y:label_batch})
      loss_avg+=batch_loss
      #print("batch_loss:"+str(batch_loss))
      batch_loss=0
      img_batch=[]
      label_batch=[]
  loss_avg=loss_avg/batch_counter
  print("epoch "+str(e+1)+":"+str(loss_avg))
  if (e+1)%save_iter==0:
    saver.save(sess,model_dir+"classic_iter_"+str(e+1)+".ckpt")

