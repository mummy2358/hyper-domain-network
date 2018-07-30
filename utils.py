from os import listdir
import numpy as np

class toolbox:
  def __init__(self):
    self.class_names=['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
    self.cmap=self.get_colormap(len(self.class_names))
    self.hashmap=np.zeros([256,256,256])
    for i in range(len(self.class_names)):
      self.hashmap[int(self.cmap[i,0]),int(self.cmap[i,1]),int(self.cmap[i,2])]=i

  def get_colormap(self,N):
    def get_bit(bitmap,place):
      return (bitmap&(1<<place)!=0)
    cmap=np.zeros([N,3])
    for i in range(N):
      idx=i-1
      r=0
      g=0
      b=0
      for j in range(8):
        r=r|get_bit(idx,0)<<(7-j)
        g=g|get_bit(idx,1)<<(7-j)
        b=b|get_bit(idx,2)<<(7-j)
        idx=idx>>3
      cmap[i,0]=r
      cmap[i,1]=g
      cmap[i,2]=b
    return cmap

  def n2onehot(self,n,N):
    # convert index n to N-dim one-hot vector
    ans=np.zeros(N)
    ans[n]=1
    return ans

  def convert_label(self,img,N):
    # read in label image and convert to one-hot numpy arrays
    # img: HWC
    sh=np.shape(img)
    label=np.zeros([sh[0],sh[1],N])
    for h in range(sh[0]):
      for w in range(sh[1]):
        label[h,w]=self.n2onehot(int(self.hashmap[int(img[h,w,0]),int(img[h,w,1]),int(img[h,w,2])]),N)
    return label

