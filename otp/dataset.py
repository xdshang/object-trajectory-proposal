from os.path import join as pjoin
import skvideo.io as skvio
from skimage.transform import resize
from skimage import img_as_ubyte
import xml.etree.ElementTree as ET
import random
import warnings
import argparse

from .video import Video
from .trajectory import Trajectory

random.seed(1701)

_datasets = dict()

def get_dataset(name):
  print('Getting dataset {}...'.format(name))
  if not name in _datasets:
    _datasets[name] = Dataset(name)
    if len(_datasets.keys()) > 1:
      print('--WARNING: using multiple datasets')
  return _datasets[name]


class Dataset():

  def __init__(self, name):
    self.name = name
    if self.name == 'ilsvrc2016-vid':
      self.rpath = '/home/xdshang/dataset/ilsvrc2016-vid'
    else:
      raise ValueError('Unknown dataset {}'.format(self.name))
    
    self.index = [line.strip() for line in
        open(pjoin(self.rpath, 'dataset.csv'), 'r')]

  def get_name(self):
    return self.name

  def get_rpath(self):
    return self.rpath

  def get_index(self, batch_size = 1, batch_id = 1):
    vinds = [[vid, None] for vid in self.index]

    if batch_size > 1 and batch_size * 10 < len(vinds):
      print('Deprecated: computing division for clusters...')
      for ele in vinds:
        anno = self.get_annotation(ele[0])
        ele[1] = anno['frame_num']
      vinds = sorted(vinds, key = lambda x: x[1], reverse = True)
      batches = []
      for i in range(batch_size):
        batches.append([[], 0])
      for vind in vinds:
        min_ind = -1
        min_val = 100000
        for i in range(batch_size):
          if batches[i][1] < min_val:
            min_val = batches[i][1]
            min_ind = i
        batches[min_ind][1] += vind[1]
        batches[min_ind][0].append(vind)

      vinds = [vind for vind, _ in batches[batch_id - 1][0]]
      random.shuffle(vinds)
      return vinds
    else:
      batch_num = (len(vinds) - 1) // batch_size + 1
      start_id = (batch_id - 1) * batch_num
      end_id = min(batch_id * batch_num - 1, len(vinds)) 
      print('Processing from %d to %d...' % (start_id, end_id))
      return [vind for vind, _ in vinds[start_id: end_id + 1]]

  def get_video(self, vid, motion_type = None):
    data_path = os.path.join(self.rpath, 'snippets', '{}.mp4'.format(vid))
    motion_path = None if motion_type is None else\
        os.path.join(self.rpath, 'motions', motion_type, '{}.h5'.format(vid))
    return Video(data_path, motion_path)

  def get_annotation(self, vid):
    path = pjoin(self.rpath, 'annotations', '{}.xml'.format(vid))
    anno = dict()
    # parse xml
    tree = ET.parse(path)
    root = tree.getroot()
    frames = root.findall('./annotation')
    anno['frame_num'] = len(frames)
    anno['ground_truth'] = dict()
    for i, frame in enumerate(frames):
      for obj in frame.findall('./object'):
        trackid = int(obj.findtext('./trackid'))
        xmax = int(obj.findtext('./bndbox/xmax'))
        xmin = int(obj.findtext('./bndbox/xmin'))
        ymax = int(obj.findtext('./bndbox/ymax'))
        ymin = int(obj.findtext('./bndbox/ymin'))
        bbox = (np.round(xmin * scale), np.round(ymin * scale), 
            np.round((xmax - xmin) * scale), np.round((ymax - ymin) * scale))
        try:
          track = anno['ground_truth'][trackid]
          for _ in range(i - track.pend):
            track.predict(None)
          track.predict(bbox)
        except KeyError:
          anno['ground_truth'][trackid] = Trajectory(i, bbox)
    return anno


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description = 'Compute motion field')
  parser.add_argument('--method', default = 'LDOF',
      choices = ['LDOF', 'DeepFlow'], help = 'Method to extract motion field')
  parser.add_argument('--dname', choices = ['ilsvrc2016-vid'],
      required = True, help = 'Dataset name')
  parser.add_argument('--vid', required = True, help = 'video index to process')
  args = parser.parse_args()

  dataset = get_dataset(args.dname)

  print('Processing video {}...'.format(args.vid))
  video = dataset.get_video(args.vid, 'ldof')
  motions = video.motions()