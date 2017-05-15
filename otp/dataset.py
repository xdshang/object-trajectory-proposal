from os.path import join as pjoin
import skvideo.io as skvio
from skimage.transform import resize
from skimage import img_as_ubyte
import xml.etree.ElementTree as ET
import random
import warnings

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
      print('Computing division for clusters...')
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

  def get_frames(self, vid):
    path = pjoin(self.rpath, 'snippets', '{}.mp4'.format(vid))
    meta = skvio.ffprobe(path)
    fps = int(meta['video']['@r_frame_rate'].split('/')[0])
    size = (int(meta['video']['@width']), int(meta['video']['@height']))
    assert fps > 0, 'Broken video %s' % path
    frames = self._preprocess(skvio.vread(path))
    return frames

  def _preprocess(self, frames, vsize = 240):
    frame_shape = frames[0].shape
    if frame_shape[0] > vsize:
      size = vsize, int(vsize * frame_shape[1] / frame_shape[0])
    else:
      size = frame_shape
    rsz_frames = []
    for frame in frames:
      frame = resize(frame, size, mode = 'constant')
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = img_as_ubyte(frame)
      rsz_frames.append(frame)
    return rsz_frames

  def get_annotation(self, vid):
    path = pjoin(self.rpath, 'annotations', '{}.xml'.format(vid))
    anno = dict()
    # parse xml
    tree = ET.parse(path)
    root = tree.getroot()
    frames = root.findall('./annotation')
    anno['frame_num'] = len(frames)
    # for i, frame in enumerate(frames):
    #   for obj in frame.findall('./object'):
    #     trackid = int(obj.findtext('./trackid'))
    #     xmax = int(obj.findtext('./bndbox/xmax'))
    #     xmin = int(obj.findtext('./bndbox/xmin'))
    #     ymax = int(obj.findtext('./bndbox/ymax'))
    #     ymin = int(obj.findtext('./bndbox/ymin'))
    #     bbox = (np.round(xmin * scale), np.round(ymin * scale), 
    #         np.round((xmax - xmin) * scale), np.round((ymax - ymin) * scale))
    #     try:
    #       track = gts[trackid]
    #       for _ in range(i - track.pend):
    #         track.predict(None)
    #       track.predict(bbox)
    #     except KeyError:
    #       gts[trackid] = Trajectory(i, bbox)
    return anno


if __name__ == '__main__':
  from IPython import embed
  dataset = get_dataset('ilsvrc2016-vid')
  index = dataset.get_index(50, 3)
  embed()