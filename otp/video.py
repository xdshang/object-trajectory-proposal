import skvideo.io as skvio
from skimage.transform import resize
from skimage import img_as_ubyte
import h5py
import logging

from .motion_field import get_motion_field_method

class Video():
  """docstring for Video"""
  def __init__(self, data_path, motion_path = None):
    self.data_path = data_path
    self.motion_path = motion_path

    self.meta = skvio.ffprobe(self.data_path)
    fps = int(meta['video']['@r_frame_rate'].split('/')[0])
    assert fps > 0, 'Broken video %s' % self.data_path
    self._size = (int(meta['video']['@nb_frames']),
        self.meta['video']['@height'], self.meta['video']['@width'])

  def scale(self):
    return self._size[1] / self.meta['video']['@height']

  def size(self):
    return self._size

  def frames(self):
    if not hasattr(self, '_frames'):
      self._frames = skvio.vread(self.data_path)
      self._size = self._frames.shape[:3]
      height = 240
      if self._size[1] > height:
        self._size = (self._size[0], height,
            height * self._size[2] // self._size[1])
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          frame_size = self._size[1:]
          rsz_frames = []
          for frame in self._frames:
            frame = resize(frame, frame_size, mode = 'constant')
            frame = img_as_ubyte(frame)
            rsz_frames.append(frame)
        self._frames = np.asarray(rsz_frames)
    return self._frames

  def motions(self):
    if not hasattr(self, '_motions'):
      frames = self.frames()
      path = '' if self.motion_path is None else self.motion_path
      # load existing one given the path
      if os.path.exists(path):
        try:
          with h5py.File(path, 'r') as fin:
            self._motions = fin['flows'][:].astype(np.float32)
          assert self._motions.shape[1: 2] == self._size[1: 2]
          return self._motions
        except Exception as err:
          print(err, end = ', ')
          print('recomputing...')
      # compute motions
      if 'mpegflow' in path:
        compute_motion_field = get_motion_field_method('mpegflow')
      elif 'ldof' in path:
        compute_motion_field = get_motion_field_method('ldof')
      elif 'deepflow' in path:
        compute_motion_field = get_motion_field_method('deepflow')
      else:
        logging.warning('Unknown motion field method, mpegflow is used.')
        compute_motion_field = get_motion_field_method('mpegflow')
      self._motions = []
      for i in range(len(frames) - 1):
        motion = compute_motion_field(frames[i], frames[i + 1])
        self._motions.append(motion)
      self._motions = np.asarray(self._motions, dtype = np.float32)
      # save
      if not self.motion_path is None:
        with h5py.File(path, 'w') as fout:
          fout.create_dataset('flows', 
              data = np.asarray(self._motions, dtype = np.float16),
              compression = 'gzip', compression_opts = 9)
    return self._motions
