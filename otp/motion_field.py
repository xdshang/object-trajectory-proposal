import subprocess
import os
import struct
import tempfile
import numpy as np
import skimage.io as imgio

_method = dict()

def get_motion_field_method(name):
  if not name in _method:
    if name == 'ldof':
      _method[name] = LDOF()
    elif name == 'deepflow':
      _method[name] = DeepFlow()
    else:
      raise ValueError('Unknown name: {}'.format(name))
  return _method[name].compute_motion_field 


class LDOF(MotionField):
  """
  LDOF: https://lmb.informatik.uni-freiburg.de/resources/software.php
  flo format: http://vision.middlebury.edu/flow/code/flow-code/README.txt
  """
  def __init__(self):
    exec_file = '/home/xdshang/workspace/optflow/ldof/ldof'
    assert os.path.exists(exec_file), 'ldof binary not found.'
    self.cmd = exec_file + ' {} {}'
    self.tempdir = tempfile.TemporaryDirectory(dir = '/dev/shm')

  def _parse_flo_file(self, flo_file):
    with open(flo_file, 'rb') as fin:
      tag, width, height = struct.unpack('<fii', fin.read(12))
      assert tag == 202021.25, 'Bad flo file'
      optflow = np.empty((height, width, 2), dtype = np.float32)
      for i in range(height):
        for j in range(width):
          u, v = struct.unpack('<ff', fin.read(8))
          optflow[i, j, 0] = u
          optflow[i, j, 1] = v
    return optflow

  def compute_motion_field(self, img1, img2):
    imgio.imsave(os.path.join(self.tempdir.name, 'img1.ppm'), img1)
    imgio.imsave(os.path.join(self.tempdir.name, 'img2.ppm'), img2)
    subprocess.run(self.cmd.format('img1.ppm', 'img2.ppm'), 
        cwd = self.tempdir.name, shell = True, check = True)
    optflow = self._parse_flo_file(os.path.join(self.tempdir.name,
        'img1LDOF.flo'))
    return optflow


class DeepFlow(MotionField):
  """
  DeepFlow implemented by OpenCV
  """
  def __init__(self):
    import cv2
    self.model = cv2.optflow.createOptFlow_DeepFlow()

  def compute_motion_field(self, img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    optflow = self.model.calc(gray1, gray2, None)
    return optflow
