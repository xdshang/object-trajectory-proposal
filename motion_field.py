import subprocess
import os
from os.path import join as pjoin
from os.path import exists as pexists
import struct
import argparse
import tempfile
import numpy as np
import skimage.io as imgio
import h5py
from IPython import embed

from mot.utils import *
from dataset import get_dataset

class MotionField():

  def __init__(self, method, dataset):
    self.dataset = dataset
    self.method = method
    self.save_path = pjoin(dataset.get_rpath(), 'motions', method)
    if not pexists(self.save_path):
      os.mkdir(self.save_path)

  def extract_motion_field(self, vid):
    path = pjoin(self.save_path, '{}.h5'.format(vid))

    if pexists(path):
      try:
        with h5py.File(path, 'r') as fin:
          motions = fin['flows'][:].astype(np.float32)
        return motions
      except Exception as err:
        print(err)

    frames = self.dataset.get_frames(vid)
    motions = []
    for i in range(len(frames) - 1):
      optflow = self.compute_motion_field(frames[i], frames[i + 1])
      motions.append(optflow)
    with h5py.File(path, 'w') as fout:
      fout.create_dataset('flows', 
          data = np.asarray(motions, dtype = np.float16),
          compression = 'gzip', compression_opts = 9)
    return motions

  def compute_motion_field(self, img1, img2):
    raise NotImplementedError


class LDOF(MotionField):
  """
  LDOF: https://lmb.informatik.uni-freiburg.de/resources/software.php
  flo format: http://vision.middlebury.edu/flow/code/flow-code/README.txt
  """
  def __init__(self, **kwargs):
    super(LDOF, self).__init__(self.__class__.__name__.lower(), **kwargs)
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


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description = 'Compute motion field')
  parser.add_argument('--method', default = 'LDOF', help = 'method: [LDOF]')
  parser.add_argument('--dname', choices = ['ilsvrc2016-vid'],
      required = True, help = 'Dataset name')
  parser.add_argument('--vid', required = True, help = 'video index to process')
  args = parser.parse_args()

  dataset = get_dataset(args.dname)
  method = eval(args.method)(dataset = dataset)

  print('Processing video {}...'.format(args.vid))
  motions = method.extract_motion_field(args.vid)

  # embed()