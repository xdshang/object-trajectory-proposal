import subprocess
import os
import struct
import argparse
import tempfile
import numpy as np
import skimage.io as imgio
import h5py

from mot.utils import *

class LDOF():
  """docstring for LDOF"""
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

  def compute_optical_flow(self, img1, img2):
    imgio.imsave(os.path.join(self.tempdir.name, 'img1.ppm'), img1)
    imgio.imsave(os.path.join(self.tempdir.name, 'img2.ppm'), img2)
    subprocess.run(self.cmd.format('img1.ppm', 'img2.ppm'), 
        cwd = self.tempdir.name, shell = True, check = True)
    optflow = self._parse_flo_file(os.path.join(self.tempdir.name,
        'img1LDOF.flo'))
    return optflow

  def extract_optical_flow(self, vid):
    optflows = []
    for i in range(len(vid) - 1):
      optflow = self.compute_optical_flow(vid[i], vid[i + 1])
      optflows.append(optflow)
    return optflows

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description = 'Compute LDOF')
  parser.add_argument('-r', '--working_root', default = '../', help = 'working root')
  parser.add_argument('-s', '--saving_root', required = True, help = 'saving root')
  parser.add_argument('--vid', help = 'video ID')
  parser.add_argument('--bsize', type = int, help = 'batch size')
  parser.add_argument('--bid', type = int, help = 'batch id')
  args = parser.parse_args()

  working_root = args.working_root
  saving_root = args.saving_root

  assert os.path.exists(os.path.join(saving_root, 'intermediate')), \
      'Directory for results of ours not found'
  if args.vid is not None:
    vinds = [args.vid]
  else:
    vinds = get_vinds(os.path.join(working_root, 'datalist.txt'), args.bsize, args.bid)
  ldof = LDOF()

  for i, vind in enumerate(vinds):
    frames, fps, orig_size = extract_frames(os.path.join(working_root, 
        'snippets/' + vind + '.mp4'))
    frames = resize_frames(frames, 240)
    optflows = ldof.extract_optical_flow(frames)
    with h5py.File(os.path.join(saving_root, 'intermediate', '{}.h5'.format(vind)), 'w') as fout:
      fout.create_dataset('flows', data = optflows)