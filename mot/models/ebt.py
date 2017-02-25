import os
import glob
import cv2
import numpy as np
from scipy import io as sio

from ..trajectory import Trajectory
from ..tracker import tracking_by_optflow_v3
from ..background import get_optical_flow
from ..utils import *
from ..boundingbox import *

from .mot import MOT


class EBT(MOT):
  """
  Edge Boxes Tracking
  """
  def __init__(self, vind, working_root, vsize = 240):
    super().__init__(vind, working_root, vsize)

    intermediate = os.path.join(working_root, 'intermediate/%s.h5' % vind)
    if not os.path.exists(intermediate):
      intermediate = None
    self.flows = get_optical_flow(self.frames, intermediate)
    self.bbs = sio.loadmat(glob.glob('%s/edgebox50_proposals/*/%s*' % \
        (working_root, vind))[0])['bbs']
    for i in range(self.bbs.shape[0]):
      self.bbs[i][0][:, :2] -= 1
      self.bbs[i][0][:, :4] *= self.working_scale
    print('\tname: %s fps: %d size: %dx%dx%d' % \
        (vind, fps, len(self.frames), self.fsize[1], self.fsize[0]))

  # @profile
  def tracking(self, fid, tracks, inner_scale = 0.5):
    """
    Predict new locations for existing tracks
    """
    tracks = list(tracks)
    bboxes = np.asarray([track.tail() for track in tracks], dtype = np.float32)
    bboxes = tracking_by_optflow_v3(bboxes, self.flows[fid - 1])
    # update the last bounding boxe for each track
    for i, track in enumerate(tracks):
      # TODO: bbox = truncate_bbox(bboxes[i], self.fsize[1], self.fsize[0])
      track.predict(bboxes[i])

    return set()