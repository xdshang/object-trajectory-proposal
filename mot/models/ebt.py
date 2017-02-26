import os
import glob
import cv2
import numpy as np
from scipy import io as sio

from ..trajectory import Trajectory
from ..tracker import tracking_by_optflow_v3
from ..background import get_optical_flow
from ..boundingbox import ciou
from ..utils import *

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

  def detecting(self, fid, initial_tracks):
    """
    Detect new bboxes from Edge Boxes
    """
    for i in range(self.bbs[fid][0].shape[0]):
      bbox = self.bbs[fid][0][i, :4]
      score = self.bbs[fid][0][i, 4]
      if self.bbox_filter(bbox):
        initial_tracks.add(Trajectory(fid, bbox, score))

  # @profile
  def associating(self, initial_tracks, active_tracks, act_len = 1):
    """
    Associate legal initial_tracks with active_tracks
    """
    stopped_tracks = set()
    # get tracks with length of act_len to associate with the active tracks
    tracklet = [track for track in initial_tracks if track.length() >= act_len]
    # initialize IoU matrix
    iou = np.zeros((len(active_tracks), len(tracklet)), dtype = np.float32)
    matches = [[-1000., None] for i in range(len(tracklet))]
    # compute IoU in 3D
    if iou.size > 0:
      for i in range(act_len):
        bboxes1 = np.asarray([track.at(i - act_len) for track in active_tracks], 
            dtype = np.float32)
        bboxes2 = np.asarray([track.at(i) for track in tracklet], 
            dtype = np.float32)
        a = ciou(bboxes1, bboxes2)
        iou += a
      # perform matching
      max_inds = np.argmax(iou, axis = 1)
      for i, track in enumerate(active_tracks):
        max_ind = max_inds[i]
        max_iou = iou[i, max_ind]
        if max_iou > 0.2 * act_len:
          # score is not reliable
          score = max_iou
          if score > matches[max_ind][0]:
            matches[max_ind][0] = score
            matches[max_ind][1] = track
            continue
        stopped_tracks.add(track)
    for i, (score, track) in enumerate(matches):
      if track:
        track.update(tracklet[i])
      else:
        active_tracks.add(tracklet[i])

    initial_tracks.difference_update(tracklet)
    active_tracks.difference_update(stopped_tracks)
    return stopped_tracks

  def pruning(self, stopped_tracks, complete_tracks):
    for track in stopped_tracks:
      push_heap(complete_tracks, (track.get_score(), track), 3000)

  def generate(self, nreturn = 3000, verbose = 0):
    complete_tracks = super().generate(verbose = verbose)
    return_tracks = [track for s, track in sorted(complete_tracks, 
        key = lambda x: x[0], reverse = True)[:nreturn]]

    return return_tracks