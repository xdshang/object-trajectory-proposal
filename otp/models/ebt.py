import os
import glob
import cv2
import numpy as np
from scipy import io as sio

from ..trajectory import Trajectory
from ..tracker import tracking_by_optflow_v3
from ..background import get_optical_flow
from ..boundingbox import ciou, viou
from ..utils import *

from .otp import OTP


class EBT(OTP):
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
    for i in range(min(self.bbs[fid][0].shape[0], 2000)):
      bbox = self.bbs[fid][0][i, :4]
      score = self.bbs[fid][0][i, 4]
      if self.bbox_filter(bbox):
        initial_tracks.add(Trajectory(fid, bbox, score))

  # @profile
  def associating(self, initial_tracks, active_tracks, act_len = 5):
    """
    Associate legal initial_tracks with active_tracks
    """
    stopped_tracks = set()
    tracks = list(active_tracks)
    # get tracks with length of act_len to associate with the active tracks
    tracklet = [track for track in initial_tracks if track.length() >= act_len]
    # initialize IoU matrix
    iou = np.zeros((len(tracks), len(tracklet)), dtype = np.float32)
    # matches = [[-1000., None] for i in range(len(tracklet))]
    s1 = np.asarray([track.get_score() for track in tracks], 
        dtype = np.float32)
    # s2 = np.asarray([track.get_score() for track in tracklet], 
    #     dtype = np.float32)
    s2 = np.ones((len(tracklet),), dtype = np.float32)
    # compute IoU in 3D
    if iou.size > 0:
      bboxes1 = np.asarray([[track.at(i - act_len) for track in tracks]
          for i in range(act_len)], dtype = np.float32)
      bboxes2 = np.asarray([[track.at(i) for track in tracklet]
          for i in range(act_len)], dtype = np.float32)
      iou = viou(bboxes1, bboxes2)
      # perform matching
      inds = np.argsort(s1)[::-1]
      for ind in inds:
        s = iou[ind] * s2
        m_ind = np.argmax(s)
        if s[m_ind] > 0.6:
          s2[m_ind] = -1
          tracks[ind].update(tracklet[m_ind])
        else:
          stopped_tracks.add(tracks[ind])
      # max_inds = np.argmax(iou, axis = 1)
      # for i, track in enumerate(active_tracks):
      #   max_ind = max_inds[i]
      #   max_iou = iou[i, max_ind]
      #   if max_iou > 0.2 * act_len:
      #     # score is not reliable
      #     score = max_iou
      #     if score > matches[max_ind][0]:
      #       matches[max_ind][0] = score
      #       matches[max_ind][1] = track
      #       continue
      #   stopped_tracks.add(track)
    for i in range(s2.size):
      if s2[i] > 0:
        active_tracks.add(tracklet[i])
    # for i, (score, track) in enumerate(matches):
    #   if track:
    #     track.update(tracklet[i])
    #   else:
    #     active_tracks.add(tracklet[i])

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