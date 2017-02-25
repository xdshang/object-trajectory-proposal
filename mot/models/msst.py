import os
import glob
import cv2
import numpy as np
from scipy import io as sio

from ..trajectory import Trajectory
from ..tracker import tracking_by_optflow
from ..background import background_motion
from ..utils import *
from ..boundingbox import *

from .mot import MOT


class MSST(MOT):
  """
  Tracking proposals by moving and static object separation (ICME2017 work)
  """
  def __init__(self, vind, working_root, vsize = 240):
    super().__init__(vind, working_root, vsize)

    intermediate = os.path.join(working_root, 'intermediate/%s.h5' % vind)
    if not os.path.exists(intermediate):
      intermediate = None
    self.flows, self.masks = background_motion(self.frames, intermediate)
    self.bbs = sio.loadmat(glob.glob('%s/edgebox50_proposals/*/%s*' % \
        (working_root, vind))[0])['bbs']
    for i in range(self.bbs.shape[0]):
      self.bbs[i][0][:, :2] -= 1
      self.bbs[i][0][:, :4] *= self.working_scale

  # @profile
  def tracking(self, fid, tracks):
    """
    Predict new locations for existing tracks
    """
    stopped_tracks = set()

    for track in tracks:
      track_type = track.get_type()
      if track_type == 'm':
        try:
          ret, bbox = track.tracker.update(self.frames[fid])
          if ret:
            bbox = truncate_bbox(bbox, self.fsize[1], self.fsize[0])
        except:
          bbox = (0., 0., 0., 0.)
        if self.bbox_filter(bbox):
          track.predict(bbox)
        else:
          stopped_tracks.add(track)
      elif track_type == 's':
        # masked_flow = self.flows[fid - 1] * \
        #     (self.masks[fid - 1][1] == 0)[:, :, None]
        bbox = tracking_by_optflow(track.tail(), self.flows[fid - 1])
        bbox = truncate_bbox(bbox, self.fsize[1], self.fsize[0])
        track.predict(bbox)
      else:
        print('Unknown track type: {}'.format(track_type))

    tracks.difference_update(stopped_tracks)
    return stopped_tracks

  def backward_tracking(self, tracks):
    for track in tracks:
      if track.get_type() == 'm':
        track.tracker = cv2.Tracker_create('KCF')
        fid = track.pstart
        track.tracker.init(self.frames[fid], track.head())
        for i in range(fid - 1, -1, -1):
          try:
            ret, bbox = track.tracker.update(self.frames[i])
            if ret:
              bbox = truncate_bbox(bbox, self.fsize[1], self.fsize[0])
          except:
            bbox = (0., 0., 0., 0.)
          if self.bbox_filter(bbox):
            track.predict(bbox, reverse = True)
          else:
            break

  def detecting(self, fid, initial_tracks):
    """
    Detect new bboxes
    """
    # moving objects detection
    if fid > 0:
      for label in range(1, self.masks[fid - 1][0] + 1):
        bbox = compute_bbox(self.masks[fid - 1][1] == label)
        if self.bbox_filter(bbox):
          track = Trajectory(fid, bbox, ttype = 'm')
          track.tracker = cv2.Tracker_create('KCF')
          track.tracker.init(self.frames[fid], bbox)
          initial_tracks.add(track)
    # static objects detection
    for i in range(self.bbs[fid][0].shape[0]):
      bbox = self.bbs[fid][0][i, :4]
      score = self.bbs[fid][0][i, 4]
      if self.bbox_filter(bbox):
        initial_tracks.add(Trajectory(fid, bbox, score, 's'))

  # @profile
  def associating(self, initial_tracks, active_tracks, act_len = 1):
    """
    Associate legal initial_tracks with active_tracks
    """
    stopped_tracks = set()

    for ttype in ['m', 's']:
      legal_init = [track for track in initial_tracks 
          if track.length() >= act_len and track.get_type() == ttype]
      initial_tracks.difference_update(legal_init)

      bboxes = np.asarray([track.head() for track in legal_init])
      matches = [[-1000., None] for i in range(bboxes.shape[0])]

      stopped_tracks = set()
      for track in active_tracks:
        track_type = track.get_type()
        if track_type == ttype:
          bbox = track.tail()
          max_ind, max_iou = find_max_iou(bbox, bboxes)
          if max_iou > 0.5:
            score = track.get_score()
            if score > matches[max_ind][0]:
              if matches[max_ind][1]:
                pass
              matches[max_ind][0] = score
              matches[max_ind][1] = track
              continue
          if track_type == 's':
            stopped_tracks.add(track)

      # update matches
      for i, (score, track) in enumerate(matches):
        if track:
          track.update(legal_init[i])
        else:
          active_tracks.add(legal_init[i])

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
