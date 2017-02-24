import os
import glob
import cv2
import numpy as np
from scipy import io as sio
from track import Track, tracking_by_optflow, tracking_by_optflow_v3
from background import background_motion, get_optical_flow
from utils import *
from boundingbox import *


class ObjTrajProposal():
  """
  General Object Trajectory Proposal(OTP) Framework
  """
  def __init__(self, vind, working_root, vsize = 240):
    frames, fps, orig_size = extract_frames(os.path.join(working_root, 
        'snippets/' + vind + '.mp4'))
    self.frames = resize_frames(frames, vsize)
    self.fsize = self.frames[0].shape[1], self.frames[0].shape[0]
    self.working_scale = self.fsize[0] / orig_size[0]
    self.bbox_filter = BBoxFilter(self.fsize[0] * self.fsize[1] * 0.001, 
        self.fsize[0] * self.fsize[1] * 0.3, 0.1)

    intermediate = os.path.join(working_root, 'intermediate/%s.h5' % vind)
    if not os.path.exists(intermediate):
      intermediate = None
    self.flows, self.masks = background_motion(self.frames, intermediate)
    self.bbs = sio.loadmat(glob.glob('%s/edgebox50_proposals/*/%s*' % \
        (working_root, vind))[0])['bbs']
    for i in range(self.bbs.shape[0]):
      self.bbs[i][0][:, :2] -= 1
      self.bbs[i][0][:, :4] *= self.working_scale
    print('\tname: %s fps: %d size: %dx%dx%d' % \
        (vind, fps, len(self.frames), self.fsize[1], self.fsize[0]))

  def get_working_scale(self):
    return self.working_scale

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
        masked_flow = self.flows[fid - 1] * \
            (self.masks[fid - 1][1] == 0)[:, :, None]
        bbox = tracking_by_optflow(track.tail(), masked_flow)
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
    # if fid > 0:
    #   for label in range(1, self.masks[fid - 1][0] + 1):
    #     bbox = compute_bbox(self.masks[fid - 1][1] == label)
    #     if self.bbox_filter(bbox):
    #       track = Track(fid, bbox, ttype = 'm')
    #       track.tracker = cv2.Tracker_create('KCF')
    #       track.tracker.init(self.frames[fid], bbox)
    #       initial_tracks.add(track)
    # static objects detection
    for i in range(self.bbs[fid][0].shape[0]):
      bbox = self.bbs[fid][0][i, :4]
      score = self.bbs[fid][0][i, 4]
      if self.bbox_filter(bbox):
        initial_tracks.add(Track(fid, bbox, score, 's'))

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
    """
    Proper pipeline: 1)tracking; 2)detecting; 3)associating; 4)pruning
    """
    initial_tracks = set()
    active_tracks = set()
    stopped_tracks = set()
    complete_tracks = []
    for i in range(len(self.frames)):
      stopped_tracks.clear()
      if verbose and i % verbose == 0:
        print('Tracking %dth frame: # initial tracks %d, # active tracks %d' % \
            (i, len(initial_tracks), len(active_tracks)))
      if i > 0:
        stopped_tracks.update(self.tracking(i, initial_tracks))
        stopped_tracks.update(self.tracking(i, active_tracks))
      self.detecting(i, initial_tracks)
      stopped_tracks.update(self.associating(initial_tracks, active_tracks))
      self.pruning(stopped_tracks, complete_tracks)

    stopped_tracks = self.associating(initial_tracks, active_tracks, 1)
    stopped_tracks.update(active_tracks)
    self.backward_tracking(stopped_tracks)
    self.pruning(stopped_tracks, complete_tracks)

    return_tracks = [track for s, track in sorted(complete_tracks, 
        key = lambda x: x[0], reverse = True)[:nreturn]]

    return return_tracks


class ObjTrajProposalV2(ObjTrajProposal):
  def __init__(self, vind, working_root, vsize = 240):
    frames, fps, orig_size = extract_frames(os.path.join(working_root, 
        'snippets/' + vind + '.mp4'))
    self.frames = resize_frames(frames, vsize)
    self.fsize = self.frames[0].shape[1], self.frames[0].shape[0]
    self.working_scale = self.fsize[0] / orig_size[0]
    self.bbox_filter = BBoxFilter(self.fsize[0] * self.fsize[1] * 0.001, 
        self.fsize[0] * self.fsize[1] * 0.3, 0.1)

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

