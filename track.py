import cv2
import numpy as np
from collections import deque
from boundingbox import truncate_bbox, round_bbox

class Track(object):

  def __init__(self, pinit, bbox, score = 1000., ttype = ''):
    self.pstart = pinit
    self.pend = pinit + 1
    self.rois = deque([bbox])
    self.scores = deque([score])
    self.ttype = ttype
    self.tracker = None

  def __lt__(self, other):
    return False

  def __getstate__(self):
    self.tracker = None
    return self.__dict__

  def head(self):
    return self.rois[0]

  def tail(self):
    return self.rois[-1]

  def bbox_at(self, p):
    return self.rois[p - self.pstart]

  def score_at(self, p):
    return self.scores[p - self.pstart]

  def length(self):
    return self.pend - self.pstart

  def get_type(self):
    return self.ttype

  def get_score(self):
    s = np.median(self.scores)
    return s * self.length()
  
  def predict(self, bbox, reverse = False):
    if reverse:
      self.rois.appendleft(bbox)
      self.scores.appendleft(self.scores[0])
      self.pstart -= 1
    else:
      self.rois.append(bbox)
      self.scores.append(self.scores[-1])
      self.pend += 1
    return bbox

  def update(self, track):
    intersec_pstart = max(self.pstart, track.pstart)
    intersec_pend = min(self.pend, track.pend)

    for i in range(intersec_pstart, intersec_pend):
      self.rois[i - self.pstart] = track.bbox_at(i)
      self.scores[i - self.pstart] = track.score_at(i)

    for i in range(intersec_pstart - 1, track.pstart - 1, -1):
      self.rois.appendleft(track.bbox_at(i))
      self.scores.appendleft(track.score_at(i))

    for i in range(intersec_pend, track.pend):
      self.rois.append(track.bbox_at(i))
      self.scores.append(track.score_at(i))

    self.pstart = min(self.pstart, track.pstart)
    self.pend = max(self.pend, track.pend)
    self.ttype = track.ttype
    self.tracker = track.tracker


class MovingTrack(Track):

  def __init__(self, pinit, bbox, frame, max_invis = 10):
    assert self.start(bbox, frame), \
        'Error: failed to initialize a tracker for %s %s' % (pinit, bbox)
    self.pstart = pinit
    self.pend = pinit + 1
    self.rois = deque([bbox])
    self.max_invis = max_invis
  
  def start(self, bbox, frame):
    # reset the tracker
    self.cnt_invis = 0
    self.tracker = cv2.Tracker_create('KCF')
    ret = self.tracker.init(frame, bbox)
    return ret
  
  def terminate(self):
    self.tracker = None

  def is_alive(self, visible):
    if self.cnt_invis < self.max_invis:
      if visible:
        self.cnt_invis = 0
      else:
        self.cnt_invis += 1
      return True
    else:
      self.terminate()
      return False
      
  def predict(self, frame, reverse = False):
    try:
      ret, bbox = self.tracker.update(frame)
    except:
      ret = False
    if ret:
      if reverse:
        self.rois.appendleft(bbox)
        self.pstart -= 1
      else:
        self.rois.append(bbox)
        self.pend += 1
      bbox = truncate_bbox(bbox, frame.shape[0], frame.shape[1])
      return bbox
    else:
      self.cnt_invis = self.max_invis
      return None
    
  def update(self, bbox, frame, reverse = False):
    assert self.start(bbox, frame), \
        'Error: failed to update a tracker for %s %s' % (self.pend, bbox)
    if reverse:
      self.rois[0] = bbox
    else:
      self.rois[-1] = bbox


class StaticTrack(Track):

  def __init__(self, pinit, bbox, score):
    self.pstart = pinit
    self.pend = pinit + 1
    self.rois = deque([bbox])
    self.scores = deque([score])
  
  def terminate(self):
    pass

  def get_score(self):
    s = np.median(self.scores)
    return s * (self.pend - self.pstart)
    
  def predict(self, flow, reverse = False):
    curr_bbox = self.rois[0] if reverse else self.rois[-1]
    bbox = tracking_by_optflow(curr_bbox, flow)
    if reverse:
      self.rois.appendleft(bbox)
      self.pstart -= 1
    else:
      self.rois.append(bbox)
      self.pend += 1
    return bbox
    
  def update(self, bbox, score, reverse = False):
    if reverse:
      self.rois[0] = bbox
      self.scores[0] = score
    else:
      self.rois[-1] = bbox
      self.scores[-1] = score


def tracking_by_optflow(curr_bbox, flow):
  curr_bbox = truncate_bbox(curr_bbox, flow.shape[0], flow.shape[1])
  rcb = round_bbox(curr_bbox)
  cmin = np.median(flow[rcb[1]: rcb[1] + rcb[3] + 1, rcb[0], 0]) \
      + curr_bbox[0]
  cmax = np.median(flow[rcb[1]: rcb[1] + rcb[3] + 1, rcb[0] + rcb[2], 0]) \
      + curr_bbox[0] + curr_bbox[2]
  rmin = np.median(flow[rcb[1], rcb[0]: rcb[0] + rcb[2] + 1, 1]) \
      + curr_bbox[1]
  rmax = np.median(flow[rcb[1] + rcb[3], rcb[0]: rcb[0] + rcb[2] + 1, 1]) \
      + curr_bbox[1] + curr_bbox[3]
  w = cmax - cmin
  h = rmax - rmin
  if w < 0:
    w = 0
    cmin = (cmin + cmax) / 2
  if h < 0:
    h = 0
    rmin = (rmin + rmax) / 2
  return cmin, rmin, w, h


def tracking_by_optflow_v2(curr_bbox, flow):
  curr_bbox = truncate_bbox(curr_bbox, flow.shape[0], flow.shape[1])
  inner = round_bbox((curr_bbox[0] + curr_bbox[2] * 0.25,
      curr_bbox[1] + curr_bbox[3] * 0.25,
      curr_bbox[2] * 0.5, curr_bbox[3] * 0.5))
  l_of = np.mean(flow[inner[1]: inner[1] + inner[3] + 1, inner[0], 0])
  r_of = np.mean(flow[inner[1]: inner[1] + inner[3] + 1, inner[0] + inner[2], 0])
  t_of = np.mean(flow[inner[1], inner[0]: inner[0] + inner[2] + 1, 1])
  b_of = np.mean(flow[inner[1] + inner[3], inner[0]: inner[0] + inner[2] + 1, 1])
  cmin = l_of - (r_of - l_of) * 0.5 + curr_bbox[0]
  cmax = r_of + (r_of - l_of) * 0.5 + curr_bbox[0] + curr_bbox[2]
  rmin = t_of - (b_of - t_of) * 0.5 + curr_bbox[1]
  rmax = b_of + (b_of - t_of) * 0.5 + curr_bbox[1] + curr_bbox[3]
  w = cmax - cmin
  h = rmax - rmin
  if w < 0:
    w = 0
    cmin = (cmin + cmax) / 2
  if h < 0:
    h = 0
    rmin = (rmin + rmax) / 2
  return cmin, rmin, w, h


if __name__ == '__main__':
  import sys
  from background import background_motion
  from utils import *

  working_root = '../'
  vind = sys.argv[1]
  # load video frames
  frames, fps, _ = extract_frames(os.path.join(working_root, 
      'snippets', '%s.mp4' % vind))
  frames = resize_frames(frames, 240)
  # load optical flows
  flows, masks = background_motion(frames, os.path.join(working_root, 
      'intermediate', '%s.h5' % vind))
  h, w = flows[0].shape[0], flows[0].shape[1]

  bbox_init = [(190.0, 40, 80, 80), (180., 20, 90, 100)]
  # bbox_init = [(200., 100, 65, 40), (100.0, 140, 40, 20)]
  tracks = []
  for binit in bbox_init:
    track = Track(0, binit)
    for i in range(len(flows)):
      # masked_flow = flows[i] * (masks[i][1] == 0)[:, :, None]
      bbox = tracking_by_optflow_v2(track.tail(), flows[i])
      track.predict(bbox)
    tracks.append(track)

  colors = get_colors()
  for i in range(len(frames)):
    for j in range(len(tracks)):
      frames[i] = draw_tracks(i, frames, [tracks[j]], 
          color = colors[j % len(colors)])
  create_video(vind, frames, fps, 
      (frames[0].shape[1], frames[0].shape[0]), True)