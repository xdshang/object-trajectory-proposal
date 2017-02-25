import cv2
import numpy as np
from collections import deque

from .boundingbox import truncate_bbox, round_bbox
from .tracker import tracking_by_optflow


class Trajectory(object):

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
    # Ensure the object is picklalbe
    self.tracker = None
    return self.__dict__

  def head(self):
    return self.rois[0]

  def tail(self):
    return self.rois[-1]

  def at(self, i):
    """
    Return the i_th bounding box
    Support fancy indexing
    """
    return self.rois[i]

  def bbox_at(self, p):
    """
    Return the bounding box at frame p
    """
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
    if not bbox is None:
      bbox = tuple(bbox)
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


class MovingTrack(Trajectory):

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


class StaticTrack(Trajectory):

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
