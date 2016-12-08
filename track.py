import cv2
import numpy as np
from collections import deque
from boundingbox import truncate_bbox

class Track(object):

  def __init__(self, pinit, bbox):
    # TODO: need bfilter for all tracks?
    self.pstart = pinit
    self.pend = pinit + 1
    self.rois = deque([bbox])

  def length(self):
    return self.pend - self.pstart
  
  def bbox_at(self, i):
    return self.rois[i - self.pstart]
  
  def predict(self, bbox, reverse = False):
    if reverse:
      self.rois.appendleft(bbox)
      self.pstart -= 1
    else:
      self.rois.append(bbox)
      self.pend += 1
    return bbox


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
    return self.tracker.init(frame, bbox)
  
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
    curr_bbox = truncate_bbox(curr_bbox, flow.shape[0], flow.shape[1])
    corners = [[curr_bbox[0], curr_bbox[1]], 
           [curr_bbox[0] + curr_bbox[2], curr_bbox[1]], 
           [curr_bbox[0], curr_bbox[1] + curr_bbox[3]], 
           [curr_bbox[0] + curr_bbox[2], curr_bbox[1] + curr_bbox[3]]]
    corners = np.asarray(corners, dtype = np.int32)
    delta = flow[corners[:, 1], corners[:, 0]]
    new_corners = delta + corners
    cmin = (new_corners[0, 0] + new_corners[2, 0]) / 2
    cmax = (new_corners[1, 0] + new_corners[3, 0]) / 2
    rmin = (new_corners[0, 1] + new_corners[1, 1]) / 2
    rmax = (new_corners[2, 1] + new_corners[3, 1]) / 2
    w = cmax - cmin
    h = rmax - rmin
    if w < 0:
      w = 0
      cmin = (cmin + cmax) / 2
    if h < 0:
      h = 0
      rmin = (rmin + rmax) / 2
    bbox = cmin, rmin, w, h
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
