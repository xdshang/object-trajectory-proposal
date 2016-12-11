import numpy as np
import h5py
import os, sys
import matplotlib.pyplot as plt
import skimage.io as skio
from background import background_motion
from track import StaticTrack
from boundingbox import *
from utils import *
from IPython import embed

class StaticTrack_v2(StaticTrack):
    
  def predict(self, flow, reverse = False):
    curr_bbox = self.rois[0] if reverse else self.rois[-1]
    curr_bbox = truncate_bbox(curr_bbox, flow.shape[0], flow.shape[1])
    corners = [[curr_bbox[0], curr_bbox[1]], 
           [curr_bbox[0] + curr_bbox[2], curr_bbox[1]], 
           [curr_bbox[0], curr_bbox[1] + curr_bbox[3]], 
           [curr_bbox[0] + curr_bbox[2], curr_bbox[1] + curr_bbox[3]]]
    corners = np.asarray(corners)
    indices = np.round(corners).astype(np.int32)
    delta = flow[indices[:, 1], indices[:, 0]]
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


def track_bbox_by_optflow(track, frames, flows, masks, color = (0, 0, 255)):
  h, w = frames[0].shape[0], frames[0].shape[1]
  for i in range(1, len(frames)):
    masked_flow = flows[i - 1] * (masks[i - 1][1] == 0)[:, :, None]
    bbox = track.predict(masked_flow)
    bbox = truncate_bbox(bbox, h, w)
    frames[i] = draw_bboxes(frames[i], [bbox], color)


if __name__ == '__main__':
  working_root = '../'
  vind = sys.argv[1]
  # load video frames
  frames, fps, _ = extract_frames(os.path.join(working_root, 
      'snippets', '%s.mp4' % vind))
  frames = resize_frames(frames, 240)
  # load optical flows
  flows, masks = background_motion(frames, os.path.join(working_root, 
      'intermediate', '%s.h5' % vind))

  bbox_init = (190.0, 110, 100, 80)

  track = StaticTrack(0, bbox_init, 0)
  track_bbox_by_optflow(track, frames, flows, masks, color = ((0, 255, 0)))

  track = StaticTrack_v2(0, bbox_init, 0)
  track_bbox_by_optflow(track, frames, flows, masks, color = ((0, 0, 255)))

  create_video(os.path.join(working_root, 'tracks', vind),
      frames, fps, (frames[0].shape[1], frames[0].shape[0]), True)

  # embed()