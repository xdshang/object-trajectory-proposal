import numpy as np
import h5py
import pickle
import os, sys
import matplotlib.pyplot as plt
import skimage.io as skio
from background import background_motion
from trajectory import StaticTrack
from boundingbox import *
from utils import *
from IPython import embed

colors = get_colors()

class StaticTrack_v0(StaticTrack):
    
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


def track_bbox_by_optflow(track, flows, masks):
  h, w = flows[0].shape[0], flows[0].shape[1]
  for i in range(len(flows)):
    masked_flow = flows[i] * (masks[i][1] == 0)[:, :, None]
    bbox = track.predict(masked_flow)
    

def draw_results(frames, tracks, output_dir, output_video = True, fps = None):
  for i in range(len(frames)):
    for j in range(len(tracks)):
      frames[i] = draw_tracks(i, frames, [tracks[j]], 
          color = colors[j % len(colors)])
  if output_video:
    create_video(output_dir, frames, fps, 
        (frames[0].shape[1], frames[0].shape[0]), True)
  else:
    create_video_frames(output_dir, frames)


if __name__ == '__main__':
  working_root = '../'
  vind = sys.argv[1]
  if len(sys.argv) > 2:
    result_dir = sys.argv[2]
  else:
    result_dir = None

  # load video frames
  frames, fps, _ = extract_frames(os.path.join(working_root, 
      'snippets', '%s.mp4' % vind))
  frames = resize_frames(frames, 240)
  # load optical flows
  flows, masks = background_motion(frames, os.path.join(working_root, 
      'intermediate', '%s.h5' % vind))

  if result_dir:
    with open(os.path.join(result_dir, '%s.pkl' % vind), 'rb') as fin:
      data = pickle.load(fin)
      if data.has_key('tracks'):
        tracks = data['tracks']
      else:
        tracks = data['mtracks'] + data['stracks']
        moving_part = len(data['mtracks'])
    hit_tracks = []
    with open(os.path.join(result_dir, '%s.txt' % vind), 'r') as fin:
      for line in fin:
        line = line.split()
        if line[0] == 'gt':
          try:
            tind = int(line[4])
            # if not tind < moving_part:
            hit_tracks.append(tracks[tind])
          except ValueError:
            pass
    tracks = hit_tracks
  else:
    # bbox_init = (80.0, 140, 40, 30)
    bbox_init = [(160., 40, 65, 40), (25., 35, 60, 50)]
    tracks = []
    for binit in bbox_init:
      track = StaticTrack(0, binit, 0)
      track_bbox_by_optflow(track, flows, masks)
      tracks.append(track)

  draw_results(frames, tracks, os.path.join(working_root, 'tracks', vind), 
      output_video = False, fps = fps)

  # embed()