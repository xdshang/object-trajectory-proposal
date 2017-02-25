import os

from ..utils import extract_frames, resize_frames
from ..boundingbox import BBoxFilter


class MOT():
  """
  General Framework for Multi-Object Tracking (MOT)
  """
  def __init__(self, vind, working_root, vsize = 240):
    frames, fps, orig_size = extract_frames(os.path.join(working_root, 
        'snippets/' + vind + '.mp4'))
    self.frames = resize_frames(frames, vsize)
    self.fsize = self.frames[0].shape[1], self.frames[0].shape[0]
    self.working_scale = self.fsize[0] / orig_size[0]
    self.bbox_filter = BBoxFilter(self.fsize[0] * self.fsize[1] * 0.001, 
        self.fsize[0] * self.fsize[1] * 0.3, 0.1)
    print('\tname: %s fps: %d size: %dx%dx%d' % \
        (vind, fps, len(self.frames), self.fsize[1], self.fsize[0]))

  def get_working_scale(self):
    return self.working_scale

  def tracking(self, fid, tracks):
    """
    Predict new locations for existing tracks
    Return: a set of tracks to stop
    """
    raise NotImplementedError

  def backward_tracking(self, tracks):
    """
    Backward track stopped tracks in the final stage
    """
    return

  def detecting(self, fid, initial_tracks):
    """
    Detect new bboxes
    """
    raise NotImplementedError

  def associating(self, initial_tracks, active_tracks, act_len = 1):
    """
    Associate legal initial_tracks with active_tracks
    Return: a set of tracks to stop
    """
    raise NotImplementedError

  def pruning(self, stopped_tracks, complete_tracks):
    """
    Move all stopped tracks into the ranking list, compelete_tracks
    """
    raise NotImplementedError

  def generate(self, verbose = 0):
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

    return complete_tracks
