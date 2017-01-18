import os
from utils import extract_frames, resize_frames

class ObjTrajProposal(object):
  """
  General Object Trajectory Proposal(OTP) Framework
  """
  def __init__(self):
    super(ObjTrajProposal, self).__init__()
    self.data_root = ''
    self.frame_path = os.path.join(self.data_root, 'frames')
    self.optflow_path = os.path.join(self.data_root, 'optflows')
    self.objproposal_path = os.path.join(self.data_root, 'objproposals')
    self.working_size = 240, None
  
  def load_frames(self, vid):
    frames, fps, size = extract_frames(os.path.join(self.frame_path, 
        '{0}.mp4'.format(vid)))
    assert frames[0].shape[1] == size[0] and frames[0].shape[0] == size[1]
    self.size = (frames[0].shape[0], frames[0].shape[1])
    self.frames = resize_frames(frames, self.working_size[0])
    self.working_size[1] = self.frames[0].shape[1]

  def get_optflows(self, vid, oid):
    if self.optflows is None:
      if os.path.exists(os.path.join()):
        pass
      else:
        pass
    return self.optflows[oid]

  def get_objproposals(self, vid, fid):
    pass

  def detecting(self, vid, fid):
    raise NotImplementedError

  def tracking(self, vid, fid):
    raise NotImplementedError

  def merging(self):
    raise NotImplementedError

  def pruning(self):
    raise NotImplementedError

  def generate_otp(self, vid):
    self.load_frames(vid)
    self.optflows = None
    self.objproposals = None
    self.otp = []

    for fid in len(self.frames):
      self.detecting(vid, fid)
      if i > 0:
        self.tracking(vid, fid)
        self.merging()
      self.pruning()

    return self.otp