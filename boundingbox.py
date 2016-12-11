import numpy as np


class BBoxFilter(object):
  def __init__(self, min_area, max_area, min_ratio):
    self.min_area = min_area
    self.max_area = max_area
    self.min_ratio = min_ratio
    
  def __call__(self, bbox):
    assert len(bbox) == 4
    area = bbox[2] * bbox[3]
    if area < self.min_area or area > self.max_area:
      return False
    if float(min(bbox[2], bbox[3])) / max(bbox[2], bbox[3]) < self.min_ratio:
      return False
    return True


def truncate_bbox(bbox, h, w):
  cmin = np.clip(bbox[0], 0, w - 1)
  cmax = np.clip(bbox[0] + bbox[2], 0, w - 1)
  rmin = np.clip(bbox[1], 0, h - 1)
  rmax = np.clip(bbox[1] + bbox[3], 0, h - 1)
  # return int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)
  return cmin, rmin, cmax - cmin, rmax - rmin


def round_bbox(bbox):
  bbox = np.floor(bbox).astype(np.int32)
  return tuple(bbox)


def compute_bbox(bimg):
  rows = np.any(bimg, axis = 1)
  cols = np.any(bimg, axis = 0)
  rmin, rmax = np.where(rows)[0][[0, -1]]
  cmin, cmax = np.where(cols)[0][[0, -1]]
  return cmin, rmin, cmax - cmin, rmax - rmin


def compute_iou(bbox1, bbox2):
  if bbox1 is None or bbox2 is None:
    return None
  cmin = max(bbox1[0], bbox2[0])
  rmin = max(bbox1[1], bbox2[1])
  cmax = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
  rmax = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
  if (cmin < cmax) and (rmin < rmax):
    intersect = float(cmax - cmin) * (rmax - rmin)
    return intersect / (bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - intersect)
  else:
    return 0.

def find_max_iou(bbox, bboxes):
  bbox = np.asarray(bbox)
  bboxes = np.asarray(bboxes)
  minp = np.maximum([bbox[:2]], bboxes[:, :2])
  maxp = np.minimum([bbox[:2] + bbox[2:]], bboxes[:, :2] + bboxes[:, 2:])
  delta = maxp - minp
  intersect_inds = np.where(np.all(delta > 0, axis = 1))[0]
  intersect = np.prod(delta[intersect_inds, :], axis = 1, dtype = np.float32)
  ious = intersect / (bbox[2] * bbox[3] + \
      np.prod(bboxes[intersect_inds, 2:], axis = 1) - intersect)
  if ious.shape[0] == 0:
    return -1, 0.
  else:
    max_ind = np.argmax(ious)
    return intersect_inds[max_ind], ious[max_ind]