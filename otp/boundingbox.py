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
    if min(bbox[2], bbox[3]) / max(bbox[2], bbox[3]) < self.min_ratio:
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
  if bboxes.shape[0] == 0:
    return -1, 0.
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


def ciou(bboxes1, bboxes2):
  """
  Compute IoUs between two sets of bounding boxes
  Input: np.array((n, 4), np.float32), np.array((m, 4), np.float32)
  Output: np.array((n, m), np.float32)
  """
  cmin = np.maximum.outer(bboxes1[:, 0], bboxes2[:, 0])
  cmax = np.minimum.outer(bboxes1[:, 0] + bboxes1[:, 2],
                          bboxes2[:, 0] + bboxes2[:, 2])
  w = cmax - cmin
  del cmax, cmin
  w.clip(min = 0, out = w)

  rmin = np.maximum.outer(bboxes1[:, 1], bboxes2[:, 1])
  rmax = np.minimum.outer(bboxes1[:, 1] + bboxes1[:, 3],
                          bboxes2[:, 1] + bboxes2[:, 3])
  h = rmax - rmin
  del rmax, rmin
  h.clip(min = 0, out = h)

  iou = w
  np.multiply(w, h, out = iou)
  del w, h

  a1 = np.prod(bboxes1[:, 2:], axis = 1)
  a2 = np.prod(bboxes2[:, 2:], axis = 1)
  np.divide(iou, np.add.outer(a1, a2) - iou, out = iou)

  return iou


# @jit('float32[:, :](float32[:, :], float32[:, :])')
# def ciou_v2(bboxes1, bboxes2):
#   """
#   Compute IoUs between two sets of bounding boxes
#   Input: np.array((n, 4), np.float32), np.array((m, 4), np.float32)
#   Output: np.array((n, m), np.float32)
#   """
#   n = bboxes1.shape[0]
#   m = bboxes2.shape[0]
#   iou = np.zeros((n, m), dtype = np.float32)

#   for i in range(n):
#     for j in range(m):
#       minp = np.maximum(bboxes1[i, :2], bboxes2[j, :2])
#       maxp = np.minimum(bboxes1[i, :2] + bboxes1[i, 2:],
#           bboxes2[j, :2] + bboxes2[j, 2:])
#       delta = maxp - minp
#       if delta[0] > 0 and delta[1] > 0:
#         intersect = np.prod(delta)
#         iou[i, j] = intersect / (np.prod(bboxes1[i, 2:]) + \
#             np.prod(bboxes2[j, 2:]) - intersect)

#   return iou

def _intersect(bboxes1, bboxes2):
  """
  bboxes: t x n x 4
  """
  assert bboxes1.shape[0] == bboxes2.shape[0]
  t = bboxes1.shape[0]
  inters = np.zeros((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
  _min = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
  _max = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
  w = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
  h = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
  for i in range(t):
    np.maximum.outer(bboxes1[i, :, 0], bboxes2[i, :, 0], out = _min)
    np.minimum.outer(bboxes1[i, :, 0] + bboxes1[i, :, 2], 
        bboxes2[i, :, 0] + bboxes2[i, :, 2], out = _max)
    np.subtract(_max, _min, out = w)
    w.clip(min = 0, out = w)
    np.maximum.outer(bboxes1[i, :, 1], bboxes2[i, :, 1], out = _min)
    np.minimum.outer(bboxes1[i, :, 1] + bboxes1[i, :, 3], 
        bboxes2[i, :, 1] + bboxes2[i, :, 3], out = _max)
    np.subtract(_max, _min, out = h)
    h.clip(min = 0, out = h)
    np.multiply(w, h, out = w)
    inters += w
  return inters

def _union(bboxes1, bboxes2):
  if id(bboxes1) == id(bboxes2):
    w = bboxes1[:, :, 2]
    h = bboxes1[:, :, 3]
    area = np.sum(w * h, axis = 0)
    unions = np.add.outer(area, area)
  else:
    w = bboxes1[:, :, 2]
    h = bboxes1[:, :, 3]
    area1 = np.sum(w * h, axis = 0)
    w = bboxes2[:, :, 2]
    h = bboxes2[:, :, 3]
    area2 = np.sum(w * h, axis = 0)
    unions = np.add.outer(area1, area2)
  return unions

def viou(bboxes1, bboxes2):
  # bboxes: t x n x 4
  iou = _intersect(bboxes1, bboxes2)
  union = _union(bboxes1, bboxes2)
  np.subtract(union, iou, out = union)
  np.divide(iou, union, out = iou)
  return iou