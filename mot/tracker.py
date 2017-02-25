import numpy as np

from .boundingbox import truncate_bbox, round_bbox


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


def tracking_by_optflow_v3(bboxes, flow, inner_scale = 0.5):
  if bboxes.shape[0] == 0:
    return bboxes
  # precompute cumulative summation of optical flow
  csflow_col = np.insert(flow[:, :, 0], 0, 0, axis = 0).cumsum(axis = 0)
  csflow_row = np.insert(flow[:, :, 1], 0, 0, axis = 1).cumsum(axis = 1)
  # compute offsets of 4 edges of inner bounding boxes
  l = bboxes[:, 0] + bboxes[:, 2] * (0.5 - inner_scale * 0.5)
  r = bboxes[:, 0] + bboxes[:, 2] * (0.5 + inner_scale * 0.5)
  t = bboxes[:, 1] + bboxes[:, 3] * (0.5 - inner_scale * 0.5)
  b = bboxes[:, 1] + bboxes[:, 3] * (0.5 + inner_scale * 0.5)
  l = np.around(l).astype(np.int32)
  r = np.around(r).astype(np.int32)
  t = np.around(t).astype(np.int32)
  b = np.around(b).astype(np.int32)
  np.clip(l, 0, flow.shape[1], out = l)
  np.clip(r, 0, flow.shape[1], out = r)
  np.clip(t, 0, flow.shape[0], out = t)
  np.clip(b, 0, flow.shape[0], out = b)
  # ensure no divide by zero
  w = np.clip(r - l, 1, flow.shape[1])
  h = np.clip(b - t, 1, flow.shape[0])
  l_of = (csflow_col[b, l] - csflow_col[t, l]) / h
  r_of = (csflow_col[b, r] - csflow_col[t, r]) / h
  t_of = (csflow_row[t, r] - csflow_row[t, l]) / w
  b_of = (csflow_row[b, r] - csflow_row[b, l]) / w
  # linearly interpolate to track original bounding boxes
  bboxes[:, 0] += l_of * 1.5 - r_of * 0.5
  bboxes[:, 1] += t_of * 1.5 - b_of * 0.5
  bboxes[:, 2] += (r_of - l_of) * 2.
  bboxes[:, 3] += (b_of - t_of) * 2.

  return bboxes
