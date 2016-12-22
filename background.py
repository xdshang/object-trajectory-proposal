import cv2
import numpy as np
import h5py

def extract_optflow(frames):
  flows = []
  flow_model = cv2.optflow.createOptFlow_DeepFlow()
  for i in range(len(frames) - 1):
    gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
    flow = flow_model.calc(gray1, gray2, None)
    flows.append(flow)
  return flows


def filter_components(components, area_min = 0, area_max = np.inf):
  mask = np.zeros_like(components[1], dtype = np.uint8)
  mask_cnt = 0
  for i in range(1, components[0]):
    area = (components[1] == i)
    tot = np.sum(area)
    if area_min < tot < area_max:
      mask_cnt += 1
      mask[area] = mask_cnt
  return mask_cnt, mask


def generate_moving_mask_v2(flows_lst, var_thres = 1, area_min = 100, area_max = np.inf):
  h, w = flows_lst[0].shape[:2]
  points1 = np.mgrid[:h, :w].reshape(2, -1).astype(np.float32).T
  Fs = []
  masks = []
  for flow in flows_lst:
    points2 = points1 + flow.reshape(-1, 2)
    F, inlier = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, var_thres)
    components = cv2.connectedComponents(1 - inlier.reshape(h, w))
    Fs.append(F)
    masks.append(filter_components(components, area_min, area_max))
  return Fs, masks


def generate_moving_mask_v3(frame_lst, flows_lst, var_thres = 1, area_min = 100, area_max = np.inf):
  h, w = flows_lst[0].shape[:2]
  points1 = np.mgrid[:h, :w].reshape(2, -1).astype(np.float32).T[:, (1, 0)]
  bgdModel = np.empty((1, 65), dtype = np.float64)
  fgdModel = np.empty((1, 65), dtype = np.float64)
  mask = np.empty((h, w), dtype = np.uint8)
  Fs = []
  masks = []
  for frame, flow in zip(frame_lst, flows_lst):
    points2 = points1 + flow.reshape(-1, 2)
    F, bg = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, var_thres)
    bg = bg.reshape(h, w).astype(np.bool)
    fg = ~bg
    if np.sum(fg) > 10:
      mask[fg] = cv2.GC_PR_FGD
      mask[bg] = cv2.GC_PR_BGD
      bgdModel[...] = 0
      fgdModel[...] = 0
      mask, _, _ = cv2.grabCut(frame, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
      components = cv2.connectedComponents((mask & 1).astype(np.uint8), ltype = cv2.CV_16U)
      masks.append(filter_components(components, area_min, area_max))
    else:
      masks.append((0, np.zeros_like(fg, dtype = np.uint16)))
    Fs.append(F)
  return Fs, masks


def background_motion(frames, intermediate = None):
  """
  masks: a list of (n, mask), where (mask == 0) is the background, 
        and (0 < mask <= n) is the outliers (moving region).
  """
  if intermediate is None:
    flows = extract_optflow(frames)
  else:
    with h5py.File(intermediate, 'r') as fin:
      flows = list(fin['/flows'][:].astype(np.float32))
  Fs, masks = generate_moving_mask_v2(flows, 1, 10, np.prod(frames[0].shape[:2]) * 0.2)
  # Fs, masks = generate_moving_mask_v3(frames, flows, 1, 
  #     np.prod(frames[0].shape[:2]) * 0.001, 
  #     np.prod(frames[0].shape[:2]) * 0.2)
  return flows, masks