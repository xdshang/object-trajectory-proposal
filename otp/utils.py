import cv2
import skvideo.io as skvio
import numpy as np
import heapq
import time
import os
import random

def get_colors():
  return ((0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (255, 0, 128),
    (128, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 0, 128),
    (0, 255, 128))


def profile(func):
  def wrap(*args, **kwargs):
    started_at = time.time()
    result = func(*args, **kwargs)
    print('\t%s comsumed %.4fs' % (func.__name__, time.time() - started_at))
    return result
  return wrap


def extract_frames(fname):
  meta = skvio.ffprobe(fname)
  fps = int(meta['video']['@r_frame_rate'].split('/')[0])
  size = (int(meta['video']['@width']), int(meta['video']['@height']))
  assert fps > 0, 'Broken video %s' % fname
  frames = list(skvio.vread(fname))
  return frames, fps, size


def resize_frames(frames, vsize = 240):
  frame_shape = frames[0].shape
  if frame_shape[0] > vsize:
    flow_size = vsize, int(vsize * frame_shape[1] / frame_shape[0])
  else:
    flow_size = frame_shape
  flow_size = flow_size[1], flow_size[0]
  rsz_frames = []
  for frame in frames:
    rsz_frames.append(cv2.resize(frame, flow_size))
  return rsz_frames


def mask_frame(mask, frame, reverse = False):
  masked_frame = np.array(frame)
  if reverse:
    masked_frame[mask[1] > 0] = 0
  else:
    masked_frame[mask[1] == 0] = 0
  return masked_frame


def create_video(fname, frames, fps, size, isColor):
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(fname + '.avi', fourcc, fps, size, isColor)
  for frame in frames:
    out.write(frame)
  out.release()


def create_video_frames(fname, frames, ext = 'jpg'):
  try:
    os.mkdir(fname)
  except OSError:
    pass
  for i, frame in enumerate(frames):
    cv2.imwrite(os.path.join(fname, '%06d.%s' % (i + 1, ext)), frame)


def draw_hsv(flow):
  if isinstance(flow, list):
    mags = [cv2.cartToPolar(f[...,0], f[...,1])[0] for f in flow]
    min_val = np.min(mags)
    max_val = np.max(mags)
    bgrs = []
    for f in flow:
      mag, ang = cv2.cartToPolar(f[...,0], f[...,1])
      hsv = np.zeros((f.shape[0], f.shape[1], 3), np.uint8)
      hsv[...,0] = ang * 180 / np.pi / 2
      hsv[...,1] = 255
      hsv[...,2] = (mag - min_val) / (max_val - min_val) * 255
      bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
      bgrs.append(bgr)
    return bgrs
  else:
    argnan = np.isnan(flow[..., 0])
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,1] = 255
    min_val = np.nanmin(mag)
    max_val = np.nanmax(mag)
    hsv[...,2] = (mag - min_val) / (max_val - min_val) * 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[argnan, :] = 0
    return bgr


def draw_tracks(ind, frames, tracks, color = (255, 0, 0)):
  img = np.array(frames[ind])
  for track in tracks:
    if track.pstart <= ind < track.pend:
      bbox = track.rois[ind - track.pstart]
      startp = int(bbox[0]), int(bbox[1])
      endp = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
      cv2.rectangle(img, startp, endp, color, 2)
  return img


def draw_bboxes(frame, bboxes, color = (255, 0, 0)):
  img = np.array(frame)
  for bbox in bboxes:
    startp = int(bbox[0]), int(bbox[1])
    endp = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
    cv2.rectangle(img, startp, endp, color, 1)
  return img


def push_heap(heap, item, max_size = 3000):
  if len(heap) < max_size:
    heapq.heappush(heap, item)
  else:
    heapq.heappushpop(heap, item)


def get_vinds(fname, batch_size, batch_id):  
  vinds = []
  with open(fname, 'r') as fin:
    for line in fin:
      line = line.split()
      vinds.append((line[0], int(line[1])))

  if batch_size * 10 < len(vinds):
    print('Computing division for clusters...')
    vinds = sorted(vinds, key = lambda x: (x[1], x[0]), reverse = True)
    batches = []
    for i in range(batch_size):
      batches.append([[], 0])
    for vind in vinds:
      min_ind = -1
      min_val = 100000
      for i in range(batch_size):
        if batches[i][1] < min_val:
          min_val = batches[i][1]
          min_ind = i
      batches[min_ind][1] += vind[1]
      batches[min_ind][0].append(vind)

    vinds = [vind for vind, _ in batches[batch_id - 1][0]]
    random.shuffle(vinds)
    return vinds
  else:
    batch_num = (len(vinds) - 1) // batch_size + 1
    start_id = (batch_id - 1) * batch_num
    end_id = min(batch_id * batch_num - 1, len(vinds)) 
    print('Processing from %d to %d...' % (start_id, end_id))
    return [vind for vind, _ in vinds[start_id: end_id + 1]]


def _onmouse(event, x, y, flags, params):
  global _rectangle, _ix, _iy, _display_img, _img

  if event == cv2.EVENT_LBUTTONDOWN:
    _rectangle = True
    _ix, _iy = x, y
  elif event == cv2.EVENT_MOUSEMOVE:
    if _rectangle == True:
      _display_img = _img.copy()
      cv2.rectangle(_display_img, (_ix, _iy), (x, y), (0, 255, 0), 1)
  elif event == cv2.EVENT_LBUTTONUP:
    _rectangle = False
    _bboxes.append((float(min(_ix, x)), float(min(_iy, y)), 
        float(abs(_ix - x)), float(abs(_iy - y))))
    _display_img = draw_bboxes(_img, _bboxes, (0, 0, 255))


def draw_bboxes_on_image(img):
  global _display_img, _img, _bboxes, _rectangle

  _rectangle = False
  _bboxes = []
  _img = img
  _display_img = img.copy()

  cv2.namedWindow('draw bounding boxes')
  cv2.setMouseCallback('draw bounding boxes', _onmouse)

  while(1):
    cv2.imshow('draw bounding boxes', _display_img)
    k = cv2.waitKey(100)
    # escape key
    if k == 1114111:
      if len(_bboxes) == 0:
        break
      _bboxes.pop()
      _display_img = draw_bboxes(_img, _bboxes, (0, 0, 255))
    # enter key
    elif k == 1048586 or k == 1113997:
      break

  cv2.destroyAllWindows()
  return _bboxes