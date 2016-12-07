import cv2
import numpy as np
import heapq
import time

def profile(func):
  def wrap(*args, **kwargs):
    started_at = time.time()
    result = func(*args, **kwargs)
    print '\t%s comsumed %.1fs' % (func.__name__, time.time() - started_at)
    return result
  return wrap


def extract_frames(fname):
  cap = cv2.VideoCapture(fname)
  fps = cap.get(cv2.CAP_PROP_FPS)
  size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  assert fps > 0, 'Broken video %s' % vind
  frames = []
  rval, frame = cap.read()
  while rval:
    frames.append(frame)
    rval, frame = cap.read()
  cap.release()
  return frames, fps, size


def resize_frames(frames, vsize = 240):
  frame_shape = frames[0].shape
  if frame_shape[0] > vsize:
    flow_size = vsize, int(round(vsize * frame_shape[1] / frame_shape[0]))
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


def draw_hsv(flow):
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
      cv2.rectangle(img, startp, endp, color, 1)
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
