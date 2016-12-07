import numpy as np
import xml.etree.ElementTree as ET
from track import Track
from boundingbox import compute_iou

def get_gt_tracks(fname, scale = 1.):
  tree = ET.parse(fname)
  root = tree.getroot()
  annos = root.findall('./annotation')
  gts = dict()
  for i, anno in enumerate(annos):
    for obj in anno.findall('./object'):
      trackid = int(obj.findtext('./trackid'))
      xmax = int(obj.findtext('./bndbox/xmax'))
      xmin = int(obj.findtext('./bndbox/xmin'))
      ymax = int(obj.findtext('./bndbox/ymax'))
      ymin = int(obj.findtext('./bndbox/ymin'))
      bbox = (np.round(xmin * scale), np.round(ymin * scale), 
          np.round((xmax - xmin) * scale), np.round((ymax - ymin) * scale))
      try:
        track = gts[trackid]
        for _ in range(i - track.pend):
          track.predict(None)
        track.predict(bbox)
      except KeyError:
        gts[trackid] = Track(i, bbox)
  return gts


def evaluate_track(tracks, gts, iou_thres = 0.5):
  res = dict([(gt_id, [None, 0]) for gt_id in gts.iterkeys()])
  for track_id, track in enumerate(tracks):
    max_ind = -1
    max_score = 0
    for gt_id, gt_track in gts.iteritems():
      cnt = 0.
      ignore = 0
      pstart = max(gt_track.pstart, track.pstart)
      pend = min(gt_track.pend, track.pend)
      for i in range(pstart, pend):
        iou = compute_iou(track.bbox_at(i), gt_track.bbox_at(i))
        if iou is None:
          ignore += 1
        else:
          cnt += iou > iou_thres
      score = cnt / (gt_track.length() + track.length() - cnt - ignore)
      if score > max_score:
        max_score  = score
        max_ind = gt_id
    if max_ind > -1:
      result = res[max_ind]
      if max_score > result[1]:
        result[0] = track_id
        result[1] = max_score
  return res