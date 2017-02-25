import numpy as np
import xml.etree.ElementTree as ET
from mot.trajectory import Trajectory
from mot.boundingbox import compute_iou

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
        gts[trackid] = Trajectory(i, bbox)
  return gts


def evaluate_track(tracks, gts, iou_thres = 0.5):
  res = dict([(gt_id, [None, 0]) for gt_id in gts.keys()])
  for track_id, track in enumerate(tracks):
    max_ind = -1
    max_score = 0
    for gt_id, gt_track in gts.items():
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

def evaluate_track_curve(tracks, gts, iou_thres = 0.5):
  res = dict([(gt_id, [0.]) for gt_id in gts.keys()])
  for track_id, track in enumerate(tracks):
    # max_ind = -1
    # max_score = 0
    for gt_id, gt_track in gts.items():
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
      res[gt_id].append(max(res[gt_id][-1], score))
      # if score > max_score:
      #   max_score  = score
      #   max_ind = gt_id
    # if max_ind > -1:
    #   result = res[max_ind]
    #   if max_score > result[1]:
    #     result[0] = track_id
    #     result[1] = max_score
  return res

if __name__ == '__main__':
  import os, sys
  import argparse
  import pickle
  from scipy import io as sio
  from utils import get_vinds
  from collections import defaultdict

  parser = argparse.ArgumentParser(description = 'Evaluation')
  parser.add_argument('-r', '--working_root', default = '../', help = 'working root')
  parser.add_argument('-s', '--saving_root', required = True, help = 'saving root')
  parser.add_argument('-n', '--nreturn', type = int, default = 1000, help = 'number of returned proposals')
  parser.add_argument('-t', '--tau', type = float, default = 0.5, help = 'threshold for recall')
  parser.add_argument('--bsize', type = int, required = True, help = 'batch size')
  parser.add_argument('--bid', type = int, required = True, help = 'batch id')
  args = parser.parse_args()

  working_root = args.working_root
  saving_root = args.saving_root
  nreturn = args.nreturn
  tau = args.tau

  assert os.path.exists(os.path.join(saving_root)), \
      'Directory for results of ours not found'
  vinds = get_vinds(os.path.join(working_root, 'datalist.txt'), args.bsize, args.bid)

  for i, vind in enumerate(vinds):
    print('Processing %dth video...' % i)

    if os.path.exists(os.path.join(saving_root, '%s.pkl' % vind)):
      print('\tLoading existing tracks for %s ...' % vind)
      with open(os.path.join(saving_root, '%s.pkl' % vind), 'rb') as fin:
        data = pickle.load(fin)
        scale = data['scale']
        if data.has_key('tracks'):
          tracks = data['tracks']
        else:
          tracks = data['mtracks'] + data['stracks']
          # tracks = data['stracks']
    else:
      print('\tTracks for %s not found.' % vind)
      tracks = []

    gt_tracks = get_gt_tracks(os.path.join(working_root, 'annotations/%s.xml' % vind), scale)
    tious = evaluate_track_curve(tracks[:nreturn], gt_tracks)
    if len(tracks) < nreturn:
      print('\tCompensating values to length of %d' % len(tracks))
      for tiou in tious.values():
        while len(tiou) < nreturn + 1:
          tiou.append(tiou[-1])

    tiou_arr = np.asarray(tious.values(), dtype = np.float32)[:, 1:].T
    recall_arr = (tiou_arr > tau).astype(np.float32)
    sio.savemat(os.path.join(saving_root, '%s.mat' % vind), {'tiou': tiou_arr, 'recall': recall_arr})