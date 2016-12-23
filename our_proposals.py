import numpy as np
from scipy import io as sio
import cv2
import os, sys
import argparse
import glob
import pickle
from collections import defaultdict
from IPython import embed
from track import MovingTrack, StaticTrack
from background import background_motion
from boundingbox import *
from evaluate import *
from utils import *

DEBUG_MODE = False

@profile
def generate_moving_object_trajectory(frames, masks, bfilter, segment = None, verbose = 0):
  tracks = []
  for ind in range(len(frames) - 1):
    active_tracks = []
    # predict new locations for existing tracks
    for track in tracks:
      bbox = track.predict(frames[ind])
      if track.is_alive(bbox and bfilter(bbox)):
        active_tracks.append(track)
    # merge dection resutls into active tracks or create new tracks
    for label in range(1, masks[ind][0] + 1):
      bbox = compute_bbox(masks[ind][1] == label)
      max_iou = 0
      for track in active_tracks:
        iou = compute_iou(bbox, track.rois[-1])
        if iou > max_iou:
          max_iou = iou
          max_iou_track = track
      # TODO: appearance matching
      if max_iou > 0.5:
        max_iou_track.update(bbox, frames[ind])
      else:
        tracks.append(MovingTrack(ind, bbox, frames[ind]))
    if verbose and ind % verbose == 0:
      print('Tracking %dth frame, active tracks %d, total tracks %d' % (ind, len(active_tracks), len(tracks)))
  if verbose:
    print('Simple backward tracking for %d tracks...' % len(tracks))
  for track in tracks:
    track.predict(frames[-1])
    track.start(track.rois[0], frames[track.pstart])
    for ind in range(track.pstart - 1, -1, -1):
      bbox = track.predict(frames[ind], reverse = True)
      if not track.is_alive(bbox and bfilter(bbox)):
        break
    track.terminate()
  return tracks


@profile
def generate_static_object_trajectory(flows, masks, bbs, nreturn = 1000, verbose = 0):
  active_tracks = set()
  complete_tracks = []
  for i in range(bbs.shape[0]):
    masked_flow = flows[i - 1] * (masks[i - 1][1] == 0)[:, :, None]
    bboxes = np.array(bbs[i][0][:, :4])
    scores = bbs[i][0][:, 4]
    # find matches and complete tracks that do not have any matching new bbox
    matches = defaultdict(list)
    for track in active_tracks:
      # TODO: mask out moving region in optical flow
      bbox = track.predict(masked_flow)
      max_ind, max_iou = find_max_iou(bbox, bboxes)
      if max_iou > 0.5:
        matches[max_ind].append((-track.get_score(), track))
      else:
        track.terminate()
        push_heap(complete_tracks, (track.get_score(), track))
    # complete tracks that match to same single new bbox
    new_inds = set(range(bboxes.shape[0]))
    active_tracks.clear()
    for ind, tracks in matches.items():
      heapq.heapify(tracks)
      # tracks[0][1].update(tuple(bboxes[ind, :]), score = scores[ind])
      tracks[0][1].update(tracks[0][1].rois[-1], score = scores[ind])
      active_tracks.add(tracks[0][1])
      for _, track in tracks[1:]:
        track.terminate()
        push_heap(complete_tracks, (track.get_score(), track))
      # remove new bboxes that continue from existing tracks
      new_inds.remove(ind)
    # start tracking on new bboxes that are not one-to-one matched by existing tracks
    for ind in new_inds:
      active_tracks.add(StaticTrack(i, tuple(bboxes[ind]), scores[ind]))
    if verbose and i % verbose == 0:
      print('Tracking %dth frame, active tracks %d, total tracks %d' % (i, len(active_tracks), len(complete_tracks)))
  for track in active_tracks:
    track.terminate()
    push_heap(complete_tracks, (track.get_score(), track))
  tracks = [track for s, track in sorted(complete_tracks, key = lambda x: x[0], reverse = True)]
  return tracks


@profile
def generate_proposals(vind, nreturn, vsize = 240, working_root = '.'):
  frames, fps, orig_size = extract_frames(os.path.join(working_root, 
      'snippets/' + vind + '.mp4'))
  frames = resize_frames(frames, vsize)
  if DEBUG_MODE:
    create_video_frames('../tracks/%s' % vind, frames)
  size = frames[0].shape[1], frames[0].shape[0]
  scale = size[0] / orig_size[0]
  print('\tname: %s, fps: %d size: %dx%dx%d' % \
      (vind, fps, len(frames), size[1], size[0]))

  intermediate = os.path.join(working_root, 'intermediate/%s.h5' % vind)
  if not os.path.exists(intermediate):
    intermediate = None
  flows, masks = background_motion(frames, intermediate)
  if DEBUG_MODE:
    create_video_frames('../tracks/%s_optflow' % vind, draw_hsv(flows))
    create_video_frames('../tracks/%s_mask' % vind, 
        [(mask > 0).astype(np.uint8) * 255 for n, mask in masks])
  
  bfilter = BBoxFilter(size[0] * size[1] * 0.001, size[0] * size[1] * 0.3, 0.1)
  if DEBUG_MODE:
    segment = []
    mtracks = generate_moving_object_trajectory(frames, masks, bfilter, segment = segment, verbose = 20)
    create_video_frames('../tracks/%s_segment' % vind, segment)
  else:
    mtracks = generate_moving_object_trajectory(frames, masks, bfilter, verbose = 20)

  bbs = sio.loadmat(glob.glob('%s/edgebox50_proposals/*/%s*' % \
      (working_root, vind))[0])['bbs']
  for i in range(bbs.shape[0]):
    bbs[i][0][:, :2] -= 1
    bbs[i][0][:, :4] *= scale
  stracks = generate_static_object_trajectory(flows, masks, bbs, nreturn, verbose = 20)

  return mtracks, stracks, scale


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description = 'Our proposal')
  parser.add_argument('-r', '--working_root', default = '../', help = 'working root')
  parser.add_argument('-s', '--saving_root', required = True, help = 'saving root')
  parser.add_argument('-n', '--nreturn', type = int, default = 2000, help = 'number of returned proposals')
  parser.add_argument('--bsize', type = int, required = True, help = 'batch size')
  parser.add_argument('--bid', type = int, required = True, help = 'batch id')
  args = parser.parse_args()

  working_root = args.working_root
  saving_root = args.saving_root
  nreturn = args.nreturn

  if DEBUG_MODE:
    print('WARNING: visualization mode!')

  assert os.path.exists(os.path.join(saving_root, 'our_results')), \
      'Directory for results of ours not found'
  vinds = get_vinds(os.path.join(working_root, 'datalist.txt'), args.bsize, args.bid)

  for i, vind in enumerate(vinds):
    print('Processing %dth video...' % i)

    if os.path.exists(os.path.join(saving_root, 'our_results', '%s.pkl' % vind)):
      print('\tLoading existing tracks for %s ...' % vind)
      with open(os.path.join(saving_root, 'our_results', '%s.pkl' % vind), 'rb') as fin:
        data = pickle.load(fin)
        mtracks = data['mtracks']
        stracks = data['stracks']
        scale = data['scale']
    else:
      mtracks, stracks, scale = generate_proposals(vind, nreturn, working_root = working_root)
      with open(os.path.join(saving_root, 'our_results', '%s.pkl' % vind), 'wb') as fout:
        pickle.dump({'mtracks': mtracks, 'stracks': stracks, 'scale': scale}, fout)

    assert len(mtracks) + len(stracks) >= nreturn
    tracks = mtracks + stracks[:(nreturn - len(mtracks))]
    gt_tracks = get_gt_tracks(os.path.join(working_root, 'annotations/%s.xml' % vind), scale)
    results = evaluate_track(tracks, gt_tracks)
    with open(os.path.join(saving_root, 'our_results', '%s.txt' % vind), 'w') as fout:
      ss = 0.
      for gt_id, result in results.items():
        ss += result[1]
        print('gt %d matches track %s with score %f' % (gt_id, result[0], result[1]), file = fout)
      print('average score %f' % (ss / len(results),), file = fout)

  # embed()