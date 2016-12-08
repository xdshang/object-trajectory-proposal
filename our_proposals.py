import numpy as np
from scipy import io as sio
import cv2
import os, sys
import glob
import pickle
from collections import defaultdict
from IPython import embed
from track import MovingTrack, StaticTrack
from background import background_motion
from boundingbox import *
from evaluate import *
from utils import *

@profile
def generate_moving_object_trajectory(frames, masks, bfilter, verbose = 0):
  h, w = frames[0].shape[:2]
  tracks = []
  bgdModel = np.empty((1, 65), dtype = np.float64)
  fgdModel = np.empty((1, 65), dtype = np.float64)
  old_mask = np.empty((h, w), dtype = np.uint8)
  mask = np.empty((h, w), dtype = np.uint8)
  for ind in range(len(frames) - 1):
    frame = frames[ind]
    motion = masks[ind][1] > 0
    old_mask[...] = mask & 1
    mask[...] = cv2.GC_PR_BGD
    active_tracks = []
    # predict new locations for existing tracks
    for track in tracks:
      bbox = track.predict(frame)
      valid_bbox = bbox and bfilter(bbox)
      if valid_bbox:
        old_bbox = truncate_bbox(track.rois[-2], h, w)
        local_inds = np.asarray(np.where(old_mask[old_bbox[1]: old_bbox[1] + old_bbox[3], 
                              old_bbox[0]: old_bbox[0] + old_bbox[2]]))
      if track.is_alive(valid_bbox):  
        # mark potential foreground
        global_inds = local_inds + np.asarray([[old_bbox[1] + track.rois[-1][1] - track.rois[-2][1]], 
                             [old_bbox[0] + track.rois[-1][0] - track.rois[-2][0]]], 
                            dtype = np.int64)
        valid_inds = np.all(global_inds >= 0, axis = 0) & (global_inds[0] < h) & (global_inds[1] < w)
        valid_global_inds = global_inds[:, valid_inds]
        mask[valid_global_inds[0], valid_global_inds[1]] = cv2.GC_PR_FGD
        active_tracks.append(track)
    # mark motion clue as highly potential foreground
    mask[motion] = cv2.GC_FGD
    if np.sum(mask & 1) > 10:
      bgdModel[...] = 0
      fgdModel[...] = 0
      mask, _, _ = cv2.grabCut(frame, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
      nc, components = cv2.connectedComponents((mask & 1).astype(np.uint8))
      # merge dection resutls into active tracks or create new tracks
      for label in range(1, nc):
        component = components == label
        overlap_motion = np.any(component & motion)
        bbox = compute_bbox(component)
        if bfilter(bbox):
          max_iou = 0
          for track in active_tracks:
            iou = compute_iou(bbox, track.rois[-1])
            if iou > max_iou:
              max_iou = iou
              max_iou_track = track
          # TODO: appearance matching
          if max_iou > 0.7:
            max_iou_track.update(frame, bbox)
          elif overlap_motion:
            # overlap either with exsiting tracks or the motion clue at current frame
            tracks.append(MovingTrack(ind, frame, bbox, bfilter))
    if verbose and ind % verbose == 0:
      print 'Tracking %dth frame, active tracks %d, total tracks %d' % (ind, len(active_tracks), len(tracks))
  if verbose:
    print 'Simple backward tracking for %d tracks...' % len(tracks)
  for track in tracks:
    track.predict(frames[-1])
    track.start(track.rois[0], frames[track.pstart])
    for ind in range(track.pstart - 1, -1, -1):
      if not track.predict(frames[ind], reverse = True):
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
    for ind, tracks in matches.iteritems():
      heapq.heapify(tracks)
      tracks[0][1].update(tuple(bboxes[ind, :]), score = scores[ind])
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
      print 'Tracking %dth frame, active tracks %d, total tracks %d' % (i, len(active_tracks), len(complete_tracks))
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
  size = frames[0].shape[1], frames[0].shape[0]
  scale = float(size[0]) / orig_size[0]
  print '\tname: %s, fps: %d size: %dx%dx%d' % \
      (vind, fps, len(frames), size[1], size[0])

  intermediate = os.path.join(working_root, 'intermediate/%s.h5' % vind)
  if not os.path.exists(intermediate):
    intermediate = None
  flows, masks = background_motion(frames, intermediate)
  
  bfilter = BBoxFilter(size[0] * size[1] * 0.001, size[0] * size[1] * 0.3, 0.1)
  mtracks = generate_moving_object_trajectory(frames, masks, bfilter, verbose = 20)
  
  bbs = sio.loadmat(glob.glob('%s/edgebox50_proposals/*/%s*' % \
      (working_root, vind))[0])['bbs']
  for i in range(bbs.shape[0]):
    bbs[i][0][:, :2] -= 1
    bbs[i][0][:, :4] *= scale
  stracks = generate_static_object_trajectory(flows, masks, bbs, nreturn, verbose = 20)

  return mtracks, stracks, scale


if __name__ == '__main__':
  working_root = '../'
  nreturn = 2000

  vinds = get_vinds(os.path.join(working_root, 'datalist.txt'), int(sys.argv[1]), int(sys.argv[2]))

  for i, vind in enumerate(vinds):
    print 'Processing %dth video...' % i

    mtracks, stracks, scale = generate_proposals(vind, nreturn, working_root = working_root)
    with open(os.path.join(working_root, 'our_results', '%s.pkl' % vind), 'wb') as fout:
      pickle.dump({'mtracks': mtracks, 'stracks': stracks, 'scale': scale}, fout)

    tracks = mtracks + stracks[:(nreturn - len(mtracks))]
    gt_tracks = get_gt_tracks(os.path.join(working_root, 'annotations/%s.xml' % vind), scale)
    results = evaluate_track(tracks, gt_tracks)
    with open(os.path.join(working_root, 'results', '%s.txt' % vind), 'w') as fout:
      for gt_id, result in results.iteritems():
        print >> fout, 'gt %d matches track %s with score %f' % (gt_id, result[0], result[1])

  embed()