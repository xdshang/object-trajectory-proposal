import numpy as np
from scipy import io as sio
import os, sys
import glob
import pickle
from IPython import embed
from track import MovingTrack
from boundingbox import *
from evaluate import *
from utils import *

@profile
def generate_proposals(vind, nreturn, baseline, vsize = 240, working_root = '.'):
  frames, fps, orig_size = extract_frames(os.path.join(working_root, 
      'snippets/' + vind + '.mp4'))
  frames = resize_frames(frames, vsize)
  size = frames[0].shape[1], frames[0].shape[0]
  scale = float(size[0]) / orig_size[0]
  print '\tname: %s, fps: %d size: %dx%dx%d' % \
      (vind, fps, len(frames), size[1], size[0])

  bfilter = BBoxFilter(size[0] * size[1] * 0.001, size[0] * size[1], 0.1)
  pinit = len(frames) / 2

  if 'edgebox' in baseline:
    bbs = sio.loadmat(glob.glob('%s/%s_proposals/*/%s*' % \
        (working_root, baseline, vind))[0])['bbs']
    bbs = bbs[pinit][0]
    bbs[:, :2] -= 1
    bbs[:, :4] *= scale
  elif 'mcg' in baseline:
    bbs = sio.loadmat(glob.glob('%s/%s_proposals/%s*' % \
        (working_root, baseline, vind))[0])['new_bbs']
    bbs[:, :2] -= 1
    bbs[:, :4] *= scale * orig_size[1] / 360

  tracks = []
  for i in range(min(bbs.shape[0], nreturn)):
    bbox_init = truncate_bbox(bbs[i, :4], size[1], size[0])
    bbox_init = round_bbox(bbox_init)
    if not bfilter(bbox_init):
      continue
    track = MovingTrack(pinit, bbox_init, frames[pinit])
    for ind in range(pinit + 1, len(frames)):
      bbox = track.predict(frames[ind])
      if not track.is_alive(bbox and bfilter(bbox)):
        break
    track.terminate()
    track.start(bbox_init, frames[pinit])
    for ind in range(pinit - 1, -1, -1):
      bbox = track.predict(frames[ind], reverse = True)
      if not track.is_alive(bbox and bfilter(bbox)):
        break
    track.terminate()
    tracks.append((bbs[i, 4] * track.length(), track))
    if i % 20 == 0:
      print 'Tracking %dth frame, total tracks %d' % (i, len(tracks))

  tracks = [track for s, track in sorted(tracks, key = lambda x: x[0], reverse = True)]
  return tracks, scale


if __name__ == '__main__':
  working_root = '../'
  saving_root = sys.argv[1]
  baseline = sys.argv[2]
  nreturn = 2000

  assert os.path.exists(os.path.join(saving_root, '%s_results' % baseline)), \
      'Directory for results of %s not found' % baseline
  vinds = get_vinds(os.path.join(working_root, 'datalist.txt'), int(sys.argv[3]), int(sys.argv[4]))

  for i, vind in enumerate(vinds):
    print 'Processing %dth video...' % i

    tracks, scale = generate_proposals(vind, nreturn, baseline, working_root = working_root)
    with open(os.path.join(saving_root, '%s_results' % baseline, '%s.pkl' % vind), 'wb') as fout:
      pickle.dump({'tracks': tracks, 'scale': scale}, fout)

    gt_tracks = get_gt_tracks(os.path.join(working_root, 'annotations/%s.xml' % vind), scale)
    results = evaluate_track(tracks, gt_tracks)
    with open(os.path.join(saving_root, '%s_results' % baseline, '%s.txt' % vind), 'w') as fout:
      ss = 0.
      for gt_id, result in results.iteritems():
        ss += result[1]
        print >> fout, 'gt %d matches track %s with score %f' % (gt_id, result[0], result[1])
      print >> fout, 'average score %f' % (ss / len(results),)

  # embed()