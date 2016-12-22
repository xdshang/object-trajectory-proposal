import os, sys
import pickle
from utils import *
from evaluate import *

if __name__ == '__main__':
  working_root = '../'
  nreturn = 2000

  vinds = get_vinds(os.path.join(working_root, 'datalist.txt'), int(sys.argv[1]), int(sys.argv[2]))

  for i, vind in enumerate(vinds):
    print('Processing %dth video...' % i)

    with open(os.path.join(working_root, 'our_results', '%s.pkl' % vind), 'rb') as fin:
      data = pickle.load(fin)
      mtracks = data['mtracks']
      stracks = data['stracks']
      scale = data['scale']

    gt_tracks = get_gt_tracks(os.path.join(working_root, 'annotations/%s.xml' % vind), scale)

    mresults = evaluate_track(mtracks, gt_tracks)
    with open(os.path.join(working_root, 'our_moving_results', '%s.txt' % vind), 'w') as fout:
      ss = 0.
      for gt_id, result in mresults.items():
        ss += result[1]
        print('gt %d matches track %s with score %f' % (gt_id, result[0], result[1]), file = fout)
      print('average score %f' % (ss / len(mresults),), file = fout)

    sresults = evaluate_track(stracks[:nreturn], gt_tracks)
    with open(os.path.join(working_root, 'our_static_results', '%s.txt' % vind), 'w') as fout:
      ss = 0.
      for gt_id, result in sresults.items():
        ss += result[1]
        print('gt %d matches track %s with score %f' % (gt_id, result[0], result[1]), file = fout)
      print('average score %f' % (ss / len(sresults),), file = fout)