import argparse
import pickle
from IPython import embed
from utils import *
from evaluate import *
from obj_traj_proposal import ObjTrajProposal, ObjTrajProposalV2


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description = 'OTP proposal')
  parser.add_argument('-r', '--working_root', default = '../', help = 'working root')
  parser.add_argument('-s', '--saving_root', required = True, help = 'saving root')
  parser.add_argument('-n', '--nreturn', type = int, default = 2000, help = 'number of returned proposals')
  parser.add_argument('--vid', help = 'video ID')
  parser.add_argument('--bsize', type = int, help = 'batch size')
  parser.add_argument('--bid', type = int, help = 'batch id')
  args = parser.parse_args()

  working_root = args.working_root
  saving_root = args.saving_root
  nreturn = args.nreturn

  assert os.path.exists(os.path.join(saving_root, 'our_results')), \
      'Directory for results of ours not found'
  if 'vid' in args:
    vinds = [args.vid]
  else:
    vinds = get_vinds(os.path.join(working_root, 'datalist.txt'), args.bsize, args.bid)

  for i, vind in enumerate(vinds):
    print('Processing %dth video...' % i)

    if os.path.exists(os.path.join(saving_root, 'our_results', '%s.pkl' % vind)):
      print('\tLoading existing tracks for %s ...' % vind)
      with open(os.path.join(saving_root, 'our_results', '%s.pkl' % vind), 'rb') as fin:
        data = pickle.load(fin)
        tracks = data['tracks']
        scale = data['scale']
    else:
      otp = ObjTrajProposalV2(vind, working_root = working_root)
      tracks = otp.generate(verbose = 20)
      scale = otp.get_working_scale()
      with open(os.path.join(saving_root, 'our_results', '%s.pkl' % vind), 'wb') as fout:
        pickle.dump({'tracks': tracks, 'scale': scale}, fout)

    tracks = tracks[:nreturn]
    gt_tracks = get_gt_tracks(os.path.join(working_root, 'annotations/%s.xml' % vind), scale)
    results = evaluate_track(tracks, gt_tracks)
    with open(os.path.join(saving_root, 'our_results', '%s.txt' % vind), 'w') as fout:
      ss = 0.
      for gt_id, result in results.items():
        ss += result[1]
        print('gt %d matches track %s with score %f' % (gt_id, result[0], result[1]), file = fout)
      print('average score %f' % (ss / len(results),), file = fout)

  # embed()