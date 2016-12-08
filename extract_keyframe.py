import cv2
import os, sys
from utils import *

if __name__ == '__main__':
  working_root = '../'

  files = os.listdir(os.path.join(working_root, 'snippets'))

  batch_num = int(sys.argv[1])
  batch_id = int(sys.argv[2])
  batch_size = (len(files) - 1) / batch_num + 1
  start_id = (batch_id - 1) * batch_size
  end_id = min(batch_id * batch_size - 1, len(files))
  print 'Processing from %d to %d...' % (start_id, end_id)

  for i, fname in enumerate(files[start_id: end_id + 1]):
    frames, _, _ = extract_frames(os.path.join(working_root, 'snippets', fname))
    keyframe = frames[len(frames) / 2]
    assert cv2.imwrite(os.path.join(working_root, 'keyframes', fname[:-4] + '.jpg'), 
        keyframe), 'Cannot write %s' % (fname[:-4] + '.jpg',)
    if i % 20 == 0:
      print 'Processing %dth video' % i
