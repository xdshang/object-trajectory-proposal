from queue import Queue
import threading
import shlex, subprocess
import progressbar
import logging
import argparse
import os, sys

from dataset import get_dataset

job_queue = Queue()

def _worker():
  while True:
    cmd = job_queue.get()
    logging.info('running command {}'.format(cmd))
    try:
      args = shlex.split(cmd)
      proc = subprocess.Popen(args, stdin = subprocess.PIPE,
          stdout = subprocess.PIPE, stderr = subprocess.PIPE)
      out, err = proc.communicate()
      logging.debug(out.decode())
      if err:
        logging.warning(err.decode())
    except Exception as err:
      logging.error('{}: {}'.format(type(err).__name__, err))
    job_queue.task_done()

def distributed_run(gen, num_workers, log_level, log_path):
  '''
  gen: generator that produce cmd
  num_workers: number of jobs running in parallel
  '''
  log_path = os.path.join(log_path, __file__)
  log_path = os.path.splitext(log_path)[0] + '.log'
  logging.basicConfig(format='%(asctime)s:%(message)s', 
      filename = log_path, level = eval('logging.{}'.format(log_level)))
  logging.info('Command prototype: {}'.format(gen.get_cmd_prototype()))

  bar = progressbar.ProgressBar(max_value = gen.get_total_num())

  for i in range(num_workers):
    t = threading.Thread(target = _worker)
    t.daemon = True
    t.start()

  stop = False
  for i, cmd in enumerate(gen):
    wait = True
    while wait:
      try:
        wait = job_queue.qsize() >= num_workers
      except KeyboardInterrupt:
        print('Stopped.')
        wait = True
        break
    if wait:
      break
    job_queue.put(cmd)
    bar.update(i)

  print('Waiting for remaining jobs...')
  job_queue.join()
  print('Finished.')

class CmdGen():
  '''command generator'''
  def __init__(self, cmd_prototype, dataset_name):
    self.cmd_prototype = cmd_prototype
    self.dname = dataset_name
    self.dataset = get_dataset(dataset_name).get_index()

  def __iter__(self):
    self.i = 0
    return self

  def __next__(self):
    if self.i >= len(self.dataset):
      raise StopIteration
    data = self.dataset[self.i]
    # get cmd
    cmd = self.cmd_prototype.format(self.dname, data)
    cmd = 'qrsh -now no -w n -cwd -noshell /bin/bash -l -c "{}"'.format(cmd)
    self.i += 1
    return cmd

  def get_cmd_prototype(self):
    return self.cmd_prototype

  def get_total_num(self):
    return len(self.dataset)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Distributed running tool')
  parser.add_argument('-c', '--cmd', required = True,
      help = 'Command prototype')
  parser.add_argument('-d', '--dname', default = 'ilsvrc2016-vid',
      choices = ['ilsvrc2016-vid'], help = 'Dataset name')
  parser.add_argument('-p', '--parallel', type = int, default = 5,
      help = 'Number of workers running in parallel')
  parser.add_argument('--log_level', default = 'INFO',
      choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR'], help = 'Log level')
  parser.add_argument('--log_rpath', default = './', help = 'Log root path')
  args = parser.parse_args()

  gen = CmdGen(args.cmd, args.dname)
  distributed_run(gen, args.parallel, args.log_level, args.log_rpath)
