from urllib.request import urlretrieve
from os.path import isfile, join
from tqdm import tqdm

FRENCH_EMBEDDINGS_URL = 'https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.fr.vec'
FILENAME = 'embeddings.fr.vec'

def report(progress: tqdm): 
  global progress_percent 
  progress_percent = 0

  def reporter(count: int, block_size: int, total: int):
      global progress_percent 
      current_percent = int(100*(count * block_size) / total)

      if(current_percent is not progress_percent):
        progress.update(current_percent - progress_percent)
        progress_percent = current_percent

  return reporter

def check_embeddings() -> bool:
  return isfile(join('data', FILENAME))

def download_embeddings():
  with tqdm(total=100) as progress_bar:
    urlretrieve(
      FRENCH_EMBEDDINGS_URL,
      join('data', FILENAME),
      report(progress_bar)  
    )
