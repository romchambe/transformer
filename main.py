from utils.embeddings import download_embeddings, check_embeddings


def main(): 
  if(not check_embeddings()): 
    print('Embeddings not available locally. Downloading...')
    download_embeddings()

  print('Embeddings are stored locally, proceeding...')


if __name__ == '__main__': 
  main()