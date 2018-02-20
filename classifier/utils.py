import sys, gzip, re

def get_review_text_from_zip(path):
  g = gzip.open(path, 'r')
  for l in g:
    dicti =  eval(l)
    yield (dicti['reviewText'], dicti['overall'])

def get_review_json_from_file(path):
  for l in open(path, 'r'):
    dicti =  eval(l)
    yield (dicti['reviewText'], dicti['overall'])

def normalize_words(review_text):
  words = list(map(lambda x: re.sub("[^a-zA-Z0-9 ']", '', x.strip().lower()), review_text.split(' ')))
  return list(filter(None, words))
