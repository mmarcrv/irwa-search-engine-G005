import os
import zipfile
import pandas as pd

from myapp.search.objects import Document
from typing import List, Dict


def load_corpus(path) -> List[Document]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    #expect a .zip file containing a single JSON file.
    if not path.lower().endswith('.zip'):
        raise ValueError(f"Expected a .zip dataset file, got: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Zip dataset not found: {path}")
    with zipfile.ZipFile(path, 'r') as z:
        # find first .json member in the archive
        json_members = [n for n in z.namelist() if n.lower().endswith('.json')]
        if not json_members:
            raise FileNotFoundError(f"No JSON file found inside zip: {path}")
        json_name = json_members[0]
        with z.open(json_name) as json_file:
            df = pd.read_json(json_file)
            corpus = _build_corpus(df)
            return corpus

def _build_corpus(df: pd.DataFrame) -> Dict[str, Document]:
    """
    Build corpus from dataframe
    :param df:
    :return:
    """
    corpus = {}
    for _, row in df.iterrows():
        doc = Document(**row.to_dict())
        corpus[doc.pid] = doc
    return corpus

