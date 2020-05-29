import sys
import os
import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer
import math
import json
import re
from argparse import ArgumentParser
from typing import Dict, List
from xml.etree import cElementTree
import numpy as np
import time


def main(input1, input2, input3):
    try:
        #Initializing lists and dictionaries so we can store in the values
        title_dict = {}
        inverted_dict = {}
        stop_words = []
        tree = ET.parse(input2)
        root = tree.getroot()
        ns = 'http://www.mediawiki.org/xml/export-0.6/'
        # Opening up and putting stop words into a list because we want to remove these regardless of collection
        with open(input1, 'r') as stopwords:
            for i in stopwords:
                stop_words.append(i.strip('\n'))
        # If statement so we can choose between which collection

        if 'mediawiki' in root.tag:
            # Running on Pixar Collection
            for page in root.iter('{%s}page' % ns):
                # Parsing and fixing the respective text in each page
                idx = page.find('{%s}id' % ns).text
                title = page.find('{%s}title' % ns).text
                text = (page.find('{%s}revision' % ns).find('{%s}text' % ns)).text
                # Adding the unique ID and Titles of the pages to the a Title Dictionary
                # At this point we are starting the process of tokenizing the titles and text
                # Splitting the strings of the title and text by any non-alphanumeric character
                concat_text = re.split(r'[^a-z0-9]', (('{}\n{}'.format(title, text)).lower()))
                # Removing all empty values from the list
                concat_text = list(filter(None, concat_text))
                # Filtering out stop words (removing them if they are in the list)
                concat_text = [word for word in concat_text if word not in stop_words]
                # Running all words in the list to a Porter Stemmer so that it reduces to the basic form
                stems = [PorterStemmer().stem(word) for word in concat_text]
                # Removing the duplicate stems in the list
                unique_stems = list(set(stems))
                # I'm making an inverted index, and updating the entry(ID of the page) as we find the same key repeated
                # Inside this nested dictionary, is the position of the stem
                # In this retrospect, we can follow down a tree to see 'stem', 'title_index', 'position of the stem within the page'

                for j in unique_stems:
                    if j in inverted_dict:
                        inverted_dict[j][idx] = [i for i, v in enumerate(stems) if v == j]
                    else:
                        inverted_dict[j] = {}
                        inverted_dict[j][idx] = [i for i, v in enumerate(stems) if v == j]
                title_dict[idx] = title


            for key in inverted_dict:
                doc_frequency = len(inverted_dict[key])
                idf = math.log10(len(title_dict) / doc_frequency)
                inverted_dict[key]['idf'] = idf


            # Writing down the json files to be exported to later put into query
            index_path = os.path.join(input3, 'index.json')
            with open(index_path, 'w') as inverted_index:
                json.dump(inverted_dict, inverted_index)
            title_path = os.path.join(input3, 'titles.json')
            with open(title_path, 'w') as title_index:
                json.dump(title_dict, title_index)



        # Same process but for the small.xml file
        else:
            # Running on small xml
            for page in root.iter('page'):
                idx = page.find('id').text
                title = page.find('title').text
                text = page.find('text').text

            #concat title and text
                concat_text = re.split(r'[^a-z0-9]', (('{}\n{}'.format(title, text)).lower()))
                concat_text = list(filter(None, concat_text))
                concat_text = [word for word in concat_text if word not in stop_words]
                stems = [PorterStemmer().stem(word) for word in concat_text]
                unique_stems = list(set(stems))
                tf = 0
                for j in unique_stems:
                    if j in inverted_dict:
                        inverted_dict[j][idx] = [i for i, v in enumerate(stems) if v == j]
                    else:
                        inverted_dict[j] = {}
                        inverted_dict[j][idx] = [i for i, v in enumerate(stems) if v == j]
                    tf += stems.count(j)
                title_dict[idx] = title

            for key in inverted_dict:
                doc_frequency = len(inverted_dict[key])
                idf = math.log10(len(title_dict)/doc_frequency)
                inverted_dict[key]['idf'] = idf

            index_path = os.path.join(input3, 'index.json')
            with open(index_path, 'w') as inverted_index:
                json.dump(inverted_dict, inverted_index)
            title_path = os.path.join(input3, 'titles.json')
            with open(title_path, 'w') as title_index:
                json.dump(title_dict, title_index)
    except:
        print('ERROR: Bad XML File encountered. Exiting')


def parse(collection_path: str) -> Dict[int, List[int]]:
    """Parses the collection file and returns a dictionary mapping
    documents to the documents that they link to.

    The dictionary keys and values are int document ids.

    Note: We recommend that you don't change this code.
    """
    root = cElementTree.parse(collection_path).getroot()
    match = re.match(r'{.*}', root.tag)
    namespace = match.group() if match else ''

    doc_ids = {}
    outlink_titles = {}
    for page in root.iter(namespace + 'page'):
        id_ = int(page.find(namespace + 'id').text)
        title = page.find(namespace + 'title').text
        assert id_ is not None and title is not None
        # Note this doesn't work on the small index, we aren't using
        # the small index anymore in the course
        text = page.find(namespace + 'revision').find(namespace + 'text').text
        if text is None:
            links = []
        else:
            links = extract_links(text)

        doc_ids[title] = id_
        outlink_titles[id_] = links

    outlink_ids = {}
    for id_, titles in outlink_titles.items():
        outlink_ids[id_] = [doc_ids[title]
                            for title in titles
                            if title in doc_ids]

    for id_ in get_isolates(outlink_ids):
        outlink_ids.pop(id_)

    return outlink_ids


def extract_links(text: str) -> List[str]:
    """Returns the links in the body text. The links are
    title strings.

    Note: We recommend that you don't change this code.
    """
    return re.findall(r'\[\[([^\]|#]+)', text)


def compute_length(x):
    squared_x = np.dot(x, x)
    length = np.sqrt(squared_x)
    return (length)


def get_isolates(outlinks: Dict[int, List[int]]) -> List[int]:
    """Returns all doc ids which have no inbound nor
    outbound links.

    Note: We recommend that you don't change this code.
    """
    connected_ids = set()
    for id_, linked_ids in outlinks.items():
        if linked_ids:
            connected_ids.add(id_)
            connected_ids.update(linked_ids)

    return [id_ for id_ in outlinks if id_ not in connected_ids]


def rank(outlinks: Dict[int, List[int]],
         eps: float = 0.01,
         d: float = 0.85) -> Dict[int, float]:
    """Returns the PageRank scores of the documents stored in
    outlinks.

    :param outlinks Mapping of doc ids to the ids that they link to
    :param eps The convergence threshold
    :param d The damping factor
    """
    # TODO: Implement PageRank here

    # This is a dictionary that create index for every pages
    # {6: 0, 993: 1, 1297: 2, 1461: 3,....}
    index_dict = dict()
    index = 0
    for key, value in outlinks.items():
        index_dict[key] = index
        index += 1

    # total number of pages
    total_num_page = len(outlinks)
    # initialize empty matrix
    matrix = np.zeros([total_num_page, total_num_page])

    for page in outlinks:
        page_index = index_dict[page]
        # sink pages vector
        if len(outlinks[page]) == 0:
            sink_weight_vec = np.zeros(total_num_page)
            sink_weight_vec.fill(1 / total_num_page)
            matrix[:, page_index] = sink_weight_vec
        # normal pages vector
        else:
            for outbound_page in outlinks[page]:
                outbound_page_index = index_dict[outbound_page]
                one_over_L = np.divide(1, len(outlinks[page]))
                matrix[outbound_page_index, page_index] += one_over_L

    # the constant term in equation
    const = np.divide(np.subtract(1, d), total_num_page)
    residual = np.zeros(total_num_page)
    residual.fill(const)

    # initialize vector ro
    ro = np.zeros(total_num_page)
    ro.fill(1 / total_num_page)

    # the first iteration
    r = np.add(residual, np.multiply(d, np.dot(matrix, ro)))
    error = np.sqrt(np.dot(np.subtract(r, ro), np.subtract(r, ro)))
    # iteration while loop update vector r
    while eps < error:
        ro = np.add(residual, np.multiply(d, np.dot(matrix, r)))
        error = np.sqrt(np.dot(np.subtract(r, ro), np.subtract(r, ro)))
        r = ro

    # save results into a dictionary
    page_rank_dict = {}
    for page in index_dict:
        page_index = index_dict[page]
        page_rank_dict[page] = r[page_index]

    # 1829 | 0.02120
    # 16774 | 0.01896
    # 3105 | 0.01807
    # print(page_rank_dict[1829])
    return page_rank_dict


def main2(collection, path):
    try:
        links_path = os.path.join(path, 'links.json')
        with open(links_path, 'r') as fp:
            with_str_keys = json.load(fp)
            outlinks = {int(key): val for key, val in with_str_keys.items()}
        print('Using existing links file')
    except FileNotFoundError:
        print('Creating new links file')
        outlinks = parse(collection)
        with open(links_path, 'w') as fp:
            json.dump(outlinks, fp)

    scores = rank(outlinks)
    score_path = os.path.join(path, 'scores.dat')
    with open(score_path, 'w') as fp:
        for id_, score in sorted(list(scores.items()), key=lambda p: -p[1]):
            fp.write('{}|{}\n'.format(id_, score))


if __name__ == '__main__':
    start_time = time.time()
    if len(sys.argv) <= 3 or len(sys.argv)>5:
        sys.exit('ERROR: Please input the values in the form "python create.py stopwords.dat collection.xml index/"')
    elif len(sys.argv) == 5:
        sys.exit('ERROR: The option to specify index files was removed in version 3a, please input an index directory instead')
    InputStop = sys.argv[1]
    Collection_pages = sys.argv[2]
    path = sys.argv[3]
    try:
        os.mkdir(str(path))
        print('Creating new directory')
    except:
        print('Using existing directory')
    main(InputStop, Collection_pages, path)
    main2(Collection_pages, path)
    print("--- create.py took %s seconds ---" % (time.time() - start_time))
    # Using link.py in file to write scores in path folder

