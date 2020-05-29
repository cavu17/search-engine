import json
import re
from argparse import ArgumentParser
from typing import Dict, List
from xml.etree import cElementTree
import numpy as np



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
    squared_x=np.dot(x,x)
    length=np.sqrt(squared_x)
    return(length)


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

    #This is a dictionary that create index for every pages
    #{6: 0, 993: 1, 1297: 2, 1461: 3,....}
    index_dict = dict()
    index = 0
    for key, value in outlinks.items():
        index_dict[key] = index
        index += 1

    #total number of pages
    total_num_page = len(outlinks)
    #initialize empty matrix
    matrix = np.zeros([total_num_page,total_num_page])

    for page in outlinks:
        page_index = index_dict[page]
        #sink pages vector
        if len(outlinks[page]) == 0:
            sink_weight_vec = np.zeros(total_num_page)
            sink_weight_vec.fill(1/total_num_page)
            matrix[:, page_index] = sink_weight_vec
        #normal pages vector
        else:
            for outbound_page in outlinks[page]:
                outbound_page_index = index_dict[outbound_page]
                one_over_L = np.divide(1, len(outlinks[page]))
                matrix[outbound_page_index, page_index] += one_over_L

    #the constant term in equation
    const = np.divide(np.subtract(1, d), total_num_page)
    residual = np.zeros(total_num_page)
    residual.fill(const)

    #initialize vector ro
    ro = np.zeros(total_num_page)
    ro.fill(1/total_num_page)

    #the first iteration
    r = np.add(residual, np.multiply(d, np.dot(matrix, ro)))
    error = np.sqrt(np.dot(np.subtract(r, ro), np.subtract(r, ro)))
    #iteration while loop update vector r
    while eps < error:
        ro = np.add(residual, np.multiply(d, np.dot(matrix, r)))
        error = np.sqrt(np.dot(np.subtract(r, ro), np.subtract(r, ro)))
        r = ro

    #save results into a dictionary
    page_rank_dict = {}
    for page in index_dict:
        page_index = index_dict[page]
        page_rank_dict[page] = r[page_index]

    #1829 | 0.02120
    #16774 | 0.01896
    #3105 | 0.01807
    #print(page_rank_dict[1829])
    return page_rank_dict
    


def main(collection_path: str):
    """Saves the outlinks dictionary as a JSON file then computes
    and saves the PageRank scores.

    Note: We recommend that you don't change this code.
    """
    try:
        with open('links.json', 'r') as fp:
            with_str_keys = json.load(fp)
            outlinks = {int(key): val for key, val in with_str_keys.items()}
        print('Using existing links file')
    except FileNotFoundError:
        print('Creating new links file')
        outlinks = parse(collection_path)
        with open('links.json', 'w') as fp:
            json.dump(outlinks, fp)

    scores = rank(outlinks)
    with open('scores.dat', 'w') as fp:
        for id_, score in sorted(list(scores.items()), key=lambda p: -p[1]):
            fp.write('{}|{}\n'.format(id_, score))


if __name__ == '__main__':
    #start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument('collection')
    args = parser.parse_args()
    main(args.collection)
    #print("--- link.py took %s seconds ---" % (time.time() - start_time))
