import sys
import re
from nltk.stem import PorterStemmer
import json
import boolparser as BP
import numpy as np
import argparse
import os
from link import *


def cosine_function(a,b):
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a,b)
    asqua = np.square(a)
    bsqua = np.square(b)
    asum =np.sum(asqua)
    bsum = np.sum(bsqua)
    asqrt= np.sqrt(asum)
    bsqrt = np.sqrt(bsum)
    total = dot/(asqrt*bsqrt)
    return total

# To use for flag -t
def return_titles(page_numbers, title_dict):
    titles = []
    for i in page_numbers:
        titles.append(title_dict[i])
    return " ".join('\"'+ str(x)+ '\"' for x in titles)

# To use for flag -v
def return_pair(page_numbers, cos_scores, title_dict):
    pairs = []
    for i, j in zip(page_numbers, cos_scores):
        pairs.append([title_dict[i],j])
    pairs = sorted(pairs, key=lambda e: e[1], reverse=True)
    return "\n".join('{}: {}'.format(z[0], z[1]) for z in pairs)

# TO us for flag -v and -rank=pagerank
def return_pair_pr(page_numbers, pr_dict, title_dict):
    pairs = []
    page_rank_call = []
    for i in page_numbers:
        try:
            page_rank_call.append(pr_dict[i])
        except:
            page_rank_call.append(str(0))
    for i, j in zip(page_numbers, page_rank_call):
        pairs.append([title_dict[i],j])
    pairs = sorted(pairs, key=lambda e: e[1], reverse=True)
    return "\n".join('{}: {}'.format(z[0], z[1]) for z in pairs)

# To use in case of no flag
def returning_form(page_numbers):
    return " ".join(str(x) for x in page_numbers)

class Query:
    def __init__(self, query_search, stop_words, inverted_dict):
        self.query_search = query_search
        self.stop_words = stop_words
        self.inverted_dict = inverted_dict

    # Functions that universally cleans all types of queries
    # This method removes mimics the create.py for the query to ensure that an equal comparison is done
    def clean_queries(self):
        clean_tokens = []
        search_edit = str(self.query_search)
        search_edit = re.sub('[^a-zA-Z0-9]', ' ', search_edit)
        search_edit = re.split(r'[^a-zA-Z0-9]', search_edit)
        search_edit = list(filter(None, search_edit))
        search_edit = [word.lower() for word in search_edit]
        search_edit = [PorterStemmer().stem(word) for word in search_edit]
        for word in search_edit:
            if word not in self.stop_words:
                clean_tokens.append(word)
        return clean_tokens

# Getting final tf-idf scores
# Creates query vector, and weight vectors for the documents, and calculates the cosine sim
    def query_vectors(self,token, page_numbers):
        cos_score = []
        for p in page_numbers:
            inner = []
            query_vectors = []
            for t in token:
                try:
                    inner.append(len(self.inverted_dict[t][p]))
                except:
                    inner.append(0)
                try:
                    query_vectors.append(self.inverted_dict[t]['idf'])
                except:
                    query_vectors.append(0)
        # Any token with idf score of 0 is omitted (since it does not exist in index)
            new_query_vector = [i for i in query_vectors if i]
            inner = [j for i, j in zip(query_vectors, inner) if i]
            query_vectors = new_query_vector
            score = cosine_function(inner,query_vectors)
            cos_score.append(score)
        return cos_score

# Returns the page ranks in the same order of cosine scores
    def return_cos_rank(self, page_numbers, cos_scores):
        Z = [x for _, x in sorted(zip(cos_scores, page_numbers), reverse=True)]
        return Z

    def page_rank(self, page_numbers, pr_dict):
        page_rank_call = []
        for i in page_numbers:
            try:
                page_rank_call.append(pr_dict[i])
            except:
                page_rank_call.append(str(0))
        Z = [x for _, x in sorted(zip(page_rank_call, page_numbers), reverse=True)]
        return Z

class OneWordQuery(Query):
    def __init__(self, query_search, stop_words, inverted_dict):
        super().__init__(query_search, stop_words, inverted_dict)

    # Here we directly go into the inverted dictionary with the cleaned key to return the values
    def return_pages(self, token):
        if not token:
            return ''
        else:
            token = token[0]
            if (token not in self.stop_words) and (token in self.inverted_dict):
                return_list = [x for x in self.inverted_dict[token] if x != 'idf']
                return sorted(return_list, key=int)
            else:
                return ''

    # Instead of returning cosine score with one word, we will return the freq of token in document
    # Also use for ranking
    def return_freq(self, token, page_numbers):
        if not token:
            return ''
        occurance_list = []
        token = token[0]
        for p in page_numbers:
            occurance_list.append(len(self.inverted_dict[token][p]))
        return occurance_list

class FreeTextQuery(Query):
    def __init__(self, query_search, stop_words, inverted_dict):
        super().__init__(query_search, stop_words, inverted_dict)

    #This function is similiar to an OR statement; we find all the pages and take only the unique ones
    def return_pages(self, token):
        if not token:
            return ''
        returned_list = []
        for words in token:
            try:
                returned_list.extend(list(self.inverted_dict[words].keys()))
            except:
                pass
        returned_list = [x for x in returned_list if x!= 'idf']
        returned_list = sorted(list(set(returned_list)), key=int)
        if not returned_list:
            return ''
        else:
            return returned_list

class PhraseQuery(Query):
    def __init__(self, query_search, stop_words, inverted_dict):
        super().__init__(query_search, stop_words, inverted_dict)

# finds exact phrase match
    def return_pages(self, token):
        if not token:
            return ''
        returned_list = []
        for words in token:
            if (words not in self.stop_words) and (words in self.inverted_dict):
                returned_list.append(list(self.inverted_dict[words].keys()))
            else:
                return ''
        same_page = set(returned_list[0])
        for s in returned_list[1:]:
            same_page.intersection_update(s)
        # Now we have the intersecting pages of all the tokens
        same_page = list(same_page)
        same_page = [x for x in same_page if x != 'idf']
        final_list = []
        # Here we subtract all the positions of the pages by the index
        # If the words are actually next to each other, there should be a common value marking it as an exact phrase
        for x in same_page:
            temp_list = []
            for y in range(len(token)):
                temp_list.append(self.inverted_dict[token[y]][x])
            # Subtract the value of the index from each list
            for q in range(len(temp_list)):
                temp_list[q] = [(int(x) - q) for x in temp_list[q]]
            # If there is a common value, then there is a phrase match
            common_items = set.intersection(*map(set, temp_list))
            if len(common_items) != 0:
                final_list.append(x)
        final_list = sorted(list(set(final_list)), key=int)
        if not final_list:
            return ''
        else:
            return final_list

class BooleanQuery(Query):
    def __init__(self, query_search, stop_words, inverted_dict):
        super().__init__(query_search, stop_words, inverted_dict)

    # This method cleans up the all the tokens first while preserving the AND and ORs
    # before running it through the boolparser
    def boolean_clean(self):
        # clean_tokens = []
        search_edit = str(self.query_search)
        search_edit = re.sub('[^a-zA-Z0-9()]', ' ', search_edit)
        search_edit = re.split(r'[^a-zA-Z0-9()]', search_edit)
        search_edit = list(filter(None, search_edit))
        temp_list = []
        for r in search_edit:
            if (r != 'AND') and (r != 'OR'):
                r = PorterStemmer().stem(r.lower())
                temp_list.append(r)
            else:
                temp_list.append(r)
        # clean tokens
        query_again = " ".join(str(x) for x in temp_list)
        search = BP.bool_expr_ast(query_again)
        return search

    def return_pages(self, tup):
        tupls = []
        terms = []
        operator = tup[0]
        stem = tup[1]
        for item in stem:
            if type(item) == type(()):
                tupls += [item]
            elif type(item) == type('string'):
                terms.append(item)
        new_stem = terms + tupls
        for (w, item) in enumerate(new_stem):
            if type(item) == type('string') and new_stem[w] not in self.stop_words and new_stem[w] in self.inverted_dict:
                new_stem[w] = list(self.inverted_dict[new_stem[w]].keys())
            elif type(item) == type('string') and (new_stem[w] in self.stop_words or new_stem[w] not in self.inverted_dict):
                # Return empty list when tokens are not valid in the query
                new_stem[w] = []
            elif type(item) == type(()):
                # Using recursion to apply the same to the other tuples
                new_stem[w] = BooleanQuery.return_pages(self, item)
        results = new_stem[0]
        if operator == 'OR':
            for lst in new_stem[1:]:
                results = set(results).union(set(lst))
        elif operator == 'AND':
            for lst in new_stem[1:]:
                results = set(results).intersection(set(lst))
        results = [x for x in results if x != 'idf']
        return sorted(results, key=int)


class QueryFactory:
    @staticmethod

    def query_find(query_sort):
        if not str(query_sort):
            return 'Query is not a valid string.'

        elif ('AND' in query_sort) or ('OR' in query_sort) or ('(' in query_sort) and (')' in query_sort):
            search = BooleanQuery(query_sort, stop_words, inverted_dict)
            token_inputs = search.boolean_clean()
            page_numbers = search.return_pages(token_inputs)
            token = search.clean_queries()
            cos_scores = search.query_vectors(token, page_numbers)
            if results.rank == 'pagerank' and results.t:
                return search.page_rank(page_numbers, page_rank_dict)
            elif results.rank == 'pagerank' and results.v:
                return return_pair_pr(page_numbers,page_rank_dict, title_dict)
            elif results.rank == 'pagerank':
                return returning_form(search.page_rank(page_numbers, page_rank_dict))
            if results.t:
                return search.return_cos_rank(page_numbers, cos_scores)
            elif results.v:
                return return_pair(page_numbers, cos_scores, title_dict)
            else:
                return returning_form(search.return_cos_rank(page_numbers, cos_scores))

        elif ('(' in query_sort) and not (')' in query_sort) or ( not '(' in query_sort) and (')' in query_sort):
            return 'ERROR: Boolean query \'' + query_sort + '\' cannot be parsed.'

        elif query_sort[0] == '\"' and query_sort[-1] == '\"':
            search = PhraseQuery(query_sort, stop_words, inverted_dict)
            token = search.clean_queries()
            page_numbers = search.return_pages(token)
            cos_scores = search.query_vectors(token, page_numbers)
            if results.rank == 'pagerank' and results.t:
                return search.page_rank(page_numbers, page_rank_dict)
            elif results.rank == 'pagerank' and results.v:
                return return_pair_pr(page_numbers,page_rank_dict, title_dict)
            elif results.rank == 'pagerank':
                return returning_form(search.page_rank(page_numbers, page_rank_dict))
            if results.t:
                return search.return_cos_rank(page_numbers, cos_scores)
            elif results.v:
                return return_pair(page_numbers, cos_scores, title_dict)
            else:
                return returning_form(search.return_cos_rank(page_numbers, cos_scores))

        elif (query_sort[0] == '\"' and not query_sort[-1] == '\"') or (not query_sort[0] == '\"' and query_sort[-1] == '\"'):
            return 'ERROR: Phrase query \'' + query_sort + '\' cannot be parsed.'

        elif len(query_sort.split()) == 1:
            search = OneWordQuery(query_sort, stop_words, inverted_dict)
            token = search.clean_queries()
            page_numbers = search.return_pages(token)
            page_frequency = search.return_freq(token, page_numbers)
            if results.rank == 'pagerank' and results.t:
                return search.page_rank(page_numbers, page_rank_dict)
            elif results.rank == 'pagerank' and results.v:
                return return_pair_pr(page_numbers,page_rank_dict, title_dict)
            elif results.rank == 'pagerank':
                return returning_form(search.page_rank(page_numbers, page_rank_dict))
            if results.t:
                return search.return_cos_rank(page_numbers, page_frequency)
            elif results.v:
                return return_pair(page_numbers, page_frequency, title_dict)
            else:
                return returning_form(search.return_cos_rank(page_numbers, page_frequency))
        else:
            search = FreeTextQuery(query_sort, stop_words, inverted_dict)
            token = search.clean_queries()
            token = [x for x in token if (x in inverted_dict and x not in stop_words)]
            page_numbers = search.return_pages(token)
            cos_scores = search.query_vectors(token, page_numbers)
            if results.rank == 'pagerank' and results.t:
                return search.page_rank(page_numbers, page_rank_dict)
            elif results.rank == 'pagerank' and results.v:
                return return_pair_pr(page_numbers,page_rank_dict, title_dict)
            elif results.rank == 'pagerank':
                return returning_form(search.page_rank(page_numbers, page_rank_dict))
            if results.t:
                return search.return_cos_rank(page_numbers, cos_scores)
            elif results.v:
                return return_pair(page_numbers, cos_scores, title_dict)
            else:
                return returning_form(search.return_cos_rank(page_numbers, cos_scores))

if __name__ == '__main__':
    if len(sys.argv) <= 2 or len(sys.argv) > 5:
        sys.exit('ERROR: Please input the values in the form "python query.py stopwords.dat index/"')
    elif len(sys.argv) == 4 and sys.argv[3] == '-t':
        pass
    elif len(sys.argv) == 4 and sys.argv[3] == '-v':
        pass
    elif len(sys.argv) == 4 and sys.argv[1] == '--rank=pagerank':
        pass
    elif len(sys.argv) == 4 and sys.argv[1] == '--rank=tfidf':
        pass
    elif len(sys.argv) == 5:
        pass
    elif len(sys.argv) == 3:
        pass
    else:
        sys.exit('ERROR: The option to specify index files was removed in version 3a, please input an index directory instead')

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', choices=['pagerank', 'tfidf'])
    parser.add_argument('StopWords')
    parser.add_argument('IndexDir')
    parser.add_argument('-t', action='store_true', default=False)
    parser.add_argument('-v', action='store_true', default=False)
    results = parser.parse_args()
    try:
        stop_words = []
        with open(results.StopWords, 'r') as stopwords:
            for i in stopwords:
                stop_words.append(i.strip('\n'))
        # Opening the inverted index and adding putting it back into a dictionary
        index_path = os.path.join(results.IndexDir, 'index.json')
        with open(index_path, 'r') as inverted_index:
            inverted_dict = json.load(inverted_index)
        # Opening the title index and putting it back into a dictionary
        title_path = os.path.join(results.IndexDir, 'titles.json')
        with open(title_path, 'r') as title_index:
            title_dict = json.load(title_index)
        page_rank_dict = {}
        score_path = os.path.join(results.IndexDir, 'scores.dat')
        with open(score_path, 'r') as pr:
            for line in pr:
                line = line.strip('\n')
                split_line = line.split('|')
                page_rank_dict[split_line[0]] = split_line[1]
    except:
        sys.exit('ERROR: required files not present')

    while True:
        try:
            query_input = input()
            if results.t:
                print(return_titles(QueryFactory.query_find(query_input), title_dict))
            else:
                print(QueryFactory.query_find(query_input))
        except EOFError:
            break
