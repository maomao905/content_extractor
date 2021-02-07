import logging
from collections import Counter
import re
import networkx as nx
import numpy as np
from difflib import SequenceMatcher
import math
import signal

from extractor.parser import Parser
from extractor.util import load_log_config, remove_space, time_limit

logging.config.dictConfig(load_log_config())
logger = logging.getLogger('applog.' + __name__)

NODE_TAGS = frozenset(['div', 'article', 'section', 'main', 'ul', 'li', 'p', 'header', 'blockquote', 'span',])
SKIP_TAGS = frozenset(['b', 'img', 'strong', 'pre', 'embed', 'input', 'i', 'small'])
# REGEX_EXCLUDE_PHRASES = re.compile('お?問い合わせ|All Rights Reserved', re.IGNORECASE)
PARAM_PRIOR_TITLE_COST = 5
PARAM_UNKNOWN_TITLE_LOCATION_RATIO = 0.2

class DOMTree():
    def __init__(self, parser_type='lxml', html_string=''):
        self.__G = None
        self.title_candidates = []
        self.title_location = None

        self.parser = Parser(parser_type, html_string)
        self._create_nodes(self.parser.body)
        self.title_el = self.__select_best_title()
        text_stats = self.set_text_density()
        self.set_distance_from_title(text_stats)

    def __select_best_title(self):
        if len(self.title_candidates) == 0:
            return None
        elif len(self.title_candidates) == 1:
            return self.title_candidates[0]
        title = self.parser.title
        if title == '':
            return self.title_candidates[0]

        t = sorted(self.title_candidates, key=lambda el: SequenceMatcher(None, title, self.parser.get_all_text(el)).quick_ratio(), reverse=True)
        return t[0]

    def _create_nodes(self, el, is_article=False):
        if self.__G is None:
            self.__G = nx.DiGraph()
            self.__G.add_node(el, score=0.0, is_article=False)

        for idx, el_ch in enumerate(self.parser.iter_children(el), 1):
            tag = self.parser.get_tag(el_ch)

            if tag in SKIP_TAGS:
                continue

            is_title = self.parser.is_title(el_ch)
            if is_title:
                self.title_candidates.append(el_ch)

            attrs = self.parser.get_attrs(el_ch)
            _is_article = is_article or tag == 'article' or 'article' in attrs
            self.__G.add_node(el_ch, score=0.0, is_article=_is_article)
            self.__G.add_edge(el, el_ch)
            if tag not in NODE_TAGS:
                continue

            self._create_nodes(el_ch, is_article=_is_article)

    def set_text_density(self):
        """
        Augment text statistics from the end node to calculate text density.
        """
        stats = {}
        def add_stat(el, num_of_tags, num_of_link_tags, text_length, link_text_length, text_density):
            if el in stats:
                stats[el]['num_of_tags'] += num_of_tags
                stats[el]['num_of_link_tags'] += num_of_link_tags
                stats[el]['text_length'] += text_length
                stats[el]['link_text_length'] += link_text_length
                stats[el]['text_density'] += text_density
            else:
                stats[el] = {
                    'num_of_tags': num_of_tags,
                    'num_of_link_tags': num_of_link_tags,
                    'text_length': text_length,
                    'link_text_length': link_text_length,
                    'text_density': text_density,
                }

        def get_body_stat():
            text = self.parser.get_all_text(self.parser.body)
            text_length = len(remove_space(text))
            link_els = self.parser._parser.find_all('a', self.parser.body)
            link_text_length = sum([len(remove_space(self.parser.get_all_text(_el))) for _el in link_els])
            return max(link_text_length, 1)/max(text_length, 1)

        edges = nx.dfs_labeled_edges(self.__G)
        body_stat = get_body_stat()

        for parent_el, el, d in edges:
            if d != 'reverse':
                continue
            stat = stats.get(el)
            num_of_tags = 0
            num_of_link_tags = 0
            text_length = 0
            link_text_length = 0
            text_density = 0.0

            # the end node
            if stat is None:
                __link_els = self.parser.find_all('a', el)
                __num_of_all_tags = self.parser.count_tag(el, recursive=True)
                __num_of_text_tags = self.parser.count_tag(el, 'p', recursive=True) + self.parser.count_tag(el, 'br', recursive=True)

                num_of_tags = __num_of_all_tags - __num_of_text_tags
                num_of_link_tags = len(__link_els)
                text_length = len(remove_space(self.parser.get_all_text(el)))

                if self.parser.get_tag(el) == 'a':
                    link_text_length = text_length
                    num_of_link_tags += 1
                    text_density = 0.0
                else:
                    link_text_length = sum([len(remove_space(self.parser.get_all_text(_el))) for _el in __link_els])
                    text_density = self.__calculate_text_density(num_of_tags, num_of_link_tags, text_length, link_text_length, body_stat)

            else:
                __num_of_all_tags = self.parser.count_tag(el)
                __num_of_text_tags = self.parser.count_tag(el, 'p') + self.parser.count_tag(el, 'br')

                num_of_tags = __num_of_all_tags - __num_of_text_tags
                text_length = len(remove_space(self.parser.get_text(el)))
                text_density = self.__calculate_text_density(
                    num_of_tags=num_of_tags + stat['num_of_tags'],
                    num_of_link_tags=stat['num_of_link_tags'],
                    text_length=text_length + stat['text_length'],
                    link_text_length=stat['link_text_length'],
                    body_stat=body_stat,
                )

            add_stat(el, num_of_tags, num_of_link_tags, text_length, link_text_length, text_density)
            # Since lxml `el.text` method does not include text following child node, we need to add tail text length to parent node.
            text_length += len(remove_space(self.parser.get_tail_text(el)))
            if stat is None:
                add_stat(parent_el, num_of_tags, num_of_link_tags, text_length, link_text_length, text_density)
            else:
                add_stat(
                    el=parent_el,
                    num_of_tags=num_of_tags + stat['num_of_tags'],
                    num_of_link_tags=stat['num_of_link_tags'],
                    text_length=text_length + stat['text_length'],
                    link_text_length=stat['link_text_length'],
                    text_density=text_density,
                )

        stat_text_density = {el: r['text_density'] for el, r in stats.items()}
        self.set_node_attributes('text_density', stat_text_density)
        return stats

    def __calculate_text_density(self, num_of_tags, num_of_link_tags, text_length, link_text_length, body_stat):
        if text_length == 0 or (text_length - link_text_length) <= 0:
            return 0.0
        base = np.log(((text_length/(text_length - link_text_length)) * link_text_length) + (body_stat * text_length) + np.e)
        M = (text_length/(link_text_length + 1)) * (max(num_of_tags, 1)/max(num_of_link_tags, 1))
        if M <= 0:
            return 0.0

        logger.debug(f'text_length={text_length}; link_text_length={link_text_length}; num_of_tags={num_of_tags}; num_of_link_tags={num_of_link_tags}; M={M}; base={base}')
        return (text_length/max(num_of_tags, 1)) * math.log(M, base)

    def has_node(self, node):
        return self.__G.has_node(node)

    def get_parent_node(self, node):
        parent_nodes = list(self.__G.predecessors(node))
        return parent_nodes[0] if len(parent_nodes) > 0 else None

    def get_descendants(self, node):
        return nx.descendants(self.__G, node)

    def set_node_attributes(self, label, values):
        nx.set_node_attributes(self.__G, values, label)

    def get_node_attributes(self, attr_name):
        return nx.get_node_attributes(self.__G, attr_name)

    def add_node_score(self, node, score):
        if self.has_node(node):
            nx.set_node_attributes(self.__G, {node: score}, 'score')
            nx.set_node_attributes(self.__G, {node: True}, 'content')
        else:
            logger.error(f'cannot add score: {self.parser.get_tag(node)}, {self.parser.get_attrs(node)}')

    def add_content_labels(self, node):
        attrs = {node: True}
        if self.__G.has_node(node):
            attrs.update({d: True for d in nx.descendants(node)})

        nx.set_node_attributes(self.__G, attrs, 'content')

    def set_distance_from_title(self, text_stats):
        locs = {}
        current_loc = 0.0
        for node in nx.dfs_preorder_nodes(self.__G):
            locs[node] = current_loc
            if node == self.title_el:
                self.title_location = current_loc

            current_loc += np.log(text_stats[node]['text_length'] + np.e)

        if self.title_location is None:
            self.title_location_auto = (sum(locs.values()) / len(locs)) * PARAM_UNKNOWN_TITLE_LOCATION_RATIO

        self.set_node_attributes('location', locs)

    def get_distance_from_title(self, el, location):
        if self.title_location is not None:
            diff = location - self.title_location
            # assign more costs when diff is negative, since content is more likely to appear subsequent to title
            dist = -diff * PARAM_PRIOR_TITLE_COST if diff < 0 else diff
            return dist
        else:
            return abs(location - self.title_location_auto)

    def iter_nodes(self):
        return self.__G.nodes.items()

    def get_all_edges(self):
        return self.__G.edges

    def get_all_nodes(self):
        return self.__G.nodes(data=True)

    def _draw(self):
        import matplotlib.pyplot as plt
        nx.draw_networkx(G)
        plt.savefig('tree.png')
