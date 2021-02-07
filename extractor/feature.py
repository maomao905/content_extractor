import logging
import requests
import csv
from collections import namedtuple

from extractor.dom import DOMTree
from extractor.util import load_log_config

logging.config.dictConfig(load_log_config())
logger = logging.getLogger('applog.' + __name__)

__REMOVE_CLASSES = ('clearfix', 'pc', 'sp',)

SKIP_TAGS = frozenset(['a', 'p', 'br', 'span',])

def get_content_related_scores(tree, els):
    score = {}
    top_content_nodes = set()
    for el in els:
        content_el = tree.parser.get_parent(el) if tree.parser.get_tag(el) == 'p' else el
        if not tree.has_node(content_el):
            continue
        score[content_el] = 1.0

        parent_el = tree.get_parent_node(content_el)
        if parent_el is None:
            top_content_nodes.add(content_el)
            continue
        score[parent_el] = 0.7

        grandparent_el = tree.get_parent_node(parent_el)
        if grandparent_el is None:
            top_content_nodes.add(parent_el)
            continue

        score[grandparent_el] = 0.3
        top_content_nodes.add(grandparent_el)

    return score, top_content_nodes

def get_attr_name(tree, node):
    attrs = tree.parser.get_attrs(node)
    attrs = [at for at in attrs if at not in __REMOVE_CLASSES]
    return attrs[0] if len(attrs) > 0 else '*'

def is_title_candidates(tree, node):
    return node in tree.title_candidates

def is_skip_tag(tree, node):
    tag = tree.parser.get_tag(node)
    return tag in SKIP_TAGS

def get_feature(data):
    features = []
    for i, d in enumerate(data, 1):
        logger.info(f"{d['url']}, {d['path']}")
        if i % 10 == 0:
            logger.info(f'processing {i}th doc.')
        try:
            res = requests.get(d['url'], timeout=30)
        except requests.exceptions.ConnectionError as e:
            logger.warn(e)
            continue
        except requests.exceptions.ContentDecodingError as e:
            logger.warn(e)
            continue
        except Exception as e:
            logger.warn(e)
            continue

        res.encoding = res.apparent_encoding
        parser_type = 'lxml' if d['path'].startswith('/') else 'soup'
        html = res.content if d['path'].startswith('/') else res.text

        try:
            tree = DOMTree(parser_type, html)
        except Exception as e:
            logger.warn(e)
            continue

        content_els = tree.parser.find_all(d['path'])
        if len(content_els) == 0:
            logger.info(f'no elements found for path: {d["path"]}')
            continue

        content_related_scores, top_content_nodes = get_content_related_scores(tree, content_els)
        if len(content_related_scores) == 0:
            continue
        tree.set_node_attributes('score', content_related_scores)

        content_attrs = {}
        for node in top_content_nodes:
            content_attrs.update({ch_node: True for ch_node in tree.get_descendants(node)})

        tree.set_node_attributes('content', content_attrs)

        feature = namedtuple('featues', ('attr_name', 'parent_attr_name', 'title_dist', 'text_density', 'is_article', 'score'))
        for node, attr in tree.iter_nodes():
            # ignore element inside content block
            if 'content' in attr and attr['score'] == 0.0:
                continue

            if is_skip_tag(tree, node):
                continue

            # ignore title element
            if is_title_candidates(tree, node):
                continue

            # get attribute of the element
            attr_name = get_attr_name(tree, node)

            # get attribute of the parent element
            parent_node = tree.parser.get_parent(node)
            parent_attr_name = get_attr_name(tree, parent_node)

            f = feature(
                attr_name=attr_name,
                parent_attr_name=parent_attr_name,
                title_dist=tree.get_distance_from_title(node, attr['location']),
                text_density=attr.get('text_density', 0.0),
                is_article=attr['is_article'],
                score=attr['score'],
            )
            logger.debug(f)
            features.append(f)

    mode = 'w'
    with open('data/features.csv', mode) as f:
        w = csv.writer(f)
        if mode == 'w':
            header = features[0]._fields
            w.writerow(header)
        for f in set(features):
            w.writerow(list(f))

    logger.info('save result.')
