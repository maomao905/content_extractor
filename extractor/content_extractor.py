import logging

import re
from collections import namedtuple
import numpy as np
import pandas as pd

from extractor.dom import DOMTree
from extractor.util import load_log_config, clean_text
from extractor import feature as _feature

logging.config.dictConfig(load_log_config())
logger = logging.getLogger('applog.' + __name__)

REGEX_INVALID_ATTR = re.compile('header|related|footer|sns', re.IGNORECASE)
PARAM_THRESHOLD_RATIO      = 0.1
PARAM_MINIMUM_TEXT_DENSITY = 5.0

def extract(model, html_string, debug=False):
    tree = DOMTree('lxml', html_string)

    nodes = []
    drop_nodes = []
    features = []
    feature = namedtuple('feature', ('concat_attr_name', 'title_dist', 'text_density', 'is_article',))

    result = {
        'score': 0.0,
        'content': '',
        'image_urls': []
    }
    for node, attr in tree.iter_nodes():
        if _feature.is_title_candidates(tree, node):
            if node == tree.title_el:
                drop_nodes.append(node)
            continue

        if _feature.is_skip_tag(tree, node):
            continue

        attrs = tree.parser.get_attrs(node)
        if any(REGEX_INVALID_ATTR.search(attr) for attr in attrs):
            drop_nodes.append(node)
            continue

        # attribute of the element
        attr_name = _feature.get_attr_name(tree, node)

        # attribute of the parent element
        parent_node = tree.parser.get_parent(node)
        parent_attr_name = _feature.get_attr_name(tree, parent_node)
        
        attr_name = '_txt'
        parent_attr_name = '_txt'
        
        
        f = feature(
            concat_attr_name='|'.join([attr_name, parent_attr_name]),
            title_dist=tree.get_distance_from_title(node, attr['location']),
            text_density=attr.get('text_density', 0.0),
            is_article=attr['is_article'],
        )
        features.append(f)
        nodes.append(node)

    if len(features) == 0:
        logger.warn('there are no features')
        return result

    pred = model.predict(pd.DataFrame(features))
    best_idx = pred.argmax()
    best_score = float(pred[best_idx])
    best_node = nodes[best_idx]

    for node in drop_nodes:
        tree.parser.drop(node)

    for idx in np.where(pred < best_score * PARAM_THRESHOLD_RATIO)[0]:
        if features[idx].text_density < PARAM_MINIMUM_TEXT_DENSITY:
            # logger.debug(f'drop: {tree.parser.get_tag(nodes[idx])}{tree.parser.get_attrs(nodes[idx])} ' + \
            #     f'{round(float(pred[idx]), 5)}, {features[idx]}')
            tree.parser.drop(nodes[idx])

    result['score'] = best_score
    result['content'] = clean_text(tree.parser.get_all_text(best_node))
    result['image_urls'] = tree.parser.get_image_urls(best_node)

    if debug:
        for _idx, idx in enumerate(np.argsort(pred)[::-1]):
            logger.info(f'{round(float(pred[idx]), 5)}: {features[idx]}')

    return result
