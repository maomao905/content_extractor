import logging
import lxml.html
from lxml.html.clean import Cleaner
from bs4 import BeautifulSoup
from bs4.element import Tag
import re
from readability import htmls

from extractor.util import load_log_config

logging.config.dictConfig(load_log_config())
logger = logging.getLogger('applog.' + __name__)

# Handle jQuery Lazy Load Plugin
IMAGE_URL_KEYS = ('src', 'data-lazy-src', 'data-original',)

cleaner = Cleaner(
    scripts=True, javascript=True, style=True, comments=True, forms=False,
    links=True, processing_instructions=True,
    kill_tags = ['footer', 'nav', 'select', 'button', 'noscript',],
)

class Parser():
    __REGEX_TITLE_ATTR = re.compile('title', re.IGNORECASE)
    __REGEX_NOT_TITLE_ATTR = re.compile('sub|side|related', re.IGNORECASE)

    def __init__(self, type_='lxml', html_string=''):
        if type_ == 'lxml':
            self._parser = LxmlParser(html_string)
        elif type_ == 'soup':
            self._parser = SoupParser(html_string)
        else:
            raise ValueError('parser type must be lxml or soup')

    @property
    def html(self):
        return self._parser.html

    @property
    def title(self):
        return self._parser.title

    @property
    def body(self):
        return self._parser.body

    def get_tag(self, el):
        return self._parser.get_tag(el)

    def select_one(self, path):
        return self._parser.select_one(path)

    def get_parent(self, el):
        return self._parser.get_parent(el)

    def get_all_text(self, el):
        return self._parser.get_all_text(el).strip()

    def get_text(self, el):
        return self._parser.get_text(el).strip()

    def get_tail_text(self, el):
        return self._parser.get_tail_text(el).strip()

    def get_attrs(self, el):
        if el is None:
            return []
        return self._parser.get_attrs(el)

    def get_parent_attrs(self, el):
        parent = self._parser.get_parent(el)
        return self.get_attrs(parent)

    def find_all(self, tag, el=None):
        els = self._parser.find_all(tag, el)
        return els if els is not None else []

    def iter_children(self, el):
        return self._parser.iter_children(el)

    def iter_ancestors(self, el):
        return self._parser.iter_ancestors(el)

    def drop(self, el):
        return self._parser.drop(el)

    def count_tag(self, el, tag=None, recursive=False):
        return self._parser.count_tag(el, tag, recursive)

    def is_title(self, el):
        if self._parser.get_tag(el) in ('h1', 'h2', 'h3'):
            return True

        for attr in self._parser.get_attrs(el):
            if self.__REGEX_TITLE_ATTR.search(attr) and \
                not self.__REGEX_NOT_TITLE_ATTR.search(attr):
                return True

        return False

    def get_image_urls(self, el):
        image_urls = []
        for img_el in self._parser.find_all('img', el):
            img = None
            for k in IMAGE_URL_KEYS:
                img = img_el.get(k)
                if img is not None:
                    break
            if img is None or img == '':
                continue
            image_urls.append(img)
        return image_urls

class LxmlParser():
    def __init__(self, html_string):
        # use readability `build_doc` func to avoid encoding error
        self.html, _ = htmls.build_doc(html_string)
        self.title = self.get_title(self.html)
        # Use self.html when self.html.body does not exist
        try:
            self.body = cleaner.clean_html(self.html.body)
        except Exception as e:
            self.body = cleaner.clean_html(self.html)
            logger.warn(repr(e))

        self.prepend_newline()

    # https://stackoverflow.com/questions/18660382/how-can-i-preserve-br-as-newlines-with-lxml-html-text-content-or-equivalent
    def prepend_newline(self):
        for br in self.body.xpath('*//br'):
            br.tail = f'\n{br.tail}' if br.tail else '\n'

    def get_title(self, html):
        head = html.find('head')
        if head is None:
            return ''
        title_el = head.find('title')
        return title_el.text.strip() if title_el is not None else ''

    def find_all(self, path, el=None):
        path = path if path.startswith('/') else f'.//{path}'
        if el is not None:
            return el.xpath(path)
        return self.body.xpath(path)

    def select_one(self, path):
        path = path if path.startswith('/') else f'.//{path}'
        els = self.body.xpath(path)
        return els[0] if len(els) > 0 else None

    def get_all_text(self, el):
        if self.get_tag(el) == 'br':
            return el.tail if el.tail is not None else ''
        return el.text_content()

    def get_text(self, el):
        if self.get_tag(el) == 'br':
            tail = el.tail
            return tail if tail is not None else ''
        text = el.text
        return text if text is not None else ''

    def get_tail_text(self, el):
        tail = el.tail
        return tail if tail is not None else ''

    def get_tag(self, el):
        try:
            return el.tag
        except AttributeError as e:
            logger.info(e)
            return ''

    def get_attrs(self, el):
        attrs = []
        attrib = el.attrib
        if 'itemprop' in attrib:
            attrs.append(attrib['itemprop'])
        elif 'role' in attrib:
            attrs.append(attrib['role'])
        elif 'id' in attrib:
            attrs.append(attrib['id'])
        elif 'class' in attrib:
            attrs.extend(attrib['class'].split())
        return attrs

    def get_parent(self, el):
        return el.getparent()

    def iter_children(self, el):
        return el.iterchildren()

    def iter_ancestors(self, el):
        return el.iterancestors()

    def drop(self, el):
        try:
            return el.drop_tree()
        except AssertionError: # assert parent is not None
            logger.warn('failed to drop node')
            return

    def count_tag(self, el, tag, recursive=False):
        tagName = tag if tag is not None else '*'
        if recursive:
            return el.xpath(f'count(.//{tagName})')
        else:
            return el.xpath(f'count(./{tagName})')

class SoupParser():
    def __init__(self, html_string):
        self.html = BeautifulSoup(html_string, 'lxml')
        self.title = self.get_title(self.html)
        self.body = self.html.body

    def get_title(self, html):
        head = html.find('head')
        if head is None:
            return ''
        title_el = head.find('title')
        return title_el.text.strip() if title_el is not None else ''

    def find_all(self, path, el=None):
        if el is not None:
            return el.select(path)
        return self.body.select(path)

    def select_one(self, path):
        return self.body.select_one(path)

    def get_all_text(self, el):
        return el.text

    def get_text(self, el):
        return ''.join(el.find_all(text=True, recursive=False))

    def get_tail_text(self, el):
        return el.tail if el.tail is not None else ''

    def get_tag(self, el):
        return el.name

    def get_parent(self, el):
        return el.find_parent()

    def get_attrs(self, el):
        attrs = []
        if 'itemprop' in el.attrs:
            attrs.append(el.attrs['itemprop'])
        elif 'role' in attrs:
            attrs.append(attrs['role'])
        elif 'id' in el.attrs:
            attrs.append(el.attrs['id'])
        elif 'class' in el.attrs:
            attrs.extend(el.attrs['class'])
        return attrs

    def iter_children(self, el):
        return [ch for ch in el.children if isinstance(ch, Tag)]

    def iter_ancestors(self, el):
        return el.findAllPrevious()

    def drop(self, el):
        el.decompose()

    def count_tag(self, el, tag, recursive=False):
        if tag is not None:
            return len(el.find_all(tag, recursive=recursive))
        else:
            return len(el.find_all(recursive=recursive))
