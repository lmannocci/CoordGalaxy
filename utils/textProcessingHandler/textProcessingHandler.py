from __future__ import annotations

import re
from typing import Sequence

import nltk
from iso_language_codes import language_name
from langdetect import detect
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
from nltk.tokenize import RegexpTokenizer, TweetTokenizer, sent_tokenize

from utils.LogManager.LogManager import LogManager


class TextProcessingHandler:
    """
    Utility class for simple text cleaning, tokenization, and language detection.
    """

    def __init__(self, logger: LogManager | None = None) -> None:
        """
        Create a text-processing handler.

        :param logger: [LogManager | None] Logger used for diagnostics. None creates the main logger.
        :return: None.
        """
        self.logger = logger or LogManager("main")

    def detect_language(self, text: str) -> str:
        """
        Detect the language of text.

        :param text: [str] Text whose language must be detected.
        :return: [str] Detected language name, or multilingual when detection cannot be mapped.
        """
        lang = detect(text)
        if lang == "zh-cn" or lang == "zh-tw":
            lang = "zh"
        try:
            language = language_name(lang).lower()
        except Exception:
            language = "multilingual"
            self.logger.printl("Error language detection")
        self.logger.printl("language=" + str(language))
        return language

    def transform_docs_to_sentences(self, docs: list[str]) -> list[str]:
        """
        Convert a list of posts to a list of sentences.

        :param docs: [list[str]] Documents to split into sentences.
        :return: [list[str]] Sentences extracted from all documents.
        """
        concat_posts, _posts = self.preprocessing_posts(docs)
        sent = sent_tokenize(concat_posts)
        return sent

    def transform_doc_to_sentences(self, doc: str) -> list[str]:
        """
        Convert a post to a list of sentences.

        :param doc: [str] Text of the post.
        :return: [list[str]] Sentences extracted from the document.
        """
        final_sent = []
        if doc != "":
            doc_lower = doc.lower().strip()
            sent = sent_tokenize(doc_lower)

            # nel dividere in frasi, commette degli errori, ovvero ci sono frasi composte solamente dalla punteggiatura di lunghezza 1, che vanno rimosse
            for sentence in sent:
                if len(sentence) == 1:
                    continue
                final_sent.append(sentence)

        return final_sent

    # scegli il miglior tokenize
    def tokenize_words(self, text: str, type_tokenizer: str) -> list[str]:
        """
        Tokenize text with the selected tokenizer.

        :param text: [str] Text to tokenize.
        :param type_tokenizer: [str] Tokenizer type: whitespace, punct, regex, or tweet.
        :return: [list[str]] Tokens.
        """
        if type_tokenizer == "whitespace":
            return text.split()
        if type_tokenizer == "punct":
            return nltk.wordpunct_tokenize(text)
        if type_tokenizer == "regex":
            tokenizer = RegexpTokenizer(r"\w+")
            # tokenizer = nltk.tokenize.RegexpTokenizer ('\w+')
            return tokenizer.tokenize(text)
        if type_tokenizer == "tweet":
            tokenizer = TweetTokenizer()
            return tokenizer.tokenize(text)
        return []

    def lower(self, token_posts: Sequence[Sequence[str]]) -> list[list[str]]:
        """
        Transform tokens to lower case.

        :param token_posts: [Sequence[Sequence[str]]] Tokenized documents.
        :return: [list[list[str]]] Tokenized documents with lower-cased tokens.
        """
        temp = []
        for elem in token_posts:
            words_post = []
            for token in elem:
                words_post.append(token.lower())
            temp.append(words_post)
        return temp

    # concatena post target
    def preprocessing_posts(self, docs: list[str]) -> tuple[str, list[str]]:
        """
        Convert docs to lower case, discard empty docs, remove white spaces, and return concatenated text.

        :param docs: [list[str]] Documents to process.
        :return: [tuple[str, list[str]]] Concatenated text and processed document list.
        """
        docs[:] = (value.lower() for value in docs if value != "")

        if len(docs) != 0:
            text = ". ".join(docs)
        else:
            text = ""
        return text.strip(), docs

    def remove_url(self, text: str) -> str:
        """
        Remove URLs from a document.

        :param text: [str] Text from which URLs are removed.
        :return: [str] Text without URLs.
        """
        return re.sub(r"http\S+", "", text)

    def remove_hashtags(self, text: str) -> str:
        """
        Remove hashtags from a document.

        :param text: [str] Text from which hashtags are removed.
        :return: [str] Text without hashtags.
        """
        return re.sub(r"#(\w+)", "", text)

    def extract_url(self, text: str) -> list[str]:
        """
        Extract URLs from a document.

        :param text: [str] Text from which URLs are extracted.
        :return: [list[str]] Extracted URLs.
        """
        return re.findall(r"http\S+", text)

    def extract_hashtags(self, text: str) -> list[str]:
        """
        Extract hashtags from a document.

        :param text: [str] Text from which hashtags are extracted.
        :return: [list[str]] Extracted hashtag values without the leading #.
        """
        return re.findall(r"#(\w+)", text)

    def remove_punctuation(self, tokens_posts: Sequence[Sequence[str]]) -> list[list[str]]:
        """
        Remove punctuation from tokenized posts.

        :param tokens_posts: [Sequence[Sequence[str]]] Tokenized documents.
        :return: [list[list[str]]] Tokenized documents without punctuation-only tokens.
        """
        temp = []
        for elem in tokens_posts:
            new_words = []
            for token in elem:
                new_word = re.sub(r"[^\w\s]", "", token)
                if new_word != "":
                    new_words.append(new_word)
            if new_words != []:
                temp.append(new_words)
        return temp

    def remove_stop_words(self, token_posts: Sequence[str], language: str) -> list[str]:
        """
        Remove stop words from a tokenized post.

        :param token_posts: [Sequence[str]] Tokens from which stop words are removed.
        :param language: [str] Stop-word language.
        :return: [list[str]] Tokens without stop words.
        """
        sw = stopwords.words(language)
        token_ns = []
        for word in token_posts:
            if word not in sw or word == "":
                token_ns.append(word)
        return token_ns


_DEFAULT_HANDLER = TextProcessingHandler()


def detectLanguage(text: str) -> str:
    """
    Compatibility wrapper for TextProcessingHandler.detect_language.

    :param text: [str] Text whose language must be detected.
    :return: [str] Detected language name.
    """
    return _DEFAULT_HANDLER.detect_language(text)


def transformDocs2Sentences(docs: list[str]) -> list[str]:
    """
    Compatibility wrapper for TextProcessingHandler.transform_docs_to_sentences.

    :param docs: [list[str]] Documents to split into sentences.
    :return: [list[str]] Sentences extracted from all documents.
    """
    return _DEFAULT_HANDLER.transform_docs_to_sentences(docs)


def transformDoc2Sentences(doc: str) -> list[str]:
    """
    Compatibility wrapper for TextProcessingHandler.transform_doc_to_sentences.

    :param doc: [str] Text of the post.
    :return: [list[str]] Sentences extracted from the document.
    """
    return _DEFAULT_HANDLER.transform_doc_to_sentences(doc)


def tokenizeWords(text: str, type_tokenizer: str) -> list[str]:
    """
    Compatibility wrapper for TextProcessingHandler.tokenize_words.

    :param text: [str] Text to tokenize.
    :param type_tokenizer: [str] Tokenizer type.
    :return: [list[str]] Tokens.
    """
    return _DEFAULT_HANDLER.tokenize_words(text, type_tokenizer)


def lower(token_posts: Sequence[Sequence[str]]) -> list[list[str]]:
    """
    Compatibility wrapper for TextProcessingHandler.lower.

    :param token_posts: [Sequence[Sequence[str]]] Tokenized documents.
    :return: [list[list[str]]] Lower-cased tokens.
    """
    return _DEFAULT_HANDLER.lower(token_posts)


def preprocessingPosts(docs: list[str]) -> tuple[str, list[str]]:
    """
    Compatibility wrapper for TextProcessingHandler.preprocessing_posts.

    :param docs: [list[str]] Documents to process.
    :return: [tuple[str, list[str]]] Concatenated text and processed documents.
    """
    return _DEFAULT_HANDLER.preprocessing_posts(docs)


def remove_URL(text: str) -> str:
    """
    Compatibility wrapper for TextProcessingHandler.remove_url.

    :param text: [str] Text from which URLs are removed.
    :return: [str] Text without URLs.
    """
    return _DEFAULT_HANDLER.remove_url(text)


def remove_hashtags(text: str) -> str:
    """
    Compatibility wrapper for TextProcessingHandler.remove_hashtags.

    :param text: [str] Text from which hashtags are removed.
    :return: [str] Text without hashtags.
    """
    return _DEFAULT_HANDLER.remove_hashtags(text)


def extract_URL(text: str) -> list[str]:
    """
    Compatibility wrapper for TextProcessingHandler.extract_url.

    :param text: [str] Text from which URLs are extracted.
    :return: [list[str]] Extracted URLs.
    """
    return _DEFAULT_HANDLER.extract_url(text)


def extract_hashtags(text: str) -> list[str]:
    """
    Compatibility wrapper for TextProcessingHandler.extract_hashtags.

    :param text: [str] Text from which hashtags are extracted.
    :return: [list[str]] Extracted hashtags.
    """
    return _DEFAULT_HANDLER.extract_hashtags(text)


def remove_punctuation(tokens_posts: Sequence[Sequence[str]]) -> list[list[str]]:
    """
    Compatibility wrapper for TextProcessingHandler.remove_punctuation.

    :param tokens_posts: [Sequence[Sequence[str]]] Tokenized documents.
    :return: [list[list[str]]] Tokens without punctuation-only values.
    """
    return _DEFAULT_HANDLER.remove_punctuation(tokens_posts)


def remove_stop_words(token_posts: Sequence[str], language: str) -> list[str]:
    """
    Compatibility wrapper for TextProcessingHandler.remove_stop_words.

    :param token_posts: [Sequence[str]] Tokens from which stop words are removed.
    :param language: [str] Stop-word language.
    :return: [list[str]] Tokens without stop words.
    """
    return _DEFAULT_HANDLER.remove_stop_words(token_posts, language)
