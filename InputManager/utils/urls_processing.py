import ast
import re
import time
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from requests.exceptions import ConnectionError

try:
    import swifter  # noqa: F401
except ModuleNotFoundError:
    swifter = None


class URLPreprocessor:
    """Shared URL extraction, parsing, and unshortening helpers."""

    def __init__(self, logger: Any, file_name: str) -> None:
        """
        Store logger context and initialize progress counters.

        :param logger: LogManager instance used for progress logging.
        :param file_name: Name of the caller module used in logs.
        :return: None.
        """
        self.lm = logger
        self.file_name = file_name
        self.unshorten_counter = 0
        self.parsing_counter = 1

    def _apply_series(self, series: pd.Series, func: Any) -> pd.Series:
        """
        Apply a function to a series, using swifter when the optional dependency is available.

        :param series: Series to transform.
        :param func: Function applied to each value.
        :return: Transformed series.
        """
        if swifter is not None:
            return series.swifter.apply(func)
        return series.apply(func)

    def unshorten_url(self, url: str) -> str:
        """
        Resolve a possibly shortened URL, falling back to the original URL on errors.

        :param url: URL string to resolve.
        :return: Resolved URL string, or the original URL when resolution fails.
        """
        self.lm.printK(self.unshorten_counter, 1000, f"Unshortening url {str(self.unshorten_counter)}.")
        self.unshorten_counter += 1
        try:
            from unshortenit import UnshortenIt
        except ModuleNotFoundError:
            return url

        unshortener = UnshortenIt()
        try:
            solved_url = unshortener.unshorten(url)
        except ConnectionError:
            solved_url = url
        except Exception:
            solved_url = url
        return solved_url

    def parse_url(self, resolved_url: str) -> tuple[str, str] | tuple[float, float]:
        """
        Parse a URL into domain and path components.

        :param resolved_url: URL string to parse.
        :return: Tuple containing domain and path, or NaN values when parsing fails.
        """
        self.lm.printK(self.parsing_counter, 100000, f"Parsing url {str(self.parsing_counter)}.")
        self.parsing_counter += 1
        try:
            parsed_url = urlparse(resolved_url)
            domain_url = parsed_url.netloc.replace('www.', '')
            path_url = parsed_url.path
        except Exception:
            return np.nan, np.nan

        return domain_url, path_url

    def build_url_dataframe(
        self,
        filter_df: pd.DataFrame,
        known_url: Sequence[str],
        parse_urls: bool = True
    ) -> pd.DataFrame:
        """
        Explode URL lists and optionally resolve them to domain URL objects.

        :param filter_df: Dataframe containing a url_list column.
        :param known_url: Domains that should not be unshortened.
        :param parse_urls: Whether to parse/unshorten URLs before returning the dataframe.
        :return: Dataframe with exploded URL rows and URL/domain columns.
        """
        result_df = filter_df.explode('url_list')
        result_df = result_df.rename(columns={"url_list": "url"})

        result_df = result_df[result_df['url'].notna() & (result_df['url'] != '')]

        if result_df.empty:
            return result_df.rename(columns={"url": "domainUrl"}) if not parse_urls else result_df

        if parse_urls:
            known_url = known_url or []
            url_df = pd.DataFrame(result_df['url'].unique(), columns=["url"])
            self.lm.printl(f"{self.file_name}. Number of urls to be processed: {str(url_df.shape[0])}")

            start_time = time.time()
            url_df['domainUrl'], url_df['pathUrl'] = zip(*self._apply_series(url_df['url'], self.parse_url))
            self.lm.printl(f"{self.file_name}. First URL parsing seconds: {str((time.time() - start_time))}")

            known_df = url_df[url_df['domainUrl'].isin(known_url)].copy()
            not_known_df = url_df[~url_df['domainUrl'].isin(known_url)].copy()

            known_df['resolved_url'] = known_df['url']

            start_time = time.time()
            if not_known_df.empty:
                not_known_df['resolved_url'] = pd.Series(dtype=object)
            else:
                not_known_df['resolved_url'] = self._apply_series(not_known_df['url'], self.unshorten_url)
            self.lm.printl(f"{self.file_name}. URL unshortening seconds: {str((time.time() - start_time))}")

            start_time = time.time()
            if not not_known_df.empty:
                not_known_df['domainUrl'], not_known_df['pathUrl'] = zip(
                    *self._apply_series(not_known_df['resolved_url'], self.parse_url)
                )
            self.lm.printl(f"{self.file_name}. Second URL parsing seconds: {str((time.time() - start_time))}")

            url_df = pd.concat([known_df, not_known_df]).reset_index()

            self.lm.printl(f"{self.file_name}. Start merging url with original dataframe.")
            result_df = pd.merge(result_df, url_df, on="url")

            result_df = result_df.dropna()
            result_df = result_df.drop(columns=['index'])
        else:
            result_df = result_df.rename(columns={"url": "domainUrl"})

        return result_df

    def clean_url(self, url: str) -> str:
        """
        Remove surrounding punctuation and add a scheme to bare www URLs.

        :param url: URL-like string extracted from text.
        :return: Cleaned URL string.
        """
        url = url.strip(".,!?;:()[]{}<>\"'")

        if url.startswith("www."):
            url = "http://" + url

        return url

    def extract_urls(self, text: Any) -> list[str]:
        """
        Extract unique URL-like strings from free text.

        :param text: Text value that may contain URLs.
        :return: List of unique cleaned URL strings.
        """
        if pd.isna(text):
            return []

        url_pattern = re.compile(
            r"""(
                (?:https?://|www\.)[^\s<>"]+
                |
                (?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}
                (?:/[^\s<>"]*)?
            )""",
            re.VERBOSE
        )

        matches = url_pattern.findall(text)
        cleaned = [self.clean_url(match) for match in matches]

        return list(dict.fromkeys(cleaned))

    def fix_and_parse_urls(self, value: Any) -> list[str]:
        """
        Cleans and extracts URLs from messy inputs.
        Handles real Python lists, string representations of lists,
        and concatenated URLs glued together without commas.

        :param value: Raw URL-list value from a dataset row.
        :return: List of cleaned URL strings.
        """
        urls = []

        if isinstance(value, list):
            urls = value
        elif isinstance(value, str):
            try:
                parsed_value = ast.literal_eval(value)
                if isinstance(parsed_value, list):
                    urls = parsed_value
                else:
                    urls = [value]
            except (ValueError, SyntaxError):
                urls = [value]
        else:
            return []

        fixed_urls = []
        for url in urls:
            if not isinstance(url, str):
                continue

            parts = re.split(r'(?=https?://)', url)
            for part in parts:
                part = part.strip()
                if part.startswith("http"):
                    fixed_urls.append(part)

        return fixed_urls
