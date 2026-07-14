from __future__ import annotations

import csv
import os
import re
from urllib.parse import urlparse


class DomainCategoryManager:
    """
    Classify URL domains using a user-provided domain-category CSV and heuristic fallback rules.
    """

    def __init__(self, category_file: str | None = None) -> None:
        """
        Create a domain-category manager.

        :param category_file: [str | None] CSV file with at least domain and category columns.
            None uses utils/DomainCategoryManager/domain_categories.csv.
        :return: None.
        """
        default_file = os.path.join(os.path.dirname(__file__), "domain_categories.csv")
        self.category_file = category_file or default_file
        self.domain_category = self._load_domain_categories(self.category_file)

    def _load_domain_categories(self, category_file: str) -> dict[str, str]:
        """
        Load exact domain-to-category mappings from a CSV file.

        Extra CSV columns such as count and source are ignored by the classifier.

        :param category_file: [str] Path to a CSV file with domain and category columns.
        :return: [dict[str, str]] Mapping from cleaned domain to category.
        """
        categories = {}
        if not os.path.exists(category_file):
            return categories

        with open(category_file, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                domain = self.clean_domain(row.get("domain"))
                category = str(row.get("category", "")).strip()
                if domain is not None and category != "":
                    categories[domain] = category
        return categories

    def clean_domain(self, value: object) -> str | None:
        """
        Normalize a URL/domain value to a comparable domain string.

        :param value: [object] Domain or URL value.
        :return: [str | None] Cleaned lower-case domain, or None for missing values.
        """
        if value is None:
            return None

        domain = str(value).strip()
        if domain == "" or domain.lower() == "nan":
            return None

        domain = domain.replace("`", "").strip()
        domain = domain.strip("[](){}<>\"'.,;:")
        if "://" in domain:
            try:
                parsed_url = urlparse(domain)
                domain = parsed_url.netloc or parsed_url.path
            except ValueError:
                domain = domain.split("://", 1)[1]
        domain = domain.replace("https://", "")
        domain = domain.replace("http://", "")
        domain = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9./:_-]+$", "", domain)
        if domain.startswith("www."):
            domain = domain[4:]
        domain = domain.split("/")[0]
        domain = domain.strip("[](){}<>\"'.,;:")
        if domain == "" or domain.lower() == "nan":
            return None
        return domain.lower()

    def infer_domain_category(self, value: object) -> str:
        """
        Infer a domain category from reusable heuristic rules.

        :param value: [object] Domain or URL value.
        :return: [str] Inferred category name.
        """
        domain = self.clean_domain(value)
        if domain is None:
            return "missing"

        if domain.endswith((
            ".md",
            ".json",
            ".js",
            ".mjs",
            ".db",
            ".yaml",
            ".yml",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".webp",
            ".pdf",
            ".csv",
            ".txt",
            ".toml",
            ".env",
            ".lock",
            ".cpp",
            ".log",
            ".load",
            ".post",
        )):
            return "internal_artifact"

        if (
            "github.com" in domain
            or "githubusercontent.com" in domain
            or "gitlab.com" in domain
            or "npmjs.com" in domain
            or "dev.to" in domain
            or "apify.com" in domain
            or "supabase.co" in domain
            or "railway.app" in domain
            or "vercel.app" in domain
            or "netlify.app" in domain
            or "render.com" in domain
            or "firebaseapp.com" in domain
            or "cloudflare" in domain
            or domain.endswith(".dev")
            or domain.endswith(".sh")
        ):
            return "developer_infrastructure"

        if (
            "trycloudflare.com" in domain
            or "ngrok" in domain
            or domain.startswith("192.168.")
            or domain.startswith("10.")
            or domain.startswith("172.16.")
            or domain.startswith("localhost")
            or re.match(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$", domain) is not None
        ):
            return "local_network"

        if "molt" in domain:
            return "molt_ecosystem"

        if "claw" in domain or "agent" in domain or "nohumans" in domain or "autonomous" in domain:
            return "agentic_ai"

        if (
            domain.endswith((".fun", ".cash", ".trade", ".gg"))
            or "wallet" in domain
            or "base.org" in domain
            or "basescan" in domain
            or "bitcoin" in domain
            or "crypto" in domain
            or "token" in domain
            or "coin" in domain
            or "flaunch" in domain
            or "dexscreener" in domain
            or "polymarket" in domain
            or "bybit" in domain
        ):
            return "crypto"

        if (
            "youtube.com" in domain
            or "youtu.be" in domain
            or "x.com" in domain
            or "twitter.com" in domain
            or "twimg.com" in domain
            or "discord" in domain
            or "kym-cdn.com" in domain
            or "t.me" == domain
            or domain.endswith(".t.me")
            or "twitch.tv" in domain
            or "imgur.com" in domain
            or "spotify.com" in domain
            or "linkedin.com" in domain
            or "reddit.com" in domain
            or "substack.com" in domain
            or "ycombinator.com" in domain
        ):
            return "social_platform"

        if "gmail.com" in domain or domain.endswith(".email") or "mail" in domain:
            return "communication"

        if (
            "gumroad.com" in domain
            or "buymeacoffee.com" in domain
            or "amazon." in domain
            or "amzn." in domain
            or "store." in domain
            or "stripe.com" in domain
            or "ko-fi.com" in domain
        ):
            return "commerce"

        if (
            "notion.site" in domain
            or "docs." in domain
            or "wiki" in domain
            or "doi.org" in domain
            or "nature.com" in domain
            or "zenodo.org" in domain
            or "ncbi.nlm.nih.gov" in domain
        ):
            return "knowledge_base"

        if domain.endswith(".bot"):
            return "agentic_ai"

        if domain.endswith((
            ".io",
            ".ooo",
            ".one",
            ".space",
            ".chat",
            ".art",
            ".witness",
            ".simulate",
            ".cn",
            ".africa",
            ".br",
            ".sx",
            ".top",
            ".living",
            ".la",
            ".moe",
            ".near",
            ".online",
            ".hk",
            ".fyi",
            ".me",
            ".work",
            ".club",
            ".new",
            ".date",
            ".gy",
        )):
            return "experimental_infrastructure"

        if domain.endswith((".ai", ".xyz", ".app", ".co", ".pro")):
            return "experimental_infrastructure"

        return "unknown"

    def categorize_domain(self, value: object) -> str:
        """
        Classify a URL/domain value into a domain category.

        Exact CSV mappings have priority. If no exact mapping exists, the category is
        inferred from reusable heuristic rules.

        :param value: [object] Domain or URL value.
        :return: [str] Category name.
        """
        domain = self.clean_domain(value)
        if domain is None:
            return "missing"

        if domain in self.domain_category:
            return self.domain_category[domain]

        return self.infer_domain_category(domain)
