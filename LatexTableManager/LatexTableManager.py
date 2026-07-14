from __future__ import annotations

import csv
import os
from typing import Any, Sequence

from DirectoryManager import DirectoryManager
from Objects.TimeWindow.TimeWindow import TimeWindow
from utils.common_variables import top_community_latex_color_map

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")


class LatexTableManager:
    """
    Generate predefined LaTeX tables from framework CSV outputs.
    """

    def __init__(
        self,
        dataset_name: str,
        user_fraction: float | None,
        type_filter: str,
        tw: TimeWindow,
        list_ca: Sequence[Any],
        dict_ca_filter: dict[str, Any],
        cda: Any,
        community_color_map: dict[str, str] | None = None,
    ) -> None:
        """
        Create a LaTeX table manager.

        :param dataset_name: [str] Dataset name.
        :param user_fraction: [float | None] User-selection fraction used in the pipeline.
        :param type_filter: [str] User-selection strategy.
        :param tw: [TimeWindow] Time-window configuration.
        :param list_ca: [Sequence[Any]] Co-actions included in the multiplex network.
        :param dict_ca_filter: [dict[str, Any]] Filter configuration by co-action id.
        :param cda: [Any] Community-detection algorithm object.
        :param community_color_map: [dict[str, str] | None] Mapping from community id to LaTeX color.
            None uses utils.common_variables.top_community_latex_color_map.
        :return: None.
        """
        self.community_color_map = community_color_map or top_community_latex_color_map
        self.cda = cda
        self.dm = DirectoryManager(
            file_name,
            dataset_name,
            data_path=data_path,
            results=results,
            user_fraction=user_fraction,
            type_filter=type_filter,
            tw=tw,
            list_ca=list_ca,
            dict_ca_filter=dict_ca_filter,
            cda=cda,
        )

    def _read_csv_rows(self, csv_path: str) -> list[dict[str, str]]:
        """
        Read a CSV file as a list of dictionaries.

        :param csv_path: [str] Input CSV path.
        :return: [list[dict[str, str]]] CSV rows.
        """
        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            return list(csv.DictReader(csv_file))

    def _save_latex(self, latex_code: str, output_path: str | None) -> None:
        """
        Save LaTeX code when an output path is provided.

        :param latex_code: [str] Generated LaTeX table code.
        :param output_path: [str | None] Destination path. None skips saving.
        :return: None.
        """
        if output_path is None:
            return

        output_dir = os.path.dirname(output_path)
        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(latex_code)

    def _format_integer(self, value: Any) -> str:
        """
        Format an integer-like value as a LaTeX siunitx number.

        :param value: [Any] Numeric value.
        :return: [str] LaTeX number string.
        """
        return f"\\num{{{int(float(value))}}}"

    def _format_float(self, value: Any, digits: int = 3) -> str:
        """
        Format a float-like value as a LaTeX siunitx number.

        :param value: [Any] Numeric value.
        :param digits: [int] Number of decimal digits.
        :return: [str] LaTeX number string.
        """
        return f"\\num{{{float(value):.{digits}f}}}"

    def _format_percentage(self, value: float | None, digits: int = 2) -> str:
        """
        Format a fraction as a LaTeX siunitx percentage value.

        :param value: [float | None] Fraction in [0, 1]. None or zero is rendered as --.
        :param digits: [int] Number of decimal digits.
        :return: [str] LaTeX percentage value.
        """
        if value is None or value == 0:
            return "--"
        return f"\\num{{{value * 100:.{digits}f}}}"

    def _community_bullet(self, community_id: Any) -> str:
        """
        Return the stable colored bullet for a community.

        :param community_id: [Any] Community id.
        :return: [str] LaTeX colored bullet code.
        """
        community_key = str(community_id)
        color = self.community_color_map.get(community_key, "black")
        return f"\\textcolor{{{color}}}{{\\Large $\\bullet$}}"

    def _top_communities_structural_row(self, row: dict[str, str]) -> str:
        """
        Build one LaTeX row for the top-community structural statistics table.

        :param row: [dict[str, str]] CSV row from top communities summary.
        :return: [str] LaTeX table row.
        """
        values = [
            self._community_bullet(row["cid"]),
            self._format_integer(row["numActorLayer"]),
            self._format_integer(row["numActors"]),
            self._format_integer(row["numLayers"]),
            self._format_float(row["avg_weight"]),
            self._format_float(row["median_weight"]),
            self._format_float(row["std_weight"]),
        ]
        return "        " + " & ".join(values) + r" \\"

    def _community_order_from_rows(self, rows: list[dict[str, str]], top_n: int) -> list[str]:
        """
        Return the community order found in a CSV row list.

        :param rows: [list[dict[str, str]]] CSV rows containing a cid column.
        :param top_n: [int] Maximum number of communities to return.
        :return: [list[str]] Community ids in first-seen order.
        """
        community_order = []
        for row in rows:
            cid = row["cid"]
            if cid not in community_order:
                community_order.append(cid)
            if len(community_order) == top_n:
                break
        return community_order

    def _top_community_order(
        self,
        top_n: int,
        summary_csv_path: str | None,
        category_rows: list[dict[str, str]],
    ) -> list[str]:
        """
        Return the community order for top-community LaTeX tables.

        :param top_n: [int] Number of communities to include.
        :param summary_csv_path: [str | None] Optional summary CSV path. If present,
            this file defines the row order.
        :param category_rows: [list[dict[str, str]]] Category percentage rows used as
            fallback ordering.
        :return: [list[str]] Ordered community ids.
        """
        if summary_csv_path is not None and os.path.exists(summary_csv_path):
            return self._community_order_from_rows(self._read_csv_rows(summary_csv_path), top_n)
        return self._community_order_from_rows(category_rows, top_n)

    def build_top_communities_structural_table(
        self,
        csv_path: str | None = None,
        output_path: str | None = None,
        top_n: int = 10,
        caption: str | None = None,
        label: str = "tab:url_layers_communities_stats",
        scale: float = 0.85,
    ) -> str:
        """
        Generate the LaTeX table for structural statistics of the top multiplex communities.

        :param csv_path: [str | None] Path to <algorithm>_top_<n>_communities_summary.csv.
            None reads from the current algorithm community-analysis directory.
        :param output_path: [str | None] Optional path where the LaTeX code is saved.
            None saves into the multi-co-action latex directory.
        :param top_n: [int] Number of CSV rows to include.
        :param caption: [str | None] Table caption. None uses the default caption.
        :param label: [str] LaTeX label.
        :param scale: [float] Scalebox value.
        :return: [str] Generated LaTeX table code.
        """
        algorithm_name = self.cda.get_algorithm_name()
        input_csv_path = csv_path or f"{self.dm.path_community_analysis}{algorithm_name}_top_{top_n}_communities_summary.csv"
        output_latex_path = output_path or f"{self.dm.path_latex}{algorithm_name}_top_{top_n}_communities_summary_table.tex"

        rows = self._read_csv_rows(input_csv_path)[:top_n]
        table_caption = caption or (
            "Structural statistics of the ten largest multiplex communities. "
            "Nodes correspond to actor-layer tuples, while actors count distinct users independently of layer membership."
        )
        table_rows = "\n".join(self._top_communities_structural_row(row) for row in rows)

        latex_code = "\n".join([
            r"\begin{table}[t]",
            f"    \\caption{{{table_caption}}}",
            f"    \\label{{{label}}}",
            r"    \centering",
            r"    \setlength{\tabcolsep}{6pt}",
            f"    \\scalebox{{{scale}}}{{",
            r"    \begin{tabular}{crrrccc}",
            r"        \toprule",
            r"        & & & &",
            r"        \multicolumn{3}{c}{\textit{weight statistics}} \\",
            r"        \cmidrule(l){5-7}",
            r"        \textbf{com.} & \textbf{nodes} & \textbf{actors} & \textbf{nLayers} & \textbf{mean} & \textbf{median} & \textbf{stdDev} \\",
            r"        \midrule",
            table_rows,
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    }",
            r"\end{table}",
        ])

        self._save_latex(latex_code, output_latex_path)
        return latex_code

    def _url_category_percentage_map(self, rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
        """
        Convert URL category percentage rows into a lookup dictionary.

        :param rows: [list[dict[str, str]]] Rows from
            <algorithm>_top_<n>_communities_url_domain_category_percentage.csv.
        :return: [dict[tuple[str, str, str], float]] Mapping keyed by
            (community id, source, domain category).
        """
        percentage_map = {}
        for row in rows:
            percentage_map[(row["cid"], row["source"], row["domain_category"])] = float(row["percentage"])
        return percentage_map

    def _url_category_table_row(
        self,
        community_id: str,
        percentage_map: dict[tuple[str, str, str], float],
    ) -> str:
        """
        Build one row of the URL-category composition table.

        :param community_id: [str] Community id.
        :param percentage_map: [dict[tuple[str, str, str], float]] Percentage lookup
            keyed by community id, URL source, and category.
        :return: [str] LaTeX table row.
        """
        post_categories = [
            "unknown",
            "internal_artifact",
            "developer_infrastructure",
            "molt_ecosystem",
            "experimental_infrastructure",
        ]
        comment_categories = [
            "rss_aggregation",
            "internal_artifact",
            "developer_infrastructure",
            "agentic_ai",
            "experimental_infrastructure",
        ]
        values = [self._community_bullet(community_id)]
        values.extend(
            self._format_percentage(percentage_map.get((community_id, "postURL", category)))
            for category in post_categories
        )
        values.extend(
            self._format_percentage(percentage_map.get((community_id, "commentURL", category)))
            for category in comment_categories
        )
        return "        " + " & ".join(values) + r" \\"

    def build_url_category_composition_table(
        self,
        csv_path: str | None = None,
        output_path: str | None = None,
        top_n: int = 10,
        summary_csv_path: str | None = None,
        caption: str | None = None,
        label: str = "tab:url_category_results",
        scale: float = 0.76,
    ) -> str:
        """
        Generate the LaTeX table for URL-category percentages in top communities.

        The input CSV is expected to contain one row per community, URL source, and
        domain category, with percentages stored as fractions. The output table keeps
        post URL percentages and comment URL percentages separated and omits narrative
        summary columns.

        :param csv_path: [str | None] Path to
            <algorithm>_top_<n>_communities_url_domain_category_percentage.csv. None
            reads from the current algorithm community-analysis directory.
        :param output_path: [str | None] Optional path where the LaTeX code is saved.
            None saves into the multi-co-action latex directory.
        :param top_n: [int] Number of top communities to include.
        :param summary_csv_path: [str | None] Optional top-community summary CSV path
            used to order rows by community size. None uses the default summary path.
        :param caption: [str | None] Table caption. None uses the default caption.
        :param label: [str] LaTeX label.
        :param scale: [float] Scalebox value.
        :return: [str] Generated LaTeX table code.
        """
        algorithm_name = self.cda.get_algorithm_name()
        input_csv_path = (
            csv_path
            or f"{self.dm.path_community_analysis}{algorithm_name}_top_{top_n}_communities_url_domain_category_percentage.csv"
        )
        input_summary_path = (
            summary_csv_path
            if summary_csv_path is not None
            else f"{self.dm.path_community_analysis}{algorithm_name}_top_{top_n}_communities_summary.csv"
        )
        output_latex_path = (
            output_path
            or f"{self.dm.path_latex}{algorithm_name}_top_{top_n}_communities_url_category_table.tex"
        )

        rows = self._read_csv_rows(input_csv_path)
        percentage_map = self._url_category_percentage_map(rows)
        community_order = self._top_community_order(top_n, input_summary_path, rows)
        table_rows = "\n".join(
            self._url_category_table_row(community_id, percentage_map)
            for community_id in community_order
        )
        table_caption = caption or (
            "URL-category composition of post and comment URLs in the ten largest multiplex communities. "
            "Values are percentages within each source type: post URL categories are normalized over post URLs, "
            "and comment URL categories over comment URLs."
        )

        latex_code = "\n".join([
            r"\begin{table*}[t]",
            f"    \\caption{{{table_caption}}}",
            f"    \\label{{{label}}}",
            r"    \centering",
            r"    \setlength{\tabcolsep}{4pt}",
            f"    \\scalebox{{{scale}}}{{",
            r"    \begin{tabular}{crrrrrrrrrr}",
            r"        \toprule",
            r"        &",
            r"        \multicolumn{5}{c}{\textit{post URLs}}",
            r"        &",
            r"        \multicolumn{5}{c}{\textit{comment URLs}} \\",
            r"        \cmidrule(lr){2-6}\cmidrule(lr){7-11}",
            r"        \textbf{com.}",
            r"        & \textbf{UNK}",
            r"        & \textbf{ART}",
            r"        & \textbf{DEV}",
            r"        & \textbf{MOLT}",
            r"        & \textbf{EXP}",
            r"        & \textbf{RSS}",
            r"        & \textbf{ART}",
            r"        & \textbf{DEV}",
            r"        & \textbf{AI}",
            r"        & \textbf{EXP} \\",
            r"        \midrule",
            table_rows,
            r"        \bottomrule",
            r"        \multicolumn{11}{l}{{\scriptsize UNK: unknown domains.}} \\",
            r"        \multicolumn{11}{l}{{\scriptsize ART: internal artifacts (e.g., \texttt{*.md}, \texttt{*.json}, local configuration files).}} \\",
            r"        \multicolumn{11}{l}{{\scriptsize DEV: developer infrastructure (e.g., GitHub, Supabase, Railway, developer platforms).}} \\",
            r"        \multicolumn{11}{l}{{\scriptsize MOLT: Molt ecosystem URLs (e.g., \moltbook, MoltCities, Molt-related services).}} \\",
            r"        \multicolumn{11}{l}{{\scriptsize EXP: experimental infrastructure and emerging ecosystem platforms.}} \\",
            r"        \multicolumn{11}{l}{{\scriptsize RSS: RSS aggregation and syndicated content propagation.}} \\",
            r"        \multicolumn{11}{l}{{\scriptsize AI: agentic AI and autonomous-agent infrastructure.}} \\",
            r"    \end{tabular}",
            r"    }",
            r"\end{table*}",
        ])

        self._save_latex(latex_code, output_latex_path)
        return latex_code
