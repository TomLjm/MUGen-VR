"""Report generation for evaluation results."""
import os
from datetime import datetime
from ..common.utils import ensure_dir, save_json


class ReportGenerator:
    """Generates structured reports in Markdown/HTML format."""

    def __init__(self, output_dir="./reports"):
        self.output_dir = output_dir
        ensure_dir(output_dir)

    def generate_markdown(self, eval_result, experiment_name="experiment"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Evaluation Report: " + experiment_name + "\n",
            "**Date**: " + timestamp + "\n",
            "\n## Overall Metrics\n",
        ]
        for evaluator_name, metrics in eval_result.metrics.items():
            if isinstance(metrics, dict):
                lines.append("\n### " + evaluator_name + "\n")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        lines.append("- **" + k + "**: " + f"{v:.4f}" + "\n")
                    else:
                        lines.append("- **" + k + "**: " + str(v) + "\n")
        report = "".join(lines)
        path = os.path.join(self.output_dir, experiment_name + "_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        return path

    def generate_html(self, eval_result, experiment_name="experiment"):
        md_path = self.generate_markdown(eval_result, experiment_name)
        html_path = md_path.replace(".md", ".html")
        html = "<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\"><title>" + experiment_name + "</title>\n"
        html += "<style>\nbody{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:900px;margin:auto;padding:2em;}\n"
        html += "h1{color:#1a73e8;}.metric{display:inline-block;margin:1em;padding:1em;background:#f5f5f5;border-radius:8px;}\n"
        html += ".value{font-size:2em;font-weight:bold;color:#1a73e8;}\n</style>\n</head>\n<body>\n"
        html += "<h1>" + experiment_name + "</h1>\n"
        for evaluator_name, metrics in eval_result.metrics.items():
            if isinstance(metrics, dict):
                html += "<h2>" + evaluator_name + "</h2>\n"
                for k, v in metrics.items():
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    html += "<div class=\"metric\"><div class=\"value\">" + val_str + "</div><div>" + k + "</div></div>\n"
        html += "\n</body>\n</html>"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return html_path
