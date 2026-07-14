import json
import os
from typing import Any, Dict


class ReportGenerator:
    def __init__(self, output_dir="reports/demo_report"):
        self.output_dir = output_dir

    def write(self, payload: Dict[str, Any], filename="demo_report.md"):
        os.makedirs(self.output_dir, exist_ok=True)
        json_path = os.path.join(self.output_dir, "demo_report.json")
        md_path = os.path.join(self.output_dir, filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# MUGen-VR Demo Report\n\n")
            for section, content in payload.items():
                f.write(f"## {section}\n\n")
                f.write("```json\n")
                f.write(json.dumps(content, ensure_ascii=False, indent=2, default=str))
                f.write("\n```\n\n")
        return md_path
