import json
import os
import sys
from typing import List
from PIL import Image, ImageOps
import pandas as pd
import time

import psutil

from utils.teds import TEDS
from utils.constant import CELL_SPECIAL
from tatr_runner import TableTransformerRunner

def build_table_from_html_and_cell(
    structure: List[str], content: List[str] = None
) -> List[str]:
    """Build table from html and cell token list"""
    assert structure is not None
    html_code = list()

    # deal with empty table
    if content is None:
        content = ["placeholder"] * len(structure)

    for tag in structure:
        if tag in ("<td>[]</td>", ">[]</td>"):
            if len(content) == 0:
                continue
            cell = content.pop(0)
            html_code.append(tag.replace("[]", cell))
        else:
            html_code.append(tag)

    return html_code

html_table_template_anno = (
    lambda table: f"""<html>
        <head> <meta charset="UTF-8">
        <style>
        table, th, td {{
            border: 1px solid black;
            font-size: 10px;
        }}
        </style> </head>
        <body>
        <table>
            {table}
        </table> </body> </html>"""
)

html_table_template_pred = (
    lambda table: f"""<html>
        <head> <meta charset="UTF-8">
        <style>
        table, th, td {{
            border: 1px solid black;
            font-size: 10px;
        }}
        </style> </head>
        <body>
            {table}
        </body> </html>"""
)

DATA_PATH = r"C:\Users\jmgarzonv\Desktop\EAFIT\Tesis\ground_truth\synthtabnet"
GROUND_TRUTH_FILE = "test_synthetic_data.jsonl"
RESULTS_PATH = r".\results"
MODE = "structure"

import logging

def print_decoding_error(e):
    logging.error(f"Error decoding JSON from line: {e.doc[:50]}...")
    logging.error(f"Error message: {e}\n")

    # How many characters to show before and after
    context_range = 50  # Adjust this value as needed

    # Calculate start and end positions to slice
    start = max(0, e.pos - context_range)
    end = min(len(e.doc), e.pos + context_range)

    # Print the substring around the position of interest
    logging.error(e.doc[start:end])
    logging.error(" " * (e.pos - start) + "^")


def process_ground_truth_data(ground_truth_data):
    # Your code to process the ground truth data goes here
    anno_html_raw = ground_truth_data["html"]["structure"]["tokens"]
    anno_cell_raw = [
        "".join(cell["tokens"])
        for cell in ground_truth_data["html"]["cells"]
        if cell["tokens"]
    ]
    anno_html = []
    idx = 0
    while idx < len(anno_html_raw):
        if "[" in anno_html_raw[idx]:
            assert idx + 1 < len(anno_html_raw)
            assert anno_html_raw[idx + 1] == "]</td>"
            anno_html.append(anno_html_raw[idx] + "]</td>")
            idx = idx + 2
        else:
            anno_html.append(anno_html_raw[idx])
            idx = idx + 1

    anno_cell = []
    for txt in anno_cell_raw:
        for black in CELL_SPECIAL:
            txt = txt.replace(black, "")
        anno_cell.append(txt)

    anno_code = "".join(build_table_from_html_and_cell(anno_html, anno_cell))
    return anno_code


class TestModel:
    def __init__(self, image_path, ground_truth_data, padding=0):
        self.image = Image.open(image_path).convert("RGB")
        if padding > 0:
            self.image = ImageOps.expand(self.image, border=padding, fill="white")
        self.anno_code = process_ground_truth_data(ground_truth_data)

    def inference(self, model):
        self.pred_code = model.predict(self.image)
        self.pred_code = html_table_template_pred(self.pred_code)
        return self.pred_code

    def compute_metric(self):
        # Your code to compute the metric goes here
        # Evaluate table structure only (S-TEDS)
        metric = TEDS(structure_only=True)
        # print("Prediction: \n",self.pred_code)
        # print("Annotations: \n", html_table_template_anno(self.anno_code))
        value = metric.evaluate(self.pred_code, html_table_template_anno(self.anno_code))

        return value


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("execution.log", "w"), logging.StreamHandler()],
    )
    logging.info("Starting evaluation")
    evaluate_ground_truth()


def evaluate_ground_truth():
    directories = os.listdir(DATA_PATH)
    model = TableTransformerRunner(MODE)
    for directory in directories:

        results = []
        ground_truth_file = os.path.join(DATA_PATH, directory, GROUND_TRUTH_FILE)
        if not os.path.exists(ground_truth_file):
            logging.error(f"File {ground_truth_file} does not exist")
            continue

        with open(ground_truth_file, "r", encoding="utf-8") as file:
            count = 0
            for line in file:
                count += 1

                if count % 100 == 0:
                    memory_info = psutil.virtual_memory()
                    available_memory = memory_info.available
                    logging.info(f"Available Memory: {available_memory / (1024 ** 3):.2f} GB")
                line = line.strip()
                if not line:
                    continue
                try:
                    start_time = time.time()  # Start timing the execution
                    ground_truth_data = json.loads(line)
                    image_path = os.path.join(
                        DATA_PATH,
                        directory,
                        "images",
                        ground_truth_data["split"],
                        ground_truth_data["filename"],
                    )
                    test = TestModel(image_path, ground_truth_data, padding=50)
                    prediction = test.inference(model)
                    
                    teds = test.compute_metric()

                    # Compute the elapsed time
                    end_time = time.time() 
                    elapsed_time = end_time - start_time

                    results.append(
                        {
                            "directory": directory,
                            "filename": ground_truth_data["filename"],
                            "prediction": prediction,
                            "teds": teds,
                        }
                    )
                    # Execution report
                    logging.info(f"Processed: Directory: {directory}: {count}, TEDS: {teds:.2f}, Time: {elapsed_time} s.")
                except json.JSONDecodeError as e:
                    print_decoding_error(e)
            logging.info(f"{count} files processed for '{directory}'")
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"results_{directory}_{MODE}.csv", index=False, sep=";", )


if __name__ == "__main__":
    main()
