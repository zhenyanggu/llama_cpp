import json
import os
from argparse import ArgumentParser

from llama_server_client import (
    DEFAULT_BASE_URL,
    chat_completion,
    extract_text_content,
    image_to_data_url,
)


def save_json(json_list, save_path):
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(json_list, file, indent=4)


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./OCRBench_Images")
    parser.add_argument("--output_folder", type=str, default="./results")
    parser.add_argument("--OCRBench_file", type=str, default="./sample_100.json")
    parser.add_argument("--save_name", type=str, default="SmolVLM2")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default="smolvlm2-gguf")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()

OCRBench_score = {
    "Regular Text Recognition": 0,
    "Irregular Text Recognition": 0,
    "Artistic Text Recognition": 0,
    "Handwriting Recognition": 0,
    "Digit String Recognition": 0,
    "Non-Semantic Text Recognition": 0,
    "Scene Text-centric VQA": 0,
    "Doc-oriented VQA": 0,
    "Key Information Extraction": 0,
    "Handwritten Mathematical Expression Recognition": 0,
}

AllDataset_score = {
    "IIIT5K": 0,
    "svt": 0,
    "IC13_857": 0,
    "IC15_1811": 0,
    "svtp": 0,
    "ct80": 0,
    "cocotext": 0,
    "ctw": 0,
    "totaltext": 0,
    "HOST": 0,
    "WOST": 0,
    "WordArt": 0,
    "IAM": 0,
    "ReCTS": 0,
    "ORAND": 0,
    "NonSemanticText": 0,
    "SemanticText": 0,
    "STVQA": 0,
    "textVQA": 0,
    "ocrVQA": 0,
    "ESTVQA": 0,
    "ESTVQA_cn": 0,
    "docVQA": 0,
    "infographicVQA": 0,
    "ChartQA": 0,
    "ChartQA_Human": 0,
    "FUNSD": 0,
    "SROIE": 0,
    "POIE": 0,
    "HME100k": 0,
}

num_all = {
    "IIIT5K": 0,
    "svt": 0,
    "IC13_857": 0,
    "IC15_1811": 0,
    "svtp": 0,
    "ct80": 0,
    "cocotext": 0,
    "ctw": 0,
    "totaltext": 0,
    "HOST": 0,
    "WOST": 0,
    "WordArt": 0,
    "IAM": 0,
    "ReCTS": 0,
    "ORAND": 0,
    "NonSemanticText": 0,
    "SemanticText": 0,
    "STVQA": 0,
    "textVQA": 0,
    "ocrVQA": 0,
    "ESTVQA": 0,
    "ESTVQA_cn": 0,
    "docVQA": 0,
    "infographicVQA": 0,
    "ChartQA": 0,
    "ChartQA_Human": 0,
    "FUNSD": 0,
    "SROIE": 0,
    "POIE": 0,
    "HME100k": 0,
}

def _should_log_progress(index, total, every):
    if total <= 0:
        return False
    if index == 1 or index == total:
        return True
    if every <= 0:
        return False
    return index % every == 0


def _score_prediction(dataset_name, answers, predict):
    if dataset_name == "HME100k":
        predict_norm = predict.strip().replace("\n", " ").replace(" ", "")
        if isinstance(answers, list):
            for answer in answers:
                if answer.strip().replace("\n", " ").replace(" ", "") in predict_norm:
                    return 1
            return 0
        return int(answers.strip().replace("\n", " ").replace(" ", "") in predict_norm)

    predict_norm = predict.lower().strip().replace("\n", " ")
    if isinstance(answers, list):
        for answer in answers:
            if answer.lower().strip().replace("\n", " ") in predict_norm:
                return 1
        return 0
    return int(answers.lower().strip().replace("\n", " ") in predict_norm)

if __name__ == "__main__":
    args = _get_args()

    data_path = args.OCRBench_file
    print(f"Loading data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)
    print(f"Evaluating {total_samples} samples against {args.base_url} with model {args.model}")

    for index, item in enumerate(data, start=1):
        img_path = os.path.join(args.image_folder, item["image_path"])
        question = item["question"]

        if _should_log_progress(index, total_samples, args.progress_every):
            print(f"[{index}/{total_samples}] {item['image_path']}")

        if not os.path.exists(img_path):
            print(f"Warning: Image not found, skipping: {img_path}")
            item["predict"] = f"ERROR: Image not found at {img_path}"
            continue

        try:
            messages_payload = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_to_data_url(img_path)
                            }
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ]

            response = chat_completion(
                base_url=args.base_url,
                model=args.model,
                messages=messages_payload,
                max_tokens=100,
                temperature=0.0,
                timeout=args.request_timeout,
            )
            item["predict"] = extract_text_content(response)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            item["predict"] = f"API_ERROR: {e}"

    for item in data:
        dataset_name = item["dataset_name"]
        answers = item["answers"]

        if "predict" not in item:
            continue

        predict = item["predict"]
        item["result"] = _score_prediction(dataset_name, answers, predict)

    os.makedirs(args.output_folder, exist_ok=True)
    save_json(data, os.path.join(args.output_folder, f"{args.save_name}.json"))

    for key in OCRBench_score:
        OCRBench_score[key] = 0

    OCRBench_num_all = {key: 0 for key in OCRBench_score}

    for key in AllDataset_score:
        AllDataset_score[key] = 0
    for key in num_all:
        num_all[key] = 0

    total_ocrbench_items = 0
    total_dataset_items = 0

    for item in data:
        item_type = item.get("type")
        if item_type and item_type in OCRBench_num_all:
            OCRBench_num_all[item_type] += 1
            total_ocrbench_items += 1
            if item.get("result") == 1:
                OCRBench_score[item_type] += 1

        dataset_name = item.get("dataset_name")
        if dataset_name and dataset_name in num_all:
            num_all[dataset_name] += 1
            total_dataset_items += 1
            if item.get("result") == 1:
                AllDataset_score[dataset_name] += 1

    if total_ocrbench_items > 0:
        recognition_score = (
            OCRBench_score["Regular Text Recognition"]
            + OCRBench_score["Irregular Text Recognition"]
            + OCRBench_score["Artistic Text Recognition"]
            + OCRBench_score["Handwriting Recognition"]
            + OCRBench_score["Digit String Recognition"]
            + OCRBench_score["Non-Semantic Text Recognition"]
        )
        recognition_total = (
            OCRBench_num_all["Regular Text Recognition"]
            + OCRBench_num_all["Irregular Text Recognition"]
            + OCRBench_num_all["Artistic Text Recognition"]
            + OCRBench_num_all["Handwriting Recognition"]
            + OCRBench_num_all["Digit String Recognition"]
            + OCRBench_num_all["Non-Semantic Text Recognition"]
        )

        Final_score = sum(OCRBench_score.values())
        Final_total = sum(OCRBench_num_all.values())

        print("###########################OCRBench##############################")

        # Only print recognition block if there are recognition items
        if recognition_total > 0:
            print(f"Text Recognition(Total {recognition_total}):{recognition_score}")
            print("------------------Details of Recognition Score-------------------")

            def print_score(type_name):
                score = OCRBench_score[type_name]
                total = OCRBench_num_all[type_name]
                if total > 0:
                    print(f"{type_name}(Total {total}): {score}")

            print_score("Regular Text Recognition")
            print_score("Irregular Text Recognition")
            print_score("Artistic Text Recognition")
            print_score("Handwriting Recognition")
            print_score("Digit String Recognition")
            print_score("Non-Semantic Text Recognition")
            print("----------------------------------------------------------------")

        def print_vqa_score(type_name):
            score = OCRBench_score[type_name]
            total = OCRBench_num_all[type_name]
            if total > 0:
                print(f"{type_name}(Total {total}): {score}")
                print("----------------------------------------------------------------")

        print_vqa_score("Scene Text-centric VQA")
        print_vqa_score("Doc-oriented VQA")
        print_vqa_score("Key Information Extraction")
        print_vqa_score("Handwritten Mathematical Expression Recognition")

        print("----------------------Final Score-------------------------------")
        print(f"Final Score(Total {Final_total}): {Final_score}")

    elif total_dataset_items > 0:
        print("###########################AllDataset##############################")
        for key in AllDataset_score.keys():
            if num_all[key] > 0:
                print(f"{key}: {AllDataset_score[key]/float(num_all[key])}")

    else:
        print("No valid data processed to generate a report.")
