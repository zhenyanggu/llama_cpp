import argparse
import json
import os
import time

from llama_server_client import (
    DEFAULT_BASE_URL,
    chat_completion,
    extract_text_content,
    image_to_data_url,
)

def save_json(json_list, save_path):
    """Saves a list of dictionaries to a JSON file."""
    with open(save_path, "w") as file:
        json.dump(json_list, file, indent=4)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run accuracy evaluation against a llama-server with OCRBench samples.")
    parser.add_argument(
        "-i", "--image_folder",
        type=str,
        default="./OCRBench_Images",
        help="Path to the folder containing OCRBench images."
    )
    parser.add_argument(
        "-d", "--OCRBench_file",
        type=str,
        default="./sample_30.json",
        help="Path to the sampled OCRBench JSON file (output of sample.py)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="acc_eval_results.json",
        help="Path to save the per-sample accuracy result JSON file."
    )
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default="local-model")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument(
        "--cache-prompt",
        dest="cache_prompt",
        action="store_true",
        help="Send cache_prompt=true for accuracy requests.",
    )
    parser.add_argument(
        "--no-cache-prompt",
        dest="cache_prompt",
        action="store_false",
        help="Send cache_prompt=false for accuracy requests.",
    )
    parser.set_defaults(cache_prompt=None)
    return parser.parse_args()


def prediction_matches(item):
    if item.get("predict", 0) == 0:
        return 0

    dataset_name = item["dataset_name"]
    answers = item["answers"]
    predict = item["predict"]

    if dataset_name == "HME100k":
        if type(answers) == list:
            for answer in answers:
                answer_norm = answer.strip().replace("\n", " ").replace(" ", "")
                predict_norm = predict.strip().replace("\n", " ").replace(" ", "")
                if answer_norm in predict_norm:
                    return 1
        else:
            answer_norm = answers.strip().replace("\n", " ").replace(" ", "")
            predict_norm = predict.strip().replace("\n", " ").replace(" ", "")
            if answer_norm in predict_norm:
                return 1
    else:
        if type(answers) == list:
            for answer in answers:
                answer_norm = answer.lower().strip().replace("\n", " ")
                predict_norm = predict.lower().strip().replace("\n", " ")
                if answer_norm in predict_norm:
                    return 1
        else:
            answer_norm = answers.lower().strip().replace("\n", " ")
            predict_norm = predict.lower().strip().replace("\n", " ")
            if answer_norm in predict_norm:
                return 1

    return 0


def preview_text(value, limit=80):
    value = str(value).replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."

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

if __name__ == "__main__":
    args = parse_args()

    data_path = args.OCRBench_file
    print(f"Loading data from: {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)

    base_url = getattr(args, "base_url", DEFAULT_BASE_URL)
    model = getattr(args, "model", "local-model")
    request_timeout = float(getattr(args, "request_timeout", 300.0))
    request_retries = int(getattr(args, "request_retries", 2))
    retry_delay = float(getattr(args, "retry_delay", 2.0))
    progress_every = int(getattr(args, "progress_every", 10))
    max_tokens = int(getattr(args, "max_tokens", 100))
    cache_prompt = getattr(args, "cache_prompt", None)
    total_items = len(data)
    correct_so_far = 0
    completed_items = 0
    eval_start = time.perf_counter()

    print(f"[acc] Starting accuracy evaluation: samples={total_items}, max_tokens={max_tokens}", flush=True)

    for i in range(len(data)):
        img_path = os.path.join(args.image_folder, data[i]["image_path"])
        qs = data[i]["question"]

        if not os.path.exists(img_path):
            print(f"Warning: Image not found, skipping: {img_path}")
            data[i]["predict"] = f"ERROR: Image not found at {img_path}"
            data[i]["result"] = 0
            completed_items += 1
            elapsed_s = time.perf_counter() - eval_start
            print(
                f"[acc] {completed_items}/{total_items} result=0 correct={correct_so_far}/{completed_items} "
                f"elapsed={elapsed_s:.1f}s image={data[i]['image_path']} status=missing-image",
                flush=True,
            )
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
                            "text": qs
                        }
                    ]
                }
            ]

            for attempt in range(request_retries + 1):
                try:
                    response = chat_completion(
                        base_url=base_url,
                        model=model,
                        messages=messages_payload,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        cache_prompt=cache_prompt,
                        timeout=request_timeout,
                    )
                    data[i]["predict"] = extract_text_content(response)
                    break
                except Exception as retry_error:
                    if attempt >= request_retries:
                        raise retry_error
                    import time
                    time.sleep(retry_delay)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            data[i]["predict"] = f"API_ERROR: {e}" # Record the error in the data

        data[i]["result"] = prediction_matches(data[i])
        completed_items += 1
        correct_so_far += int(data[i]["result"] == 1)
        elapsed_s = time.perf_counter() - eval_start
        print(
            f"[acc] {completed_items}/{total_items} result={data[i]['result']} "
            f"correct={correct_so_far}/{completed_items} elapsed={elapsed_s:.1f}s "
            f"image={data[i]['image_path']} type={data[i].get('type', '')} "
            f"dataset={data[i].get('dataset_name', '')} predict=\"{preview_text(data[i].get('predict', ''))}\"",
            flush=True,
        )

    for i in range(len(data)):
        data[i]["result"] = prediction_matches(data[i])

    # Save the final JSON with 'predict' and 'result' fields
    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    save_json(data, output_path)

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
            OCRBench_num_all[item_type] += 1 # Count total items for this type
            total_ocrbench_items += 1
            if item.get("result") == 1: # Only add score if result is 1 (correct)
                OCRBench_score[item_type] += 1

        dataset_name = item.get("dataset_name")
        if dataset_name and dataset_name in num_all:
            num_all[dataset_name] += 1 # Count total items for this dataset
            total_dataset_items += 1
            if item.get("result") == 1: # Only add score if result is 1 (correct)
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

            # Helper function to print only if total > 0
            def print_score(type_name):
                score = OCRBench_score[type_name]
                total = OCRBench_num_all[type_name]
                if total > 0: # Only print if this type was present
                    print(f"{type_name}(Total {total}): {score}")

            print_score("Regular Text Recognition")
            print_score("Irregular Text Recognition")
            print_score("Artistic Text Recognition")
            print_score("Handwriting Recognition")
            print_score("Digit String Recognition")
            print_score("Non-Semantic Text Recognition")
            print("----------------------------------------------------------------")

        # Helper function for VQA scores
        def print_vqa_score(type_name):
            score = OCRBench_score[type_name]
            total = OCRBench_num_all[type_name]
            if total > 0: # Only print if this type was present
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
            if num_all[key] > 0: # Only print if this dataset was present in the data
                print(f"{key}: {AllDataset_score[key]/float(num_all[key])}")

    else:
        print("No valid data processed to generate a report.")
