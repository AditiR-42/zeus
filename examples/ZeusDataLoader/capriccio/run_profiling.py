"""Run profiling for cifar100 dataset"""

import os
import argparse
import json

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile_folder", type=str, default=None, help="Unique folder name for profiling", required=True
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Maximum number of epochs to profile.", required=True
    )
    parser.add_argument(
        "--batch_sizes", type=int, nargs="+", help="Batch size.", required=True
    )
    parser.add_argument(
        "--power_limits", type=int, nargs="+", help="Define range of power limits", required=True
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=5, help="Warm up steps for profiling"
    )
    parser.add_argument(
        "--profile_steps", type=int, default=10, help="Profile steps"
    )
    parser.add_argument(
        "--data_dir", type=str, default="workspace/zeus/capriccio/data", help="Profile steps"
    )
    parser.add_argument(
        "--slice_number", type=int, default=9, help="Profile steps"
    )

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    power_limits = " ".join(str(pl) for pl in args.power_limits)
    os.system(f"mkdir {args.profile_folder}")
    for bs in args.batch_sizes:
        profile_path =f"{args.profile_folder}/{str(bs)}.json"
        os.system(
            f"python train.py --profile True --profile_path {profile_path} --epochs {args.epochs} --batch_size {bs} --power_limits {power_limits} --warmup_steps {args.warmup_steps} --profile_steps {args.profile_steps} --data_dir {args.data_dir} --slice_number {args.slice_number} --model_name_or_path bert-base-uncased"
        )

    result = {}
    for file in os.listdir(f"{args.profile_folder}"):
        with open(f"{args.profile_folder}/{file}", 'r') as infile:
            result.update(json.load(infile))
            key = str(file)[:-5]
            result[key] = result["measurements"]
            del result["measurements"]
    
    with open(f"{args.profile_folder}/profiling.json", 'w') as output_file:
        json.dump(result, output_file)

if __name__ == "__main__":
    main(parse_args())
    
