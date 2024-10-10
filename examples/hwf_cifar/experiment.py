import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(prog="HWF Experiment")
    parser.add_argument("--exp", type=str, default="abl", choices=["abl", "wsabl"])
    parser.add_argument("--dataset", type=str, default="HWF")
    parser.add_argument("--digit_size", type=int, default=1)
    parser.add_argument("--epoches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dist_func", type=str, choices=['hamming', 'confidence'], default='hamming')
    parser.add_argument(
        "--topk",
        type=int,
        default=-1,
        help="choose only top k candidates, k=-1 means use all of them.",
    )
    args = parser.parse_args()
    return args


def main(args):
    command = ""
    if args.exp == "abl":
        command = f"python nabl.py --dataset={args.dataset} --epoches={args.epoches} --seed={args.seed} --dist_func={args.dist_func}"
    elif args.exp == "wsabl":
        command = f"python wsabl.py --dataset={args.dataset} --epoches={args.epoches} --topk={args.topk} --seed={args.seed}"
    else:
        raise NotImplementedError
    print(command)
    os.system(command)


if __name__ == "__main__":
    args = get_args()
    main(args)
