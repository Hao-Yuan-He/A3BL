from concurrent.futures import thread
import os
import threading
import datetime
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", "--d", nargs="+", default=0, required=True)
    parser.add_argument("--cmd", "--c", default="", help="Command to run!")
    parser.add_argument("--times", "-t", default=1, type=int)
    args = parser.parse_args()
    cmds = [f"CUDA_VISIBLE_DEVICES={i} {args.cmd}" for i in args.devices] * args.times
    threads = []
    for cmd in cmds:
        th = threading.Thread(target=lambda: (os.system(cmd)))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()


if __name__ == "__main__":
    main()
