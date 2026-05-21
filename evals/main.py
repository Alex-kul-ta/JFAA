# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import os
import pprint

import yaml

from src.utils.distributed import cleanup_distributed, find_free_port, init_distributed

parser = argparse.ArgumentParser()
parser.add_argument("--val_only", action="store_true", help="only run eval", default=False)
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    help="which devices to use on local machine",
)
parser.add_argument(
    "--debugmode",
    type=bool,
    default=False,
    help="Setting this to true will not spin up new processes. "
    "The main code runs the main process, which makes it easier to debug with checkpointing.",
)
parser.add_argument(
    "--folder",
    type=str,
    help="location to save logs",
    default="",
)
parser.add_argument("--override_config_folder", action="store_true")
parser.add_argument("--checkpoint", type=str, help="location of pretrained ckpt")
parser.add_argument("--model_name", type=str, help="Model name")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--use_fsdp", action="store_true")
parser.add_argument("--master_addr", type=str, default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
parser.add_argument(
    "--master_port",
    type=str,
    default=os.environ.get("MASTER_PORT", "auto"),
    help="port for torch.distributed. Use an integer or 'auto'.",
)


def _device_to_id(device: str) -> str:
    if not device.startswith("cuda:"):
        raise ValueError(f"Only cuda devices are supported here, got {device!r}")
    return device.split(":", 1)[1]


def _resolve_master_port(master_port) -> int:
    if master_port not in (None, "", "auto"):
        return int(master_port)
    env_port = os.environ.get("MASTER_PORT")
    if env_port:
        return int(env_port)
    return find_free_port()


def _configure_local_worker_env(rank: int, world_size: int, device: str, master_addr: str, master_port: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _device_to_id(device)
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Each locally spawned worker only sees one GPU after CUDA_VISIBLE_DEVICES is set.
    os.environ["LOCAL_RANK"] = "0"



def process_main(args, rank, fname, world_size, devices):
    import logging

    if not args.debugmode:
        _configure_local_worker_env(
            rank=rank,
            world_size=world_size,
            device=devices[rank],
            master_addr=args.master_addr,
            master_port=int(args.master_port),
        )

    # Import after CUDA_VISIBLE_DEVICES / RANK / WORLD_SIZE are already set.
    from evals.scaffold import main as eval_main

    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        if args.val_only:
            params["val_only"] = True

        if args.checkpoint:
            params["model_kwargs"]["checkpoint"] = args.checkpoint

        if args.model_name:
            params["model_kwargs"]["pretrain_kwargs"]["encoder"]["model_name"] = args.model_name

        if args.batch_size:
            params["experiment"]["optimization"]["batch_size"] = args.batch_size

        if args.override_config_folder:
            params["folder"] = args.folder
        params["use_fsdp"] = args.use_fsdp
        logger.info("loaded params...")

    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)

    world_size, rank = init_distributed(port=int(args.master_port), rank_and_world_size=(rank, world_size))
    logger.info(
        "Running... rank=%s/%s master=%s:%s CUDA_VISIBLE_DEVICES=%s",
        rank,
        world_size,
        os.environ.get("MASTER_ADDR"),
        os.environ.get("MASTER_PORT"),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    try:
        eval_main(params["eval_name"], args_eval=params)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debugmode:
        # FSDP debugging (use torchrun)
        if args.use_fsdp:
            if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
                raise RuntimeError("--debugmode --use_fsdp expects torchrun to set RANK/WORLD_SIZE/LOCAL_RANK.")
            args.master_port = _resolve_master_port(args.master_port)
            process_main(
                args=args,
                rank=int(os.environ["RANK"]),
                fname=args.fname,
                world_size=int(os.environ["WORLD_SIZE"]),
                devices=args.devices,
            )
        # Single-GPU debugging
        else:
            args.master_port = _resolve_master_port(args.master_port)
            process_main(args=args, rank=0, fname=args.fname, world_size=1, devices=["cuda:0"])
    else:
        num_gpus = len(args.devices)
        args.master_port = _resolve_master_port(args.master_port)
        print(
            f"[launcher] master_addr={args.master_addr} master_port={args.master_port} "
            f"world_size={num_gpus} devices={args.devices}"
        )

        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

        procs = []
        for rank in range(num_gpus):
            p = mp.Process(target=process_main, args=(args, rank, args.fname, num_gpus, args.devices))
            p.start()
            procs.append(p)

        exit_code = 0
        for p in procs:
            p.join()
            if p.exitcode not in (0, None) and exit_code == 0:
                exit_code = int(p.exitcode)

        if exit_code != 0:
            raise SystemExit(exit_code)
