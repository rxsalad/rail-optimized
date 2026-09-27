import os
import time
import torch
import torch.distributed as dist

SIZES_GB = [1, 2, 4, 8]
WARMUP_ITERS = 3
ITERS = 5


def human_gbps(nbytes, seconds):
    return (nbytes / 1e9) / seconds


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "this test is point-to-point between exactly 2 ranks"

    print(f"[rank{rank}] initializing process group...", flush=True)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"[rank{rank}] process group initialized", flush=True)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    peer = 1 - rank

    for size_gb in SIZES_GB:
        nbytes = size_gb * 1024 * 1024 * 1024
        numel = nbytes // 4  # float32
        print(f"[rank{rank}] size={size_gb}GB allocating tensor...", flush=True)
        tensor = torch.empty(numel, dtype=torch.float32, device=device)
        if rank == 0:
            tensor.fill_(1.0)
        print(f"[rank{rank}] size={size_gb}GB warmup...", flush=True)

        # warmup
        for _ in range(WARMUP_ITERS):
            if rank == 0:
                dist.send(tensor, dst=peer)
            else:
                dist.recv(tensor, src=peer)
        torch.cuda.synchronize()
        print(f"[rank{rank}] size={size_gb}GB warmup done, benchmarking...", flush=True)
        dist.barrier()

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(ITERS):
            if rank == 0:
                dist.send(tensor, dst=peer)
            else:
                dist.recv(tensor, src=peer)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        avg_seconds = elapsed / ITERS
        gbps = human_gbps(nbytes, avg_seconds)

        if rank == 0:
            print(
                f"[size={size_gb}GB] avg_time={avg_seconds*1000:.2f}ms  "
                f"bandwidth={gbps:.2f} GB/s ({gbps*8:.2f} Gb/s)",
                flush=True,
            )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
