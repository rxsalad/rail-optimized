import os
import time
import torch
import torch.distributed as dist

SIZES_GB = [1, 2, 4, 8]
WARMUP_ITERS = 3
ITERS = 5


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if rank == 0:
        print(f"[info] world_size={world_size} (expect 16 = 2 pods x 8 GPUs)", flush=True)

    for size_gb in SIZES_GB:
        nbytes = size_gb * 1024 * 1024 * 1024
        numel = nbytes // 4  # float32
        tensor = torch.full((numel,), float(rank + 1), dtype=torch.float32, device=device)

        # warmup
        for _ in range(WARMUP_ITERS):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(ITERS):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        avg_seconds = elapsed / ITERS

        # NCCL-tests style bandwidth reporting:
        #   algbw = size / time                         (data moved per rank per call)
        #   busbw = algbw * 2*(n-1)/n                    (effective bus bandwidth for ring all-reduce)
        algbw_gbs = (nbytes / 1e9) / avg_seconds
        busbw_gbs = algbw_gbs * 2 * (world_size - 1) / world_size

        if rank == 0:
            print(
                f"[size={size_gb}GB] avg_time={avg_seconds*1000:.2f}ms  "
                f"algbw={algbw_gbs:.2f} GB/s  busbw={busbw_gbs:.2f} GB/s "
                f"({busbw_gbs*8:.2f} Gb/s)",
                flush=True,
            )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
