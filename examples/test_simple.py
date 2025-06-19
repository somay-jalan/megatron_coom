import os
import torch
import torch.distributed as dist

def run():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    print(f"[Rank {rank}] Approaching the barrier...")
    dist.barrier()
    print(f"✅ [Rank {rank}] Successfully passed the barrier!")
    dist.destroy_process_group()

if __name__ == "__main__":
    run()