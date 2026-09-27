#!/bin/bash
# 在 8-gpu-2-fabric-pod-0 (node_rank=0) 上运行
set -ex

export MASTER_ADDR=10.121.3.77   # pod-0 自己的 eth0 IP（node_rank=0，torchrun 的 rendezvous host）
export MASTER_PORT=29500

export NCCL_SOCKET_IFNAME=eth0        # OOB 走 Pod 默认网络
export NCCL_IB_HCA=mlx5_0,mlx5_1      # 数据面用 2 张 RDMA NIC（fabric0 + fabric1）
export NCCL_IB_GID_INDEX=3            # RoCEv2 GID
export NCCL_DEBUG=INFO
export NCCL_CROSS_NIC=0               # rail-optimized 默认按 rail 对齐；如需对比可改成 1 重跑

torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /root/allreduce_bw_test.py
