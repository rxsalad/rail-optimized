#!/bin/bash
# 在 8-gpu-2-fabric-pod-1 (node_rank=1) 上运行
set -ex

export MASTER_ADDR=10.121.3.77   # 必须填 node_rank=0（pod-0）的 eth0 IP，不是自己的
export MASTER_PORT=29500

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_CROSS_NIC=0

torchrun \
  --nnodes=2 \
  --node_rank=1 \
  --nproc_per_node=8 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /root/allreduce_bw_test.py
