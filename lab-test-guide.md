# Rail-Optimized GPU Fabric Lab Guide

> This document is both a lab record and a reproducible operating guide: follow the steps in order and you can rebuild this same RDMA passthrough network setup on a similar environment, and run through every validation experiment in this document (connectivity, RDMA bandwidth, GPU-to-GPU communication, NCCL AllReduce). Every step includes the actual command output we captured, so you can compare it against your own results.
>
> **Important caveat**: this is a simulated rail-optimized environment. None of the performance/bandwidth numbers in this document (RDMA, NCCL, etc.) should be taken as authoritative — every test here is meant to validate functional connectivity, not to serve as a rigorous performance benchmark.

<a id="toc"></a>
## Table of Contents

0. [Environment Overview and Known Limitations](#sec-0)
1. [RDMA/RoCE Passthrough Network: Principles and Deployment](#sec-1)
2. [Deploy and Verify the Test Workload](#sec-2)
3. [Experiment 1: Connectivity Test (Same-Rail / Cross-Rail Ping)](#sec-3)
4. [Experiment 2: RDMA Bandwidth Test (`ib_write_bw`)](#sec-4)
5. [Experiment 3: PyTorch GPU-to-GPU RDMA (Single NIC)](#sec-5)
6. [Experiment 4: PyTorch AllReduce (Dual NIC, 8 GPU/Pod)](#sec-6)
7. [Cleaning Up the Environment](#sec-7)
8. [Experiment 5: MPI Operator AllReduce (nccl-tests, Dual NIC, 8 GPU/Pod)](#sec-8)
9. [Pitfall Quick-Reference Table](#sec-9)
10. [To-Do / Next Steps](#sec-10)
11. [Appendix: Full YAML and Script Code](#sec-11)

---

<a id="sec-0"></a>
## 0. Environment Overview and Known Limitations
[↑ Back to TOC](#toc)

### 0.1 Cluster

- **Cluster**: DO Kubernetes (DOKS), version v1.36.0.
- **Access method**: accessed through the jump host `nuc`, which already has `kubectl` configured with a kubeconfig; you can also SSH directly from `nuc` to each node's public IP (the two GPU nodes' private IPs cannot reach each other directly — you have to go through the public IP via `nuc`).

### 0.2 Nodes

| Node | Role | Private IP | Public IP | GPU |
|---|---|---|---|---|
| rs-cpu-pool-3y0p31 | CPU pool | 10.120.0.6 | 165.22.160.233 | none |
| rs-gpu-pool-3xnkk5 | GPU pool | 10.120.0.7 | 138.68.59.208 | 8x NVIDIA B300 SXM6 |
| rs-gpu-pool-3xnkkk | GPU pool | 10.120.0.8 | 134.209.13.101 | 8x NVIDIA B300 SXM6 |

OS: Debian 13 (trixie), Kernel 6.12.94, containerd 2.2.3.

### 0.3 Limitations You Need to Know Before Starting

- **None of the performance/bandwidth numbers in this document should be treated as authoritative** — the goal is to validate that things are functionally connected, not to benchmark peak performance.
- **Known hardware issue**: of the 2 B300x8 GPU nodes, one has a bad RDMA NIC, which means the RDMA NIC numbering on the two machines stops lining up from `fabric3` onward (on the two nodes, `fabric3` and beyond are no longer the same physical rail). Because of this, every experiment in this document only uses the 2 RDMA NICs that are aligned across both machines: `fabric0`/`fabric1`. If your environment's NICs are all healthy, we'd recommend using the same method first to confirm which rail numbers actually line up across your machines before deciding which rails to use for testing.

---

<a id="sec-1"></a>
## 1. RDMA/RoCE Passthrough Network: Principles and Deployment
[↑ Back to TOC](#toc)

### 1.1 Overall Network Architecture Principles

- **Baseline topology (before the spine layer was introduced)**: each SU (Scaling Unit) is made up of up to 64 B300x8 servers; the RDMA backend network is isolated at the SU level — RDMA-dependent distributed applications are confined to a single SU and cannot communicate across SUs. The network topology is purely rail-only, consisting of 16 Layer-2 domains (corresponding to the `fabric0`~`fabric15` rails used throughout this document — each rail forms its own L2 broadcast domain, and they don't talk to each other).
- The newly added **spine layer** connects the RDMA backend networks of the individual SUs (Scaling Units), which were previously isolated from each other, making cross-SU, cross-rail communication possible.
- As a result, the RDMA backend network is upgraded from isolated **Layer-2 domains** into a single **Layer-3 routed network**: IP addressing is planned and allocated per infrastructure layer (server, leaf, fabric), and routes are summarized at the corresponding layers too.
- Every RDMA NIC on a DOKS worker node gets two kinds of IPv6 addresses at the same time:
  - a **link-local address** in the `fe80::/10` range;
  - an **IPv6 ULA address** assigned per infrastructure layer (e.g. `fd02::/64`, the same family as the `fd02:0:0:58::1`-style addresses used throughout this document).
- The rail-optimized GPU fabric has **ECMP enabled by default** (equal-cost multi-path), providing multiple available paths for cross-rail/cross-SU communication.
- On DOKS worker nodes, **each RDMA NIC is associated with its own independent VRF and routing table**, providing an isolated routing context for IP-based connectivity and path establishment (Section 2.4 and Section 3 will repeatedly demonstrate this "each NIC has its own independent routing table" phenomenon — this is where it comes from).
- To let applications running in a Pod use these RDMA NICs, DO provides a dedicated `ip-preserving-host-device` CNI plugin — it moves the NIC into the Pod's network namespace while preserving its IPv6 ULA address and associated routes. **The standard Host-Device plugin doesn't have this capability**; we'll deploy and verify this plugin next.

### 1.2 How the `ip-preserving-host-device` Plugin Works

It's an extended version of the standard Host-Device CNI plugin — the standard version doesn't preserve the original network configuration when it moves a NIC into a Pod's network namespace, whereas this plugin fully preserves the RDMA NIC's original IPv6 network identity while moving it, including:

- the NIC's IPv6 ULA address (e.g. `fd02:0:0:58::1`)
- its link-local IPv6 address
- the IPv6 routes associated with the interface
- its VRF membership and the corresponding routing table

In other words, once the RDMA NIC moves from the host's network namespace into a Pod, its network identity and routing configuration come along unchanged — RDMA applications inside the Pod can immediately use the same fabric-specific IP and routing configuration the NIC already had on the DOKS worker node, with zero extra configuration. You'll see this in action later in Section 2 when we deploy the test workload, and again in Section 2.4 when we verify the NIC is handed back.

### 1.3 Step 1: Deploy `daemonset.yaml` (Install the CNI Plugin)

Full contents in Appendix 11.1. What it does: deploys the `ip-preserving-host-device-cni` DaemonSet in `kube-system`, running on every node (including the CPU pool), in privileged mode, installing the `ip-preserving-host-device` CNI plugin (image `ghcr.io/digitalocean-packages/ip-preserving-host-device:v1.0.0`), which passes the host's RDMA NICs through into Pods while preserving their original IPs.

```bash
kubectl apply -f daemonset.yaml
```

Expected output:

```
$ kubectl apply -f daemonset.yaml
daemonset.apps/ip-preserving-host-device-cni created

$ kubectl get ds -n kube-system ip-preserving-host-device-cni
NAME                            DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   AGE
ip-preserving-host-device-cni   3         3         3       3            3           15s
```

**Checkpoint**: confirm the CNI binary was actually delivered onto the host (not just that the Pod shows Running):

```
root@rs-gpu-pool-3xnkkk:/opt/cni/bin# ls -la
...
-rwxr-xr-x 1 root root  5159967 Apr 24 10:38 host-device               # standard host-device plugin
-rwxr-xr-x 1 root root  4006048 Sep 27 15:33 ip-preserving-host-device  # plugin installed by this DaemonSet run, timestamp matches the apply time
-rwxr-xr-x 1 root root 50876101 Sep 26 00:03 multus-shim
...
```

The `ip-preserving-host-device` binary is confirmed present, sitting alongside the standard `host-device` plugin in the same directory (`/opt/cni/bin`, which Multus reads from), with a timestamp matching the `apply` time — confirming the DaemonSet's `install` container really did distribute the plugin to the node.

### 1.4 Step 2: Deploy `network-attachments.yaml` (Define a NAD for Each RDMA NIC)

Full contents in Appendix 11.2. What it does: a `NetworkAttachmentDefinition` (NAD) is Multus CNI's configuration object for defining an "attachment network" — Multus reads these definitions to decide how to create the corresponding network interface and attach it to a Pod (the Pod only needs to reference a NAD's name in an annotation, e.g. `roce-net-fabric0@pod-fabric0`; all the details of how the interface gets created are encapsulated inside the NAD). This file defines 16 NADs (`roce-net-fabric0` ~ `roce-net-fabric15`), each corresponding to one physical RDMA NIC on the host (`fabric0`~`fabric15`), all of type `ip-preserving-host-device`.

```bash
kubectl apply -f network-attachments.yaml
```

Expected output:

```
$ kubectl apply -f network-attachments.yaml
networkattachmentdefinition.k8s.cni.cncf.io/roce-net-fabric0 created
... (fabric1 ~ fabric15 created one after another, 16 total)

$ kubectl get network-attachment-definitions -A | wc -l
17   # 16 NADs + 1 header row
```

### 1.5 Physical Affinity Between GPUs and RDMA NICs (PCI Topology)

Before running any actual experiments, it's worth first nailing down one thing: **physically, which GPU is each of these RDMA NICs closest to?** This determines where the topological bottleneck will be in later multi-GPU tests. Instead of relying on the higher-level abstraction of `nvidia-smi topo -m`, we derive this directly from the raw PCI bus topology (`lspci`). Using node `rs-gpu-pool-3xnkkk` as the example:

**Step 1: get each GPU's PCI Bus ID**

```
root@rs-gpu-pool-3xnkkk:~# nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader
0, 00000000:85:00.0
1, 00000000:8D:00.0
2, 00000000:95:00.0
3, 00000000:9D:00.0
4, 00000000:A5:00.0
5, 00000000:AD:00.0
6, 00000000:B5:00.0
7, 00000000:BD:00.0
```

**Step 2: get each mlx5 NIC's PCI Bus ID** (`ibv_devinfo`/`rdma link` only give the device name, e.g. `mlx5_0`; to get the actual PCI address you need to follow the `device` symlink in sysfs)

```
root@rs-gpu-pool-3xnkkk:~# for i in $(seq 0 15); do
>   dev=mlx5_$i
>   pci=$(readlink -f /sys/class/infiniband/$dev/device | sed "s#.*/##")
>   echo "$dev -> $pci"
> done
mlx5_0  -> 0000:86:00.0
mlx5_1  -> 0000:86:00.1
mlx5_2  -> 0000:8e:00.0
mlx5_3  -> 0000:8e:00.1
mlx5_4  -> 0000:96:00.0
mlx5_5  -> 0000:96:00.1
mlx5_6  -> 0000:9e:00.0
mlx5_7  -> 0000:9e:00.1
mlx5_8  -> 0000:a6:00.0
mlx5_9  -> 0000:a6:00.1
mlx5_10 -> 0000:ae:00.0
mlx5_11 -> 0000:ae:00.1
mlx5_12 -> 0000:b6:00.0
mlx5_13 -> 0000:b6:00.1
mlx5_14 -> 0000:be:00.0
mlx5_15 -> 0000:be:00.1
```

**Step 3: print the full PCIe bus tree with `lspci -tv`** to see which NICs share a Root Complex with which GPUs:

```
root@rs-gpu-pool-3xnkkk:~# lspci -tv
-+-[0000:00]-+-00.0  Intel Corporation 82G33/G31/P35/P31 Express DRAM Controller
 |           +-... (QEMU virtual devices / ICH9 southbridge devices, not relevant here, omitted)
 +-[0000:80]---00.0-[81-86]----00.0-[82-86]--+-00.0-[83-85]----00.0-[84-85]----00.0-[85]----00.0  NVIDIA Corporation GB110 [B300 SXM6 AC]   # GPU0
 |                                           \-01.0-[86]--+-00.0  Mellanox ConnectX Family mlx5Gen Virtual Function   # mlx5_0
 |                                                        \-00.1  Mellanox ConnectX Family mlx5Gen Virtual Function   # mlx5_1
 +-[0000:88]---00.0-[89-8e]----00.0-[8a-8e]--+-00.0-[8b-8d]----00.0-[8c-8d]----00.0-[8d]----00.0  NVIDIA Corporation GB110 [B300 SXM6 AC]   # GPU1
 |                                           \-01.0-[8e]--+-00.0  Mellanox ConnectX Family mlx5Gen Virtual Function   # mlx5_2
 |                                                        \-00.1  Mellanox ConnectX Family mlx5Gen Virtual Function   # mlx5_3
 ... (GPU2~GPU7 and mlx5_4~mlx5_15 have exactly the same structure, mapped in order, omitted here)
 \-[0000:c0]---00.0-[c1-c3]----00.0-[c2-c3]----00.0-[c3]--+-00.0  Mellanox MT2910 Family [ConnectX-7]
                                                          +-00.1  Mellanox MT2910 Family [ConnectX-7]
                                                          +-00.2  Mellanox MT2910 Family [ConnectX-7]
                                                          \-00.3  Mellanox MT2910 Family [ConnectX-7]   # 4 extra ports, not sharing a Root Complex with any GPU
```

The tree structure is unambiguous: each GPU and its "nearest" two NICs branch off the **same PCIe Root Complex** (e.g. `0000:80`) — one branch goes through a few levels of PCIe Switch to reach the GPU, the other connects directly to the two VFs (NICs) carved out of the same physical ConnectX card. Summarized as a table:

| GPU | GPU PCI Bus | Same-root (closest) NIC | NIC PCI Bus |
|---|---|---|---|
| GPU0 | 85:00.0 | mlx5_0 / mlx5_1 | 86:00.0 / 86:00.1 |
| GPU1 | 8D:00.0 | mlx5_2 / mlx5_3 | 8e:00.0 / 8e:00.1 |
| GPU2 | 95:00.0 | mlx5_4 / mlx5_5 | 96:00.0 / 96:00.1 |
| GPU3 | 9D:00.0 | mlx5_6 / mlx5_7 | 9e:00.0 / 9e:00.1 |
| GPU4 | A5:00.0 | mlx5_8 / mlx5_9 | a6:00.0 / a6:00.1 |
| GPU5 | AD:00.0 | mlx5_10 / mlx5_11 | ae:00.0 / ae:00.1 |
| GPU6 | B5:00.0 | mlx5_12 / mlx5_13 | b6:00.0 / b6:00.1 |
| GPU7 | BD:00.0 | mlx5_14 / mlx5_15 | be:00.0 / be:00.1 |

Pattern: `GPU[n]` maps to `mlx5_[2n]` and `mlx5_[2n+1]`, which lines up exactly with the distance matrix `nvidia-smi topo -m` reports (same group = `PXB`, different group = `SYS`).

**This is the physical basis of the "rail-optimized" topology: every GPU is paired with the two RDMA NICs physically nearest to it**, so data movement only has to cross a local PCIe Switch, without going across NUMA nodes or through the CPU host bridge. **The key implication: in every experiment in this document, `mlx5_0`/`mlx5_1` (`fabric0`/`fabric1`) are only physically co-rooted with GPU0** — they don't share a Root Complex with GPU1~7. This fact becomes important later, in the bottleneck analysis of the AllReduce test in Section 6 — if you're planning a multi-GPU test, work out this topology first before interpreting the results.

### 1.6 Pre-Deployment Baseline: Checking the NIC State on the Hosts

Deploying the CNI (Sections 1.3/1.4) doesn't immediately move any NICs — that only happens once a Pod actually references a NAD. So right now (before any Pod is deployed), let's record the baseline state on the hosts for comparison later:

```
$ ssh root@138.68.59.208 "ip -br link show | grep -i fabric"
fabric0 ~ fabric14  UP  (fabric15 missing)
vrf-fabric0 ~ vrf-fabric14  UP

$ ssh root@134.209.13.101 "ip -br link show | grep -i fabric"
fabric0 ~ fabric15  UP  (all 16 NICs present)
vrf-fabric0 ~ vrf-fabric15  UP
```

**Interface / VRF / routing relationships**: on the host, `fabric0`/`fabric1` are each enslaved to their own independent `vrf-fabric0`/`vrf-fabric1` (each with its own separate routing table), and their IPs are:

| Node | fabric0 (rail0) | fabric1 (rail1) |
|---|---|---|
| rs-gpu-pool-3xnkkk (134.209.13.101) | `fd02:0:0:58::1` | `fd02:0:0:59::1` |
| rs-gpu-pool-3xnkk5 (138.68.59.208) | `fd02:0:0:178::1` | `fd02:0:0:179::1` |

Each rail's VRF routing table has, besides the local `/64` connected route, a default route `fd02::/32 via <this rail's gateway> onlink` (the gateway address is that rail's corresponding port address on the leaf switch — for example `fabric0`'s gateway is `fd02:0:0:58::2`); any `fd02::/32` traffic not within the local `/64` subnet has to go to this gateway (the leaf switch) first, which then forwards it. **The two rails' gateways are independent, different interfaces** — there's no local routing shortcut between `vrf-fabric0` and `vrf-fabric1` on the same host, even though fabric0 and fabric1 are physically plugged into the same server; cross-rail communication still has to go out over the wire to the leaf switch and back.

**Mini-experiment: using ping's TTL to verify the "leaf loopback" behavior** (ping sends with TTL=64 by default):

```
# Cross-node, same rail: node2(134.209.13.101).fabric0 -> node1(138.68.59.208).fabric0
$ ssh root@134.209.13.101 "ping -6 -I fabric0 -c 4 fd02:0:0:178::1"
64 bytes from fd02:0:0:178::1: icmp_seq=1 ttl=62 ...   # 2 fewer hops (e.g. leaf+spine, or two levels of leaf)
rtt avg 0.129 ms

# Cross-node, same rail: node2.fabric1 -> node1.fabric1
$ ssh root@134.209.13.101 "ping -6 -I fabric1 -c 4 fd02:0:0:179::1"
64 bytes from fd02:0:0:179::1: icmp_seq=1 ttl=62 ...   # also 2 fewer hops
rtt avg 0.116 ms

# Same node, cross-rail (on rs-gpu-pool-3xnkkk): fabric0 -> fabric1
$ ssh root@134.209.13.101 "ping -6 -I fabric0 -c 4 fd02:0:0:59::1"
64 bytes from fd02:0:0:59::1: icmp_seq=1 ttl=63 ...    # only 1 fewer hop
rtt avg 0.102 ms

# Same node, cross-rail (on rs-gpu-pool-3xnkk5): fabric0 -> fabric1
$ ssh root@138.68.59.208 "ping -6 -I fabric0 -c 4 fd02:0:0:179::1"
64 bytes from fd02:0:0:179::1: icmp_seq=1 ttl=63 ...   # only 1 fewer hop
rtt avg 0.081 ms
```

All four sets show 0% packet loss. **The TTL difference confirms the "leaf loopback" hypothesis**: cross-node, same-rail traffic (2 fewer hops in TTL) crosses 2 routing hops; whereas fabric0 pinging fabric1 on the same machine (only 1 fewer hop in TTL) crosses only 1 routing hop — the packet leaves `fabric0` over the wire to the leaf switch, and the switch forwards it straight back into the `fabric1` port on that same machine, without needing to go up to the second tier (spine). Even though they're physically on the same server, the two rails are treated at the network layer as two completely independent segments that can only reach each other through the external switch.

**Reverse verification: without binding an egress device via `-I`, pinging a remote address directly fails**, which further confirms that these routes really only exist in the VRF table for the corresponding rail, and not in the default `main` table:

```
$ ssh root@134.209.13.101 "ping -6 -c 4 fd02:0:0:178::1"     # without -I fabric0
ping: connect: Network is unreachable
```

Reason: the `1000: from all lookup [l3mdev-table]` rule in `ip -6 rule show` needs **an already-determined egress device** in order to locate the VRF table it belongs to; without `-I` (without binding a device), the kernel falls back to the default `32766: from all lookup main` rule, and the `main` table simply has no route for `fd02::/32` (it only exists in `vrf-fabric0`/`vrf-fabric1`'s own tables), hence the immediate `Network is unreachable`. **Remember this conclusion — the `ib_write_bw` test in Section 4 will hit the same trap.**

**RDMA device-layer baseline**: everything above is at the network layer (netdevice/IP/routing). Let's also use a few RDMA/verbs-specific tools to confirm the state of `fabric0`/`fabric1` (`mlx5_0`/`mlx5_1`) versus the other 14 NICs at the device layer (using `rs-gpu-pool-3xnkkk` as the example):

```
$ rdma link | grep -i fabric
link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev fabric0
link mlx5_1/1 state ACTIVE physical_state LINK_UP netdev fabric1
... (mlx5_2~mlx5_15 listed in turn, all ACTIVE/LINK_UP, netdev fabric2~fabric15 respectively)

$ ibv_devices
    device          	   node GUID
    ------          	----------------
    mlx5_0          	4ebb47fffe67f96a
    mlx5_1          	4ebb47fffe67f96b
    ... (mlx5_2~mlx5_15, plus 4 extra ports mlx5_16~mlx5_19 — see the ibdev2netdev note below)

$ ibv_devinfo -v | grep -E "hca_id|GID\["
hca_id:	mlx5_0
			GID[  0]:		fe80:0000:0000:0000:4cbb:47ff:fe67:f96a, RoCE v1
			GID[  1]:		fe80::4cbb:47ff:fe67:f96a, RoCE v2
			GID[  2]:		fd02:0000:0000:0058:0000:0000:0000:0001, RoCE v1
			GID[  3]:		fd02:0:0:58::1, RoCE v2      # the GID index 3 used throughout this document's experiments
hca_id:	mlx5_1
			GID[  0]:		fe80:0000:0000:0000:4cbb:47ff:fe67:f96b, RoCE v1
			GID[  1]:		fe80::4cbb:47ff:fe67:f96b, RoCE v2
			GID[  2]:		fd02:0000:0000:0059:0000:0000:0000:0001, RoCE v1
			GID[  3]:		fd02:0:0:59::1, RoCE v2

$ show_gids | grep -iE "fabric0|fabric1\b"
mlx5_0	1	0	fe80:...:f96a			v1	fabric0
mlx5_0	1	1	fe80:...:f96a			v2	fabric0
mlx5_0	1	2	fd02:0000:0000:0058:...			v1	fabric0
mlx5_0	1	3	fd02:0000:0000:0058:...			v2	fabric0
mlx5_1	1	0	fe80:...:f96b			v1	fabric1
... (fabric1 has 2 v1/v2 entries each, same structure)

$ ibdev2netdev
mlx5_0 port 1 ==> fabric0 (Up)
mlx5_1 port 1 ==> fabric1 (Up)
mlx5_2 port 1 ==> fabric2 (Up)
... (mlx5_3~mlx5_15 are all fabricN (Up))
mlx5_16 port 1 ==> ibp195s0f0 (Down)
mlx5_17 port 1 ==> ibp195s0f1 (Down)
mlx5_18 port 1 ==> ibp195s0f2 (Down)
mlx5_19 port 1 ==> ibp195s0f3 (Down)
```

This `ibdev2netdev` output is exactly what confirms the `NCCL_IB_HCA` prefix-matching trap we hit while debugging the MPI Operator in Section 8.4: `mlx5_16`~`mlx5_19`, these 4 extra ports, are **already Down and not associated with any `fabricN` netdev** (they correspond to `ibp195s0f0~3` in the `ibdev2netdev` output) — they're an entirely different animal from `mlx5_0`~`mlx5_15`. That's exactly why, back when NCCL mistakenly matched them into its device list, `ibv_create_ah` immediately failed with `No such device`.

---

<a id="sec-2"></a>
## 2. Deploy and Verify the Test Workload
[↑ Back to TOC](#toc)

### 2.1 Workload Structure (`test-workload.yaml`)

Full contents in Appendix 11.3. This is the test workload shared by every experiment in this document:

- A `StatefulSet` named `8-gpu-2-fabric-pod`, with 2 replicas, targeting one Pod on each of the two GPU nodes.
- The Pod annotation `k8s.v1.cni.cncf.io/networks` attaches `roce-net-fabric0` (named `pod-fabric0`) and `roce-net-fabric1` (`pod-fabric1`).
- `nodeSelector: doks.digitalocean.com/gpu-brand: nvidia` plus a toleration for the `nvidia.com/gpu:NoSchedule` taint, to make sure it schedules onto GPU nodes.
- Image `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel` (PyTorch 2.13.0+cu130, NCCL 2.29.7); on startup it installs network diagnostic tools (`iproute2`/`ibverbs-utils`/`infiniband-diags`, etc.) and then runs `sleep infinity` so we can manually exec in for debugging.
- Resources: per Pod, `nvidia.com/gpu: 8`, `rdma/fabric0: 100`, `rdma/fabric1: 100`; a 128Gi memory-backed `/dev/shm` (for NCCL), plus the host's HuggingFace cache directory mounted in.

### 2.2 Deploy

```bash
kubectl apply -f test-workload.yaml
```

Expected output:

```
$ kubectl apply -f test-workload.yaml
statefulset.apps/8-gpu-2-fabric-pod created

$ kubectl get pods -l app=8-gpu-2-fabric-pod -o wide
NAME                   READY   STATUS    IP             NODE
8-gpu-2-fabric-pod-0   1/1     Running   10.121.3.77    rs-gpu-pool-3xnkkk  (134.209.13.101)
8-gpu-2-fabric-pod-1   1/1     Running   10.121.2.170   rs-gpu-pool-3xnkk5  (138.68.59.208)
```

### 2.3 Verify: In-Pod State

```
$ kubectl exec 8-gpu-2-fabric-pod-0 -- ip -br link
pod-fabric0      UP  4e:bb:47:67:f9:6a  <BROADCAST,MULTICAST,UP,LOWER_UP>
pod-fabric1      UP  4e:bb:47:67:f9:6b  <BROADCAST,MULTICAST,UP,LOWER_UP>
eth0@if63        UP  ...

$ kubectl exec 8-gpu-2-fabric-pod-0 -- nvidia-smi -L
GPU 0~7: NVIDIA B300 SXM6 AC  (all 8 visible)
```

**Checkpoint**: the MAC addresses of `pod-fabric0`/`pod-fabric1` should be identical to the original `fabric0`/`fabric1` on the corresponding host — confirming that this is the same physical NIC moved into the Pod's network namespace (not a virtual device), and that `ip-preserving-host-device`'s passthrough behavior is working.

The fabric addresses inside the Pods (used repeatedly in later experiments — worth writing down):

| Pod | pod-fabric0 (rail0/mlx5_0) | pod-fabric1 (rail1/mlx5_1) | eth0 (Pod default network) |
|---|---|---|---|
| `8-gpu-2-fabric-pod-0` (node `rs-gpu-pool-3xnkkk`) | `fd02:0:0:58::1` | `fd02:0:0:59::1` | `10.121.3.77` |
| `8-gpu-2-fabric-pod-1` (node `rs-gpu-pool-3xnkk5`) | `fd02:0:0:178::1` | `fd02:0:0:179::1` | `10.121.2.170` |

RDMA device mapping: `mlx5_0` ↔ `pod-fabric0` (rail0), `mlx5_1` ↔ `pod-fabric1` (rail1). GID[3] (RoCEv2) matches the addresses above — you can verify this yourself with `kubectl exec ... -- ibv_devinfo -v | grep -E "hca_id|GID\["`.

**VRF and routing-table structure**: exactly the same behavior we saw on the host side in Section 1.6 — inside the Pod, `pod-fabric0`/`pod-fabric1` are each enslaved to their own independent VRF (`vrf100`/`vrf101`, mapping to routing tables 100/101):

```
$ kubectl exec 8-gpu-2-fabric-pod-0 -- ip -6 route show           # default main table: nearly empty
fe80::/64 dev eth0 proto kernel metric 256 pref medium

$ kubectl exec 8-gpu-2-fabric-pod-0 -- ip -6 route show vrf vrf100  # the VRF routing table for pod-fabric0
fd02:0:0:58::/64 dev pod-fabric0 proto kernel metric 256 pref medium
fd02::/32 via fd02:0:0:58::2 dev pod-fabric0 proto static metric 1024 onlink pref medium   # cross-node route, only in this VRF table
fe80::/64 dev pod-fabric0 proto kernel metric 256 pref medium
multicast ff00::/8 dev pod-fabric0 proto kernel metric 256 pref medium

$ kubectl exec 8-gpu-2-fabric-pod-0 -- ip vrf list
Name              Table
-----------------------
vrf100             100
vrf101             101

$ kubectl exec 8-gpu-2-fabric-pod-0 -- ip -d link show pod-fabric0
4: pod-fabric0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 4200 ... master vrf100 state UP ...
    vrf_slave table 100 ...
    alias fabric0
```

**Remember this conclusion**: the cross-node route only exists in the corresponding VRF routing table (100/101), not in the default `main` routing table. Section 4's `ib_write_bw` test will trip over this exact thing if you don't already know it.

### 2.4 After Deployment: Are the Host-Side NIC and Routes Removed?

```
$ ssh root@134.209.13.101 "ip -br link show | grep -iE 'fabric0|fabric1\b'"   # the node hosting pod-0
vrf-fabric0 / vrf-fabric1  UP   (fabric0/fabric1 themselves are no longer in the link list)

$ ssh root@138.68.59.208  "ip -br link show | grep -iE 'fabric0|fabric1\b'"   # the node hosting pod-1
vrf-fabric0 / vrf-fabric1  UP   (same as above)

# The routing tables are cleared out too:
$ ssh root@138.68.59.208 "ip route show | grep -i fabric"        # main routing table: no output
$ ssh root@138.68.59.208 "ip route show vrf vrf-fabric0"          # the VRF's own routing table: no output
$ ssh root@138.68.59.208 "ip route show vrf vrf-fabric1"          # no output
(same result on 134.209.13.101)
```

**Conclusion: on both hosts, the original `fabric0`/`fabric1` physical NIC interfaces and their routes have been completely removed from the host's default network namespace, leaving only the corresponding `vrf-fabric0`/`vrf-fabric1` (the VRF master device itself hasn't been removed, but it's now an empty shell — no slave device, no routes).** Reason: routes like `fd02:0:0:xx::/64 dev fabric0` are attached to the physical device `fabric0`; once the CNI plugin moves the whole device into the Pod's network namespace, the routes move along with it. The rest of the NICs not used by any Pod (`fabric2`~`fabric14/15`) stay on the host, unaffected — matching the expected behavior of the `ip-preserving-host-device` plugin.

### 2.5 Bonus Experiment: The Host-Side "Disappearance" Is Only at the Network Layer — the GID at the RDMA verbs Layer Is Still Shared Globally

The conclusion above (2.4) can easily give the impression that `mlx5_0`/`mlx5_1` have "completely stopped existing" on the host. Digging further, using three different tools on the host to query these two NICs' GIDs produces inconsistent results:

| Tool | Result | Reason |
|---|---|---|
| Raw read of `/sys/class/infiniband/mlx5_0/ports/1/gids/*` | all zeros / error | logic is "check if there's an associated netdevice first" — `mlx5_0`/`mlx5_1`'s netdevice has already been moved into the Pod, so the host can't see it |
| `show_gids` | doesn't list `mlx5_0`/`mlx5_1` at all | same reason — it just skips NICs with no associated netdevice |
| `ibv_devinfo -v` | **non-empty**, and matches exactly what you see from inside the Pod (e.g. `fd02:0:0:58::1`) | goes through the verbs API to query the hardware's global GID table directly, without caring about the current network namespace |

Root cause: `rdma system show` shows this machine is in `netns shared` mode, meaning the RDMA subsystem in the kernel **is not isolated per network namespace** — there's a single global instance, which is why `ibv_devinfo -v` can bypass the network-namespace boundary and query the actually-effective GID straight from the hardware table.

**Conclusion (clarifying "can the host actually use this NIC")**:
1. The host being able to "see the GID" is a side effect of the Pod's own usage being exposed via kernel-level sharing — it's not something the host has natively; if the Pod never assigned an IP to the NIC, this GID entry simply wouldn't exist.
2. The host genuinely cannot use these two NICs for network communication — RDMA/RoCE communication needs a corresponding netdevice/IP/route in the local network namespace, and all of that has been fully transferred into the Pod; the host's `ip link` shows no `fabric0`/`fabric1` (consistent with the conclusion in 2.4).
3. There are two different isolation layers at play: the **network layer** (netdevice/IP/routes) is exclusively owned by the Pod — the host simply doesn't have access; the **RDMA device layer** (verbs/uverbs, the hardware GID table) is globally shared under `netns shared` mode — the host can see it, but shouldn't and can't safely bypass this to actually use it. The proper, safe conclusion is: **these two NICs' usage rights belong entirely to the Pod — the host can see them but can't use them**.
4. Extra note: `mlx5_0`/`mlx5_1` are SR-IOV VFs carved out of a physical function (PF, `c3:00.x`, ConnectX-7), assigned to the Pod as a unit — they aren't standalone physical cards.

(All the experiments in Sections 3~6 below run on this already-deployed workload. Once all of those experiments are done, Section 7 will delete it again, to verify that the NICs can be correctly handed back to the host.)

---

<a id="sec-3"></a>
## 3. Experiment 1: Connectivity Test (Same-Rail / Cross-Rail Ping)
[↑ Back to TOC](#toc)

Background: the GPU fabric follows a **rail-optimized** design (`fabric0`=rail0/`mlx5_0`, `fabric1`=rail1/`mlx5_1`), and this design is meant to allow cross-rail communication — we need to verify that "mismatched rails" between different nodes (e.g. pod0's rail0 talking to pod1's rail1) also work correctly, not just same-rail-to-same-rail.

Once both Pods are Running, run 4 test cases using the fabric addresses recorded in Section 2.3:

```bash
# 1. Same rail: pod0 fabric0(rail0) -> pod1 fabric0(rail0)
kubectl exec 8-gpu-2-fabric-pod-0 -- ping -6 -I pod-fabric0 -c 4 fd02:0:0:178::1

# 2. Same rail: pod0 fabric1(rail1) -> pod1 fabric1(rail1)
kubectl exec 8-gpu-2-fabric-pod-0 -- ping -6 -I pod-fabric1 -c 4 fd02:0:0:179::1

# 3. Cross rail: pod0 fabric0(rail0) -> pod1 fabric1(rail1)
kubectl exec 8-gpu-2-fabric-pod-0 -- ping -6 -I pod-fabric0 -c 4 fd02:0:0:179::1

# 4. Cross rail: pod0 fabric1(rail1) -> pod1 fabric0(rail0)
kubectl exec 8-gpu-2-fabric-pod-0 -- ping -6 -I pod-fabric1 -c 4 fd02:0:0:178::1
```

Expected results (all 4 groups at 0% packet loss, latency consistent at ~0.1ms):

```
# 1
4 packets transmitted, 4 received, 0% packet loss, rtt avg 0.110 ms
# 2
4 packets transmitted, 4 received, 0% packet loss, rtt avg 0.101 ms
# 3
4 packets transmitted, 4 received, 0% packet loss, rtt avg 0.097 ms
# 4
4 packets transmitted, 4 received, 0% packet loss, rtt avg 0.101 ms
```

**Conclusion: cross-rail L3 connectivity works perfectly — 0% packet loss, latency identical to same-rail (~0.1ms)** — the underlying switching fabric correctly routes/forwards cross-rail traffic, and there's no network isolation issue between the two rails.

---

<a id="sec-4"></a>
## 4. Experiment 2: RDMA Bandwidth Test (`ib_write_bw`)
[↑ Back to TOC](#toc)

### 4.1 The Trap: Using the Fabric NIC's IPv6 Address for OOB Directly

The natural first instinct is to just run it against the fabric address directly:

```bash
kubectl exec 8-gpu-2-fabric-pod-0 -- ib_write_bw -d mlx5_0 --ipv6-addr fd02:0:0:178::1
```

Which gets you:

```
Network is unreachable / Couldn't connect
```

The root cause is the same story as Sections 1.6/2.3: `pod-fabric0`/`pod-fabric1` are enslaved to their own individual VRFs, and the cross-node route for `fd02::/32` only exists in the corresponding VRF's routing table — not in the container's default `main` routing table. `ping -I <iface>` succeeds because it explicitly binds an egress device, sidestepping the routing ambiguity; `ib_write_bw` has no equivalent "bind to device" option. Every workaround we tried failed: `ip vrf exec` (the container's cgroup filesystem is read-only), `ip -6 rule add` (missing `NET_ADMIN`), `ib_write_bw -R` rdma_cm mode (also goes through the kernel routing table, equally constrained).

### 4.2 The Fix: Route OOB Over eth0, Keep the RDMA Data Plane on the Fabric NIC's GID

`ib_write_bw`'s connection target address is only used to exchange QP metadata over an **out-of-band (OOB) control channel** — the actual RDMA data movement happens over the GID of the device specified with `-d` (RoCEv2 GRH addressing); the two can be decoupled. Route the OOB traffic over the Pod's default network `eth0` (which has a normal `main`-table default route), while keeping the RDMA data plane on the fabric NIC's GID, and you sidestep the limitation from 4.1:

```bash
# server (pod1)
kubectl exec 8-gpu-2-fabric-pod-1 -- ib_write_bw -d <mlx5_x> -x 3

# client (pod0), target is pod1's eth0 IP (OOB)
kubectl exec 8-gpu-2-fabric-pod-0 -- ib_write_bw -d <mlx5_y> -x 3 10.121.2.170
```

Using this pattern, we ran 4 rail combinations (RDMA_Write BW, 65536B, 5000 iterations, single QP):

| # | client (pod0) | server (pod1) | type | BW peak (MB/s) | BW avg (MB/s) | MsgRate (Mpps) |
|---|---|---|---|---|---|---|
| 1 | mlx5_0 (rail0) | mlx5_0 (rail0) | same rail | 43834.02 | 6789.66 | 0.1086 |
| 2 | mlx5_1 (rail1) | mlx5_0 (rail0) | **cross rail** | 44117.65 | 3913.35 | 0.0626 |
| 3 | mlx5_0 (rail0) | mlx5_1 (rail1) | **cross rail** | 44117.65 | 19508.49 | 0.3121 |
| 4 | mlx5_1 (rail1) | mlx5_1 (rail1) | same rail | 43885.31 | 15975.77 | 0.2556 |

Sample output (#1, same-rail baseline):

```
$ kubectl exec 8-gpu-2-fabric-pod-1 -- ib_write_bw -d mlx5_0 -x 3        # server, running in the background
$ kubectl exec 8-gpu-2-fabric-pod-0 -- ib_write_bw -d mlx5_0 -x 3 10.121.2.170

 Device : mlx5_0   Link type : Ethernet   GID index : 3
 local address:  GID: 253:02:00:00:00:00:00:88:00:00:00:00:00:00:00:01
 remote address: GID: 253:02:00:00:00:00:01:120:00:00:00:00:00:00:00:01
 #bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
 65536      5000             43834.02            6789.66		   0.108635
```

**Conclusion: all 4 rail combinations (same-rail x2, cross-rail x2) successfully established RDMA_Write connections and transferred data, proving that RDMA on the data plane works normally in cross-rail scenarios too.** The BW average numbers vary quite a bit across runs (3913~19508 MB/s), which is because the test used default parameters (single QP, a single 65536B message size, only 5000 iterations, without `-a`/`-D` for a full sweep) — the goal of this round was to verify connectivity/functionality, not to rigorously measure peak bandwidth, so treat the numbers as reference only.

---

<a id="sec-5"></a>
## 5. Experiment 3: PyTorch GPU-to-GPU RDMA (Single NIC)
[↑ Back to TOC](#toc)

Goal: write a simple script that does GPU-to-GPU communication between 2 Pods using only 1 RDMA NIC, and measure 1/2/4/8 GB transfers.

### 5.1 Approach

`torch.distributed` + NCCL backend, `world_size=2`, rank0/rank1 each corresponding to one Pod; for each size, run 3 warmup iterations of `dist.send`/`dist.recv` first, then 5 real iterations to compute average latency and bandwidth.

- OOB (process-group rendezvous) goes over the Pod's default network `eth0` (`NCCL_SOCKET_IFNAME=eth0`); `MASTER_ADDR` must be set to **rank0**'s Pod's eth0 IP (c10d's TCPStore server is created by rank0).
- The GPU-to-GPU data plane is forced onto a single RDMA NIC: `NCCL_IB_HCA=mlx5_0`, `NCCL_IB_GID_INDEX=3` (the RoCEv2 GID).

Full code in Appendix 11.4 (`gpu_rdma_bw_test.py`).

### 5.2 Run Commands

```bash
# rank1 (Pod 8-gpu-2-fabric-pod-1, not the master)
kubectl exec 8-gpu-2-fabric-pod-1 -- bash -lc "\
  MASTER_ADDR=10.121.3.77 MASTER_PORT=29500 RANK=1 WORLD_SIZE=2 \
  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
  python3 -u /root/gpu_rdma_bw_test.py > /root/rank1.log 2>&1"

# rank0 (Pod 8-gpu-2-fabric-pod-0, the master — MASTER_ADDR must be its own eth0 IP)
kubectl exec 8-gpu-2-fabric-pod-0 -- bash -lc "\
  MASTER_ADDR=10.121.3.77 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 \
  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
  python3 -u /root/gpu_rdma_bw_test.py > /root/rank0.log 2>&1"
```

> ⚠️ **Pitfall warning**: when `dist.init_process_group` uses the `env://` method, the TCPStore server is created and listened on by `RANK=0` — `MASTER_ADDR` has to be an address reachable at rank0's Pod. If you accidentally set it to rank1's IP instead, both processes will try to connect to an address nobody is listening on, and both sides will hang in `init_process_group` with no error and no timeout (you have to manually kill it to even notice). A useful technique for debugging this kind of hang: add staged `print(..., flush=True)` calls in the script, run with unbuffered `python3 -u`, and redirect output to a file — that way you can immediately tell which step it's stuck on (rendezvous / NCCL init / send/recv).

### 5.3 Key Log Lines (Confirming It's Using a Single RDMA NIC + GPUDirect RDMA)

```
NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth0
NCCL INFO Bootstrap: Using eth0:10.121.3.77<0>              # OOB over eth0
NCCL INFO NCCL_IB_HCA set to mlx5_0                          # data plane restricted to a single NIC
NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [RO]; OOB eth0:10.121.3.77<0>
NCCL INFO DMA-BUF is available on GPU device 0
NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0/GDRDMA   # GPUDirect RDMA, over IB/0 = mlx5_0
NCCL INFO Connected all rings, use ring PXN 0 GDR 1
```

### 5.4 Test Results

| Data size | Avg latency | BW average | BW (Gb/s) |
|---|---|---|---|
| 1 GB | 22.84 ms | 47.00 GB/s | 376.01 |
| 2 GB | 44.69 ms | 48.05 GB/s | 384.43 |
| 4 GB | 89.63 ms | 47.92 GB/s | 383.36 |
| 8 GB | 177.32 ms | 48.44 GB/s | 387.55 |

**Conclusion: a single RDMA NIC (`mlx5_0`/`fabric0`/rail0) delivers a stable ~47-48 GB/s (roughly 380 Gb/s) of GPU-to-GPU bandwidth across all 4 data sizes, barely changing as the size grows — meaning the link's bandwidth ceiling is already saturated, with no noticeable startup overhead or jitter as message size increases.**

---

<a id="sec-6"></a>
## 6. Experiment 4: PyTorch AllReduce (Dual NIC, 8 GPU/Pod)
[↑ Back to TOC](#toc)

Goal: run AllReduce between 2 Pods using both `mlx5_0`+`mlx5_1` RDMA NICs, with each Pod using its full 8 GPUs, measuring 1/2/4/8 GB.

### 6.1 Approach and Run Method

- `world_size=16` (2 Pods × 8 GPUs), using `torch.distributed` + NCCL backend; each rank runs 3 warmup iterations followed by 5 real iterations of `dist.all_reduce(SUM)` on a 1/2/4/8 GB float32 tensor, and following the nccl-tests convention we report both `algbw` (=size/time) and `busbw` (=algbw×2×(n-1)/n, the effective bus bandwidth for a 16-GPU ring all-reduce).
- Launched via `torchrun` (`--nnodes=2 --nproc_per_node=8`); each Pod runs its own script for its `node_rank`, with `MASTER_ADDR` fixed to node_rank=0 (pod0)'s eth0 IP `10.121.3.77`.
- Key environment variables: `NCCL_SOCKET_IFNAME=eth0` (OOB), `NCCL_IB_HCA=mlx5_0,mlx5_1` (restrict the data plane to these two aligned NICs), `NCCL_IB_GID_INDEX=3`, `NCCL_CROSS_NIC=0|1` (the main comparison point in this section).

Full code in Appendix 11.5 (`allreduce_bw_test.py` + `run_allreduce_pod0.sh` + `run_allreduce_pod1.sh`).

### 6.2 Results: `NCCL_CROSS_NIC=0` (rail-optimized default, strict rail alignment)

```
NCCL INFO NCCL_IB_HCA set to mlx5_0,mlx5_1
NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [1]mlx5_1:1/RoCE [RO]; OOB eth0:10.121.3.77<0>
```

| Data size | Avg latency | algbw | busbw | busbw (Gb/s) |
|---|---|---|---|---|
| 1 GB | 424.55 ms | 2.53 GB/s | 4.74 GB/s | 37.94 |
| 2 GB | 833.67 ms | 2.58 GB/s | 4.83 GB/s | 38.64 |
| 4 GB | 1649.85 ms | 2.60 GB/s | 4.88 GB/s | 39.05 |
| 8 GB | 3297.98 ms | 2.60 GB/s | 4.88 GB/s | 39.07 |

### 6.3 Results: `NCCL_CROSS_NIC=1` (allows a channel to freely switch NICs — comparison run)

| Data size | Avg latency | algbw | busbw | busbw (Gb/s) |
|---|---|---|---|---|
| 1 GB | 418.29 ms | 2.57 GB/s | 4.81 GB/s | 38.51 |
| 2 GB | 826.20 ms | 2.60 GB/s | 4.87 GB/s | 38.99 |
| 4 GB | 1640.47 ms | 2.62 GB/s | 4.91 GB/s | 39.27 |
| 8 GB | 3277.36 ms | 2.62 GB/s | 4.91 GB/s | 39.31 |

**`CROSS_NIC=0` and `CROSS_NIC=1` produce almost identical results (busbw both around ~4.8-4.9 GB/s), meaning this setting isn't the bottleneck in this scenario.**

### 6.4 Key Finding: The Bottleneck Is the Topology (8 GPUs Sharing 2 NICs, Both Owned by GPU0), Not NCCL_CROSS_NIC

Looking at the NCCL logs, cross-node network channels only ever appear between **rank0↔rank8 and rank1↔rank8/rank9**, i.e. "gateway GPU" pairs (for example `Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/2/GDRDMA`); the other 6 GPUs' cross-node traffic all funnels first through `P2P/CUMEM` (NVLink) to converge on this pair of "gateway GPUs" before going out over the RDMA NIC.

**Root cause (echoing the PCI topology in Section 1.5)**: this hardware follows a strictly rail-aligned design — `GPU[n]` only shares a PCIe Root Complex (`PXB` distance) with `mlx5_[2n]`/`mlx5_[2n+1]`; reaching any other NIC requires crossing NUMA nodes at `SYS` distance. In other words, **`mlx5_0`/`mlx5_1` are physically co-rooted only with GPU0** — they share no Root Complex with GPU1~7 at all. So in this test, GPU0 is the only GPU with a truly "zero-hop" direct connection to these two NICs; the other 7 GPUs first have to hop over NVLink into GPU0's PCIe domain before GPU0 forwards the data out — the standard rail-optimized (1 GPU : 1 pair of dedicated NICs) topology, in a scenario where only `mlx5_0`/`mlx5_1` are serving all 8 GPUs, degenerates into a structure of "7 GPUs forwarding over NVLink + 1 GPU monopolizing the outbound bandwidth."

This explains why this section's AllReduce busbw (~4.9 GB/s) is so much lower than Section 5's single-NIC point-to-point test (~48 GB/s): the AllReduce bus bandwidth is bottlenecked by the serial path of "converging over NVLink onto GPU0, then only GPU0 going out over the network" — not by the NIC's own physical bandwidth ceiling — and toggling `NCCL_CROSS_NIC` doesn't change this structural bottleneck at all (that setting only controls whether NCCL is allowed to switch NICs within a single channel; it can't magically give GPU1~7 a physical path directly to a NIC).

**Conclusion:**
1. Both RDMA NICs (`mlx5_0`/`mlx5_1`) really are recognized and registered by NCCL (there's ample log evidence), and the 16-GPU AllReduce functional test passed across all 4 data sizes.
2. But because of this test environment's limitation — "one machine has a bad NIC, so only 2 aligned NICs are available to serve 8 GPUs" (see Section 0.3) — and because these 2 NICs are only co-rooted with GPU0 in the PCIe topology, the AllReduce bus bandwidth is capped at ~4.9 GB/s. **This number does not represent a ceiling on the RDMA hardware or on the rail-optimized approach itself** — it's simply the inevitable result of a "standard 8 GPU:8 NIC (one dedicated NIC pair per GPU) topology degenerating into 8 GPU:2 NIC (with both NICs belonging only to GPU0)."
3. `NCCL_CROSS_NIC=0` vs `1` makes no meaningful difference in this scenario, so there's no need to tune this setting to chase this bottleneck — the real fix is to repair the broken NIC so every GPU can use its own physically co-rooted NIC pair.

If your environment gives every GPU its own co-rooted NIC pair (the standard 1:1 rail mapping), you'd expect busbw significantly higher than the ~4.9 GB/s seen here — worth re-running this section's test on such an environment for comparison.

---

<a id="sec-7"></a>
## 7. Cleaning Up the Environment
[↑ Back to TOC](#toc)

After finishing all the experiments in Sections 2~6, delete the test workload and verify that the NICs get correctly handed back to the host. This step is also a prerequisite for the MPI Operator experiment in Section 8 — `test-workload.yaml`'s 2 Pods each fully occupy 8 GPUs, which conflicts with the MPIJob Worker's resource requests, so we need to clean up first to free the GPUs.

```bash
kubectl delete -f test-workload.yaml
```

```
$ kubectl delete -f test-workload.yaml
statefulset.apps "8-gpu-2-fabric-pod" deleted

$ kubectl get statefulset,pods -l app=8-gpu-2-fabric-pod
No resources found in default namespace.
```

**Verify `fabric0`/`fabric1` recover on both hosts:**

```
# Interface restored, re-attached to the corresponding VRF (compare to Section 2.4: before this, fabric0/fabric1 were completely invisible)
$ ssh root@134.209.13.101 "ip -d link show fabric0"
4: fabric0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 4200 ... master vrf-fabric0 state UP ...
    vrf_slave table 100 ... parentbus pci parentdev 0000:86:00.0
    altname enp134s0f0

# IP address preserved — identical to what the Pod was using (IP-preserving works in both directions: preserved going into the Pod, and preserved coming back to the host)
$ ssh root@134.209.13.101 "ip -6 addr show fabric0"
inet6 fd02:0:0:58::1/64 scope global

# VRF routing table restored, including the cross-node fd02::/32 route
$ ssh root@134.209.13.101 "ip -6 route show vrf vrf-fabric0"
fd02:0:0:58::/64 dev fabric0 proto kernel metric 256 pref medium
fd02::/32 via fd02:0:0:58::2 dev fabric0 proto static metric 1024 onlink pref medium
fe80::/64 dev fabric0 proto kernel metric 256 pref medium
multicast ff00::/8 dev fabric0 proto kernel metric 256 pref medium

# fabric1 on 134.209.13.101, and fabric0/fabric1 on 138.68.59.208, show the exact same symmetric result (IPs restored to
# fd02:0:0:59::1 / fd02:0:0:178::1 / fd02:0:0:179::1 respectively, VRF routing tables also fully restored)
```

**Conclusion: after deleting `test-workload.yaml`, `fabric0`/`fabric1` were cleanly handed back from the Pod's network namespace to the host's default network namespace by the CNI plugin — the interface, the VRF attachment (`master vrf-fabric0`/`vrf-fabric1`), the IP address, and the routes (including the cross-node `fd02::/32` route) were all fully restored to the pre-deployment state from Section 1.6, with no leftovers or corruption. This also confirms that the `ip-preserving-host-device` plugin's "preserve IP on hand-back" design works correctly in both directions (assigning into a Pod, and reclaiming back to the host).**

If you just want to clean up the CNI itself (and don't plan to use this setup again), you can also run `kubectl delete -f network-attachments.yaml` and `kubectl delete -f daemonset.yaml`.

---

<a id="sec-8"></a>
## 8. Experiment 5: MPI Operator AllReduce (nccl-tests, Dual NIC, 8 GPU/Pod)
[↑ Back to TOC](#toc)

After cleaning up `test-workload.yaml` in Section 7 and freeing the GPUs, we run a separate experiment: instead of hand-writing a PyTorch script, we use the community-standard [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator) + the official [`nccl-tests`](https://github.com/NVIDIA/nccl-tests) `all_reduce_perf` binary, running an AllReduce test at the same scale as Section 6 (2 Pods × 8 GPUs, 2 RDMA NICs). This experiment uses Pods created by the MPIJob itself (it doesn't depend on `test-workload.yaml`), and is completely independent of the resources used in earlier sections.

### 8.1 Deploy MPI Operator

```bash
kubectl apply -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.4.0/deploy/v2beta1/mpi-operator.yaml
```

```
namespace/mpi-operator created
customresourcedefinition.apiextensions.k8s.io/mpijobs.kubeflow.org created
serviceaccount/mpi-operator created
...
deployment.apps/mpi-operator created

$ kubectl get pods -n mpi-operator
NAME                            READY   STATUS    RESTARTS   AGE
mpi-operator-65474ddc85-ddfqn   1/1     Running   0          12s
```

### 8.2 MPIJob Definition and Code Review

We started from an existing `allreduce-2-b300x8-2-fabric.yaml` (final version in full in Appendix 11.6). Its core structure:

- `slotsPerWorker: 8` + `Worker.replicas: 2` → 16 slots total (matching 16 GPUs).
- The `Launcher` uses `mpirun` over SSH (port 2222) to remotely start `/opt/nccl-tests/build/all_reduce_perf` on the `Worker`s.
- The `Worker` attaches `roce-net-fabric0@net0` and `roce-net-fabric1@net1` (the same NAD set as earlier sections' `pod-fabric0/1`, just with the interfaces renamed to `net0/net1`), and requests `nvidia.com/gpu: 8` + `rdma/fabric0: 100` + `rdma/fabric1: 100`.
- Uses the image `ghcr.io/coreweave/nccl-tests:13.2.0-devel-ubuntu24.04-nccl2.29.7-1-7112046` (an official pre-built image with `nccl-tests` + MPI already installed).

### 8.3 Deploy, Check Results, and Clean Up

```bash
kubectl apply -f allreduce-2-b300x8-2-fabric.yaml
```

**Check MPIJob / Pod status**:

```
$ kubectl get mpijob
NAME                AGE
mpi-multus-b300x8   19s

$ kubectl get pods -o wide | grep mpi-multus-b300x8
mpi-multus-b300x8-launcher-xxxxx   1/1     Running   0   19s   10.121.0.xx    rs-cpu-pool-3y0p31
mpi-multus-b300x8-worker-0         1/1     Running   0   19s   10.121.3.xx    rs-gpu-pool-3xnkkk
mpi-multus-b300x8-worker-1         1/1     Running   0   19s   10.121.2.xx    rs-gpu-pool-3xnkk5
```

**Check test results**: all `all_reduce_perf` output ends up in the `Launcher` Pod's logs (the `Worker`s are only remotely started via SSH and executed; their output gets aggregated back to the `Launcher` through `mpirun`), so you only need to look at the `Launcher`:

```bash
kubectl logs -f mpi-multus-b300x8-launcher-xxxxx   # follow in real time
kubectl logs mpi-multus-b300x8-launcher-xxxxx      # view the full log after it's done
```

Once the `Launcher` container finishes running `mpirun`, it exits, and the Pod's status becomes `Completed` (`0/1 Completed`) — this is expected and not an error; if instead you see `mpirun detected that one or more processes exited with non-zero status` or `Test NCCL failure`, that's a real failure. The `runPolicy.cleanPodPolicy: Running` setting causes the `Worker` Pods to be automatically cleaned up once the job finishes (they go into `Terminating`), while the `Launcher`'s `Completed` Pod is kept around so you can go back and review the results with `kubectl logs` afterward.

**Clean up the test job**:

```bash
kubectl delete -f allreduce-2-b300x8-2-fabric.yaml
```

Deleting the `MPIJob` resource removes the `Launcher` (even if it's already `Completed`) and all `Worker` Pods together. If you want to run it again (say, after changing `NCCL_CROSS_NIC` or some other parameter), you need to `delete` cleanly first and then `apply` — you can't just `apply` again against the same-named `MPIJob` (`Launcher`/`Worker` are one-shot Pods that run to completion and don't automatically restart or get reused).

### 8.4 Debugging Process: Two Independent Failures, Two Different Root Causes

After deploying, we hit two completely different rounds of failures, one at the MPI layer and one at the NCCL layer:

#### Round 1: MPI/UCX Layer — `uct_iface_open(tcp/net0) failed: Input/output error`

```
[mpi-multus-b300x8-worker-1:996  :0]      ucp_worker.c:1415 UCX  ERROR uct_iface_open(tcp/net0) failed: Input/output error
[mpi-multus-b300x8-worker-1:01000] pml_ucx.c:314  Error: Failed to create UCP worker
...
mpirun noticed that process rank 12 ... exited on signal 11 (Segmentation fault).
```

- **Root cause**: Open MPI defaults to using UCX as its PML, and when the UCX library initializes, it enumerates every network interface in the container, including `net0` (`fabric0`/`mlx5_0`). But as we've repeatedly confirmed in Sections 1/2, this NIC is enslaved to its own independent VRF and isn't configured to work as a normal TCP socket interface — UCX trying to open a TCP transport on it immediately fails with an I/O error, which cascades into a failure to create the entire UCP worker, and ultimately a segfault crash.
- **A dead-end tried during the investigation** (worth recording so we don't repeat it): adding `-mca pml ^ucx` on its own (disabling UCX as the PML) **didn't help** — the error was identical. Reason: this only tells Open MPI "don't pick UCX for messaging"; it doesn't stop the UCX library's own device-probing logic from running during initialization, which isn't gated by the PML-selection switch.
- **The fix that actually worked**: `-x UCX_NET_DEVICES=eth0` — this is UCX's own environment variable, telling it directly "the only thing you're allowed to see is eth0," which stops it from ever touching `net0`/`net1` at the root, instead of relying on an indirect Open MPI switch.

#### Round 2: NCCL Layer — `ibv_create_ah failed: No such device`

Once round 1 was fixed, the MPI layer got past its problem, but NCCL initialization then reported:

```
[5] ibvwrap.cc:205 NCCL WARN Call to ibv_create_ah failed with error No such device
[5] devx_utils.cc:260 NCCL WARN devx_utils.cc:260 Call to wrap_ibv_create_ah(&ah, pd, &attr) failed: 2
mpi-multus-b300x8-worker-0: Test NCCL failure all_reduce.cu:451 'unhandled system error...'
```

This consistently showed up on local ranks 5/6/7 (i.e. GPU5/6/7). The investigation:
- First we suspected `NCCL_NET_DISABLE_INTRA=1` (which forces intra-node communication to avoid the network) might be forcing GPUs that don't have a direct NIC connection (recall the PCI topology in Section 1.5: only GPU0 shares a Root Complex with `mlx5_0`/`mlx5_1`) to try going over the network anyway — but removing this variable and re-running produced **exactly the same error**, ruling it out.
- Turning on `NCCL_DEBUG=INFO` for the detailed log revealed the real clue:

  ```
  NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [1]mlx5_1:1/RoCE [2]mlx5_10:1/RoCE [3]mlx5_11:1/RoCE
                      [4]mlx5_12:1/RoCE [5]mlx5_13:1/RoCE [6]mlx5_14:1/RoCE [7]mlx5_15:1/RoCE
                      [8]mlx5_16:1/IB/SHARP [9]mlx5_17:1/IB/SHARP [10]mlx5_18:1/IB/SHARP [11]mlx5_19:1/IB/SHARP
  ```

- **Root cause**: `NCCL_IB_HCA=mlx5_0,mlx5_1` was meant to restrict things to just these two NICs, but NCCL's default name-matching behavior for this variable is a **prefix match** — `"mlx5_1"` happens to be a string prefix of `mlx5_10`~`mlx5_19` (4 ConnectX-7 management ports that don't share a root with any GPU and that the Pod isn't even authorized to access), and all of them got mistakenly swept into the device list. Some GPUs' channels ended up assigned to these "phantom devices," and naturally failed to find the device when trying to build an address handle (AH).
- **Fix**: force an exact match using `=`, and **add exactly one leading `=` for the whole list** (not one per device name):

  ```
  NCCL_IB_HCA==mlx5_0,mlx5_1
  ```

  Our first attempt wrote it as `=mlx5_0,=mlx5_1` (a `=` on each item), which didn't error, but the log showed every rank only registering `mlx5_0` — `mlx5_1` had been silently dropped, which means the exact-match flag applies "once for the whole list," not "once per item." After switching to a single leading `=`, both NICs correctly showed up together in `NET/IB : Using [0]mlx5_0:1/RoCE [1]mlx5_1:1/RoCE`.

### 8.5 Final Results

After fixing both of the above issues, and keeping `NCCL_NET_DISABLE_INTRA=1` enabled (confirmed unrelated to this failure, and performance was actually better with it back on), the test ran cleanly with `Out of bounds values: 0 OK`.

**`NCCL_CROSS_NIC=0`**:

| size | busbw (GB/s) |
|---|---|
| 8.4 MB | 36.15 |
| 134 MB | 84.88 |
| 2 GB | 89.69 |

`Avg bus bandwidth: 26.8761 GB/s`

**`NCCL_CROSS_NIC=1`** (comparison run):

| size | busbw (GB/s) |
|---|---|
| 8.4 MB | 37.61 |
| 134 MB | 84.68 |
| 2 GB | 89.77 |

`Avg bus bandwidth: 27.0751 GB/s`

Both sets are nearly identical, matching the conclusion from Sections 6.3/6.4's PyTorch AllReduce test: **`NCCL_CROSS_NIC` isn't the bottleneck in this "8 GPUs sharing 2 NICs, both belonging to GPU0" topology** — it makes no difference either way.

Interestingly, the busbw numbers from this run using the official nccl-tests binary (~27-90 GB/s, depending on size) are noticeably higher than the numbers from Section 6's PyTorch script (~4.9 GB/s). A reasonable explanation is the difference in warmup/iteration counts, message-size sweep methodology, and how the official nccl-tests implementation handles multiple sizes — we didn't dig further into the exact cause. **Once again, the caveat from Section 0.3 applies: none of these performance numbers should be treated as authoritative — they only validate functional connectivity** — numbers from different test tools shouldn't be directly compared against each other across tables.

### 8.6 Topology Deep-Dive: How Exactly Do 16 GPUs Form a Ring and Use These 2 NICs?

Sections 1.5/6.4 already inferred from PCI topology that "only GPU0 shares a root with `mlx5_0`/`mlx5_1`." Here, using the full `NCCL_DEBUG=INFO` log, we actually verify how data flows between all 16 GPUs.

**It's not "each Pod forms its own ring that then gets bridged separately" — it's a single ring spanning all 16 GPUs.** The full ring order NCCL prints (`Channel 00/08`):

```
0  7  6  5  4  3  2  1  8  15  14  13  12  11  10  9   (wraps back to 0 after 9, closing the loop)
```

Breaking it down:

**1. The intra-node portions (NVLink, `P2P/CUMEM`)**:
- inside worker-0: `0→7→6→5→4→3→2→1` (7 direct NVLink hops)
- inside worker-1: `8→15→14→13→12→11→10→9` (also 7 direct NVLink hops)

**2. The cross-node portion (RDMA, `GDRDMA`) — the ring only crosses nodes at two points**: edge A `1[1]→8[0]`, edge B `9[1]→0[0]`. Log evidence:

```
worker-0 [1] Channel 00/0 : 1[1] -> 8[0] [send] via NET/SPCX/2(0)/GDRDMA
worker-1 [0] Channel 00/0 : 1[1] -> 8[0] [receive] via NET/SPCX/2/GDRDMA
worker-1 [1] Channel 00/0 : 9[1] -> 0[0] [send] via NET/SPCX/2(8)/GDRDMA
worker-0 [0] Channel 00/0 : 8[0] -> 0[0] [receive] via NET/SPCX/2/GDRDMA
```

**Key detail**: these two cross-node edges look like they're initiated by GPU1 (rank1, rank9), but GPU1 itself has no direct NIC connection — the `(0)` and `(8)` annotations in the log are NCCL's **PXN (PCI×NVLink) forwarding markers**: GPU1 first sends the data over NVLink to GPU0 on the same node (`(0)` = forwarded to local rank0, `(8)` = forwarded to local rank8), and it's **GPU0 that actually holds the physical NIC and sends the data out**. Only the GPU0↔GPU0 pair (rank0↔rank8) is a genuinely direct NIC connection — the only one without a forwarding annotation in the log.

**Conclusion**: 16 GPUs form one ring, with 7 hops over NVLink inside it, and the ring breaks across the node boundary at two points, producing 2 cross-node edges; those two edges appear to be initiated by GPU1 on the surface, but the data on both edges ultimately has to "hitch a ride" over NVLink to the local GPU0 first, and it's GPU0 that actually sends it out over these 2 RDMA NICs (which this particular NCCL build displays merged together as a single virtual device index `NET/SPCX/2`) — this lines up exactly with Section 6.4's conclusion that "GPU0 is the one true network gateway," just with an extra layer of actual log evidence for exactly how the "hitching a ride" happens.

### 8.7 This Section's Pitfalls, Summarized (Folded into Section 9's Table)

| Symptom | What we tried that didn't work | The fix that actually worked |
|---|---|---|
| UCX reports `uct_iface_open(tcp/net0) failed` | `-mca pml ^ucx` | `-x UCX_NET_DEVICES=eth0` |
| NCCL reports `ibv_create_ah failed: No such device` (consistently on GPUs with no direct NIC connection) | removing `NCCL_NET_DISABLE_INTRA=1` | force `NCCL_IB_HCA` to an exact match with a single leading `=`: `NCCL_IB_HCA==mlx5_0,mlx5_1` |

---

<a id="sec-9"></a>
## 9. Pitfall Quick-Reference Table
[↑ Back to TOC](#toc)

| Symptom | Root cause | Fix / Reference section |
|---|---|---|
| `ib_write_bw` inside the Pod reports `Network is unreachable` when used directly with a fabric address | the fabric NIC is enslaved to its own VRF; the cross-node route only exists in the VRF table, not in `main` | Sections 4.1/4.2: route OOB over `eth0`, keep the data plane on `-d mlx5_x`'s GID |
| Pinging a remote address directly on the host/Pod fails when not binding a device via `-I` | same as above — the `l3mdev-table` rule needs an egress device already determined to take effect | Section 1.6 |
| PyTorch `dist.init_process_group` hangs with no error | `MASTER_ADDR` was set to an IP other than rank0's — the TCPStore server never gets created at the expected address | Section 5.2: `MASTER_ADDR` must be rank0's Pod's IP |
| `show_gids`/raw sysfs reads on the host can't find the GID for a NIC already assigned to a Pod, but `ibv_devinfo -v` can | the network layer (netdevice/IP/routes) is exclusively owned by the Pod; the RDMA verbs layer is globally shared under `netns shared` mode | Section 2.5 |
| Multi-GPU AllReduce bandwidth is far lower than the single-NIC point-to-point test | the NIC:GPU ratio isn't the standard 1:1 (2:8 in this case), and these NICs are only physically co-rooted with 1 of the GPUs on the PCIe Root Complex | Sections 1.6/6.4 |
| Pinging between two rails (e.g. fabric0/fabric1) on the same host shows the TTL only dropping by 1 hop | the two rails belong to independent VRFs — there's no local routing shortcut, so traffic must go out over the wire to the leaf switch and back | Section 1.6 |
| MPI Operator's `mpirun` reports `not enough slots` as soon as it's submitted | the `-np` count doesn't match `slotsPerWorker × Worker replicas` (e.g. forgetting to update `-np` after changing a 4-node config down to 2 nodes) | set `-np` equal to `slotsPerWorker × Worker replicas` |
| MPI Operator reports UCX's `uct_iface_open(tcp/net0) failed`, process segfaults | the UCX library's own device probing enumerates VRF-isolated fabric interfaces, and isn't gated by `-mca pml ^ucx` | Section 8.4: `-x UCX_NET_DEVICES=eth0` |
| MPI Operator reports NCCL's `ibv_create_ah failed: No such device`, consistently on certain GPUs | `NCCL_IB_HCA=mlx5_0,mlx5_1`'s prefix matching mistakenly sweeps in `mlx5_10`~`mlx5_19` as well | Section 8.4: force an exact match with a single leading `=`: `NCCL_IB_HCA==mlx5_0,mlx5_1` |

---

<a id="sec-10"></a>
## 10. To-Do / Next Steps
[↑ Back to TOC](#toc)

- **Move to testing on a real production environment, not this simulated one**: every performance number in this document is limited by the simulated-environment constraints described in Section 0.3 (one bad NIC, only 2 aligned rails available), and doesn't carry any reference value — we need to re-run everything on a real rail-optimized cluster to get meaningful numbers.
- **Larger-scale testing**: more nodes, using every RDMA NIC on each machine (instead of just 2 out of 16 like this document does), to validate real throughput under a standard 1:1 (or richer) rail mapping, and to measure cross-node, cross-rail scalability at larger scale.
- **Cover more communication frameworks**, not just NCCL:
  - Mooncake Transfer Engine
  - NIXL / UCX
  - NCCL (already covered in this document — re-test on a real environment later)
- **Application-level testing of distributed inference scenarios**, not just connectivity/bandwidth validation of low-level communication primitives:
  - PD disaggregation (Prefill-Decode disaggregation)
  - KV Cache Sharing
  - Wide EP (expert parallelism)

---

<a id="sec-11"></a>
## 11. Appendix: Full YAML and Script Code
[↑ Back to TOC](#toc)

### 11.1 `daemonset.yaml`

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
 name: ip-preserving-host-device-cni
 namespace: kube-system
 labels:
   app: ip-preserving-host-device-cni
spec:
 selector:
   matchLabels:
     app: ip-preserving-host-device-cni
 updateStrategy:
   type: RollingUpdate
 template:
   metadata:
     labels:
       app: ip-preserving-host-device-cni
   spec:
     priorityClassName: system-node-critical
     terminationGracePeriodSeconds: 5
     tolerations:
       - operator: Exists
     hostNetwork: true
     containers:
       - name: install
         image: ghcr.io/digitalocean-packages/ip-preserving-host-device:v1.0.0
         imagePullPolicy: IfNotPresent
         securityContext:
           privileged: true
         volumeMounts:
           - name: cni-bin
             mountPath: /host/opt/cni/bin
           - name: cni-state
             mountPath: /var/lib/cni/ip-preserving-host-device
     volumes:
       - name: cni-bin
         hostPath:
           path: /opt/cni/bin
           type: DirectoryOrCreate
       - name: cni-state
         hostPath:
           path: /var/lib/cni/ip-preserving-host-device
           type: DirectoryOrCreate
```

### 11.2 `network-attachments.yaml`

```yaml
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric0
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric0"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric1
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric1"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric2
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric2"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric3
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric3"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric4
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric4"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric5
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric5"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric6
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric6"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric7
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric7"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric8
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric8"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric9
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric9"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric10
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric10"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric11
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric11"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric12
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric12"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric13
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric13"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric14
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric14"
   }'
---
apiVersion: "k8s.cni.cncf.io/v1"
kind: NetworkAttachmentDefinition
metadata:
 name: roce-net-fabric15
spec:
 config: '{
     "cniVersion": "0.3.1",
     "type": "ip-preserving-host-device",
     "device": "fabric15"
   }'
```

### 11.3 `test-workload.yaml`

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: 8-gpu-2-fabric-pod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: 8-gpu-2-fabric-pod
  template:
    metadata:
      labels:
        app: 8-gpu-2-fabric-pod
      annotations:
        k8s.v1.cni.cncf.io/networks: >-
          roce-net-fabric0@pod-fabric0,
          roce-net-fabric1@pod-fabric1
    spec:
      restartPolicy: Always
      nodeSelector:
        doks.digitalocean.com/gpu-brand: nvidia
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      volumes:
        - name: temp-hf-cache
          hostPath:
            path: /root/.cache/huggingface
            type: DirectoryOrCreate
        - name: dshm 
          emptyDir:
            medium: Memory
            sizeLimit: 128Gi
      containers:
        - name: server
          image: pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
              name: http-example
          command: ["/bin/bash", "-lc"]
          args:
            - |
             set -ex
             apt update
             apt install -y iproute2 iputils-ping infiniband-diags ibverbs-utils net-tools
             exec sleep infinity 
          securityContext:
            capabilities:
              add:
              - IPC_LOCK
          volumeMounts:
            - name: temp-hf-cache
              mountPath: /root/.cache/huggingface
            - name: dshm
              mountPath: /dev/shm
          resources:
            limits:
              nvidia.com/gpu: "8"
              rdma/fabric0: 100
              rdma/fabric1: 100
            requests:
              nvidia.com/gpu: "8"
              rdma/fabric0: 100
              rdma/fabric1: 100
```

### 11.4 `gpu_rdma_bw_test.py` (Section 5: Single-NIC GPU-to-GPU Test)

```python
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
```

### 11.5 Section 6: AllReduce Test Code

#### `allreduce_bw_test.py`

```python
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
```

#### `run_allreduce_pod0.sh` (run on `8-gpu-2-fabric-pod-0`, `node_rank=0`)

```bash
#!/bin/bash
# Run on 8-gpu-2-fabric-pod-0 (node_rank=0)
set -ex

export MASTER_ADDR=10.121.3.77   # pod-0's own eth0 IP (node_rank=0, the torchrun rendezvous host)
export MASTER_PORT=29500

export NCCL_SOCKET_IFNAME=eth0        # OOB over the Pod's default network
export NCCL_IB_HCA=mlx5_0,mlx5_1      # data plane uses 2 RDMA NICs (fabric0 + fabric1)
export NCCL_IB_GID_INDEX=3            # RoCEv2 GID
export NCCL_DEBUG=INFO
export NCCL_CROSS_NIC=0               # rail-optimized default: strict rail alignment; set to 1 to re-run for comparison

torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /root/allreduce_bw_test.py
```

#### `run_allreduce_pod1.sh` (run on `8-gpu-2-fabric-pod-1`, `node_rank=1`)

```bash
#!/bin/bash
# Run on 8-gpu-2-fabric-pod-1 (node_rank=1)
set -ex

export MASTER_ADDR=10.121.3.77   # must be node_rank=0's (pod-0's) eth0 IP, not its own
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
```

(To compare against `NCCL_CROSS_NIC=1`, just change `NCCL_CROSS_NIC=0` to `1` in both scripts and use a different `MASTER_PORT` — everything else stays the same.)

### 11.6 Section 8: MPI Operator Final Version of `allreduce-2-b300x8-2-fabric.yaml`

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: mpi-multus-b300x8
spec:
  slotsPerWorker: 8
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
            - name: mpi-launcher
              image: ghcr.io/coreweave/nccl-tests:13.2.0-devel-ubuntu24.04-nccl2.29.7-1-7112046
              command:
                - mpirun
                - --allow-run-as-root
                - -np
                - "16"
                - -bind-to
                - none
                - -map-by
                - slot
                - -x
                - NCCL_SOCKET_IFNAME=eth0
                - -x
                - NCCL_DEBUG=INFO
                - -x
                - NCCL_IB_GID_INDEX=3
                - -x
                - NCCL_IB_HCA==mlx5_0,mlx5_1
                - -x
                - NCCL_CROSS_NIC=0
                - -x
                - NCCL_PXN_DISABLE=0
                - -x
                - NCCL_NET_DISABLE_INTRA=1
                - -x
                - NCCL_IB_TC=104 
                #- -x
                #- NCCL_IB_FIFO_TC=192
                #
                - -x
                - PATH
                - -mca
                - plm_rsh_args
                - "-p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
                - -mca
                - btl
                - self,tcp
                - -mca
                - pml
                - ^ucx
                - -x
                - UCX_NET_DEVICES=eth0
                - /opt/nccl-tests/build/all_reduce_perf
                - -b
                - "8"
                - -e
                - 16G
                - -f
                - "16"
                - -g
                - "1"
                - -w
                - "5"
                - --iters
                - "100"
    Worker:
      replicas: 2
      template:
        metadata:
          annotations:
            k8s.v1.cni.cncf.io/networks: >-
              roce-net-fabric0@net0,
              roce-net-fabric1@net1
        spec:
          tolerations:
            - key: "nvidia.com/gpu"
              operator: "Exists"
              effect: "NoSchedule"
          initContainers:
            - name: setup-ssh
              image: ghcr.io/coreweave/nccl-tests:13.2.0-devel-ubuntu24.04-nccl2.29.7-1-7112046
              command:
                - /bin/bash
                - -c
                - |
                  set -ex
                  
                  echo "Setting up SSH configuration..."
                  mkdir -p /ssh-setup/sshd
                  
                  echo "Generating SSH host keys..."
                  ssh-keygen -t rsa -f /ssh-setup/sshd/ssh_host_rsa_key -N '' 2>/dev/null
                  ssh-keygen -t ecdsa -f /ssh-setup/sshd/ssh_host_ecdsa_key -N '' 2>/dev/null
                  ssh-keygen -t ed25519 -f /ssh-setup/sshd/ssh_host_ed25519_key -N '' 2>/dev/null
                  
                  cat > /ssh-setup/sshd/sshd_config << 'SSHD_EOF'
                  Port 2222
                  HostKey /etc/ssh-runtime/ssh_host_rsa_key
                  HostKey /etc/ssh-runtime/ssh_host_ecdsa_key
                  HostKey /etc/ssh-runtime/ssh_host_ed25519_key
                  PermitRootLogin yes
                  PubkeyAuthentication yes
                  AuthorizedKeysFile /root/.ssh/authorized_keys /root/.ssh/id_rsa.pub
                  PasswordAuthentication no
                  ChallengeResponseAuthentication no
                  UsePAM no
                  PrintMotd no
                  PidFile /var/run/sshd.pid
                  StrictModes no
                  SSHD_EOF
                  
                  echo "SSH setup complete"
                  echo "Files in /ssh-setup/sshd:"
                  ls -la /ssh-setup/sshd/
              volumeMounts:
                - name: ssh-setup
                  mountPath: /ssh-setup
          containers:
            - name: mpi-worker
              image: ghcr.io/coreweave/nccl-tests:13.2.0-devel-ubuntu24.04-nccl2.29.7-1-7112046
              command:
                - /bin/bash
                - -c
                - |
                  set -ex
                  
                  echo "Starting worker container..."
                  mkdir -p /var/run/sshd /etc/ssh-runtime
                  
                  echo "Copying SSH host keys and config from init container..."
                  cp /ssh-setup/sshd/* /etc/ssh-runtime/
                  
                  echo "SSH runtime files:"
                  ls -la /etc/ssh-runtime/
                  
                  echo "Verifying operator-provided SSH keys at /root/.ssh..."
                  ls -la /root/.ssh/ || echo "Warning: /root/.ssh not found"
                  
                  echo "Starting sshd in foreground..."
                  exec /usr/sbin/sshd -D -e -f /etc/ssh-runtime/sshd_config
              securityContext:
                privileged: true
                capabilities:
                  add:
                    - IPC_LOCK
              volumeMounts:
                - name: ssh-setup
                  mountPath: /ssh-setup
                - name: dshm
                  mountPath: /dev/shm
              resources:
                limits:
                  nvidia.com/gpu: 8
                  rdma/fabric0: 100
                  rdma/fabric1: 100
                requests:
                  nvidia.com/gpu: 8
                  rdma/fabric0: 100
                  rdma/fabric1: 100
              readinessProbe:
                tcpSocket:
                  port: 2222
                initialDelaySeconds: 5
                periodSeconds: 3
                timeoutSeconds: 2
              livenessProbe:
                tcpSocket:
                  port: 2222
                initialDelaySeconds: 15
                periodSeconds: 10
                timeoutSeconds: 2
          volumes:
            - name: ssh-setup
              emptyDir: {}
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 16Gi
```

(To compare against `NCCL_CROSS_NIC=1`, just change `NCCL_CROSS_NIC=0` to `1` — everything else stays the same.)
