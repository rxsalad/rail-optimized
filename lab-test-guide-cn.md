# Rail-Optimized GPU Fabric 实验指南

> 本文档既是一份实验记录，也是一份可复现的操作指南：按顺序执行每一步，就能在同类环境里重新搭建这套 RDMA 直通网络方案，并跑通文档里的全部验证性实验（连通性、RDMA 带宽、GPU-to-GPU 通信、NCCL AllReduce）。每一步都附带实际跑出来的命令输出作为参考，方便对照自己的结果。
>
> **重要前提**：这是一个模拟的 rail-optimized 环境，本文档中所有性能/带宽数据（RDMA、NCCL 等）都不具备严格参考价值，所有测试的目的是验证功能连通性，不是做严谨的性能压测。

<a id="toc"></a>
## 目录

0. [环境概况与已知限制](#sec-0)
1. [RDMA/RoCE 直通网络：原理与部署](#sec-1)
2. [部署测试负载并验证](#sec-2)
3. [实验一：连通性测试（同 rail / 跨 rail Ping）](#sec-3)
4. [实验二：RDMA 带宽测试（`ib_write_bw`）](#sec-4)
5. [实验三：PyTorch GPU-to-GPU RDMA（单 NIC）](#sec-5)
6. [实验四：PyTorch AllReduce（双 NIC，8 GPU/Pod）](#sec-6)
7. [清理环境](#sec-7)
8. [实验五：MPI Operator AllReduce（nccl-tests，双 NIC，8 GPU/Pod）](#sec-8)
9. [踩坑速查表](#sec-9)
10. [待办 / 下一步](#sec-10)
11. [附录：完整 YAML 与脚本代码](#sec-11)

---

<a id="sec-0"></a>
## 0. 环境概况与已知限制
[↑ 回到目录](#toc)

### 0.1 集群

- **集群**：DigitalOcean Kubernetes (DOKS)，版本 v1.36.0。
- **访问方式**：通过跳板机 `nuc` 访问，`kubectl` 已配置好 kubeconfig；也可从 `nuc` 直接 SSH 到各节点公网 IP（两台 GPU 节点的内网 IP 互相之间不可直连，只能走公网 IP 经 `nuc` 跳转）。

### 0.2 节点

| 节点名 | 角色 | 内网 IP | 外网 IP | GPU |
|---|---|---|---|---|
| rs-cpu-pool-3y0p31 | CPU 池 | 10.120.0.6 | 165.22.160.233 | 无 |
| rs-gpu-pool-3xnkk5 | GPU 池 | 10.120.0.7 | 138.68.59.208 | 8x NVIDIA B300 SXM6 |
| rs-gpu-pool-3xnkkk | GPU 池 | 10.120.0.8 | 134.209.13.101 | 8x NVIDIA B300 SXM6 |

OS：Debian 13 (trixie)，Kernel 6.12.94，containerd 2.2.3。

### 0.3 开始之前必须知道的限制

- **本文档所有性能/带宽数据都不具备严格参考价值**——目的是验证功能是否连通，不是压测最大性能。
- **硬件已知问题**：2 台 B300x8 GPU 节点中，有一台的 1 个 RDMA NIC 是坏的，导致两台机器上的 RDMA NIC 编号从 `fabric3` 往后就不对齐了（两台节点 `fabric3` 及以后不再是同一条物理 rail）。因此本文档所有实验都只使用两台机器上唯一对齐的 2 张 RDMA NIC：`fabric0`/`fabric1`。如果你的环境网卡是好的，建议按同样方法先确认哪些 rail 编号在多台机器间是对齐的，再决定用哪几条 rail 做实验。

---

<a id="sec-1"></a>
## 1. RDMA/RoCE 直通网络：原理与部署
[↑ 回到目录](#toc)

### 1.1 整体网络架构原理

- **基线拓扑（引入 spine 层之前）**：每个 SU（Scaling Unit）最多由 64 台 B300x8 服务器组成；RDMA 后端网络在 SU 级别是隔离的——依赖 RDMA 的分布式应用被限制在单个 SU 内，无法跨 SU 通信。网络拓扑是纯 rail-only 的，由 16 个 Layer-2 域组成（对应本文档反复用到的 `fabric0`~`fabric15` 这 16 条 rail，每条 rail 自成一个 L2 广播域，彼此互不相通）。
- 新增的 **spine 层**把原本各个 SU（Scaling Unit）互相隔离的 RDMA 后端网络连接起来，使得跨 SU、跨 rail 的通信成为可能。
- RDMA 后端网络因此从原来彼此隔离的 **Layer-2 域**，升级成了一个 **Layer-3 路由网络**：IP 地址规划会按 server、leaf、fabric 等基础设施层级分配，路由也按对应层级做汇总（summarization）。
- DOKS 工作节点上的每张 RDMA NIC 会同时拿到两种 IPv6 地址：
  - 一个 `fe80::/10` 范围内的 **link-local 地址**；
  - 一个按基础设施划分的 **IPv6 ULA 地址**（如 `fd02::/64`，即本文档反复用到的 `fd02:0:0:58::1` 这类地址）。
- Rail-optimized 的 GPU fabric **默认开启 ECMP**（等价多路径），为跨 rail/跨 SU 通信提供多条可用路径。
- DOKS 工作节点上，**每张 RDMA NIC 各自关联一个独立的 VRF 和路由表**，为 IP 层面的连接建立（QP 建连）和转发提供隔离的路由上下文（后面第 2.4 节、第 3 节会反复验证到"每张 NIC 路由表各自独立"这个现象，根源就在这里）。
- 要让 Pod 里的应用能用上这些 RDMA NIC，DigitalOcean 提供了专门的 `ip-preserving-host-device` CNI 插件——把网卡搬进 Pod netns 的同时，保留它的 IPv6 ULA 地址和关联路由。**标准的 Host-Device 插件不具备这个能力**，下面就来部署和验证这个插件。

### 1.2 `ip-preserving-host-device` 插件原理

它是标准 Host-Device CNI 插件的扩展版——标准版把网卡移进 Pod netns 时不保留原有网络配置，而这个插件在搬移 RDMA 网卡的同时，完整保留了该网卡原有的 IPv6 网络身份，包括：

- 网卡上的 IPv6 ULA 地址（如 `fd02:0:0:58::1`）
- Link-local IPv6 地址
- 该接口关联的 IPv6 路由
- VRF 归属关系及对应的路由表

也就是说，RDMA 网卡从主机 netns 移进 Pod 之后，网络身份和路由配置原样带过去——Pod 内的 RDMA 应用能直接用上这张网卡在 DOKS 工作节点上原本就有的 fabric 专属 IP 和路由配置，不需要额外配置。后面第 2 节部署测试负载、第 2.4 节归还验证都会看到这个特性的实际效果。

### 1.3 步骤一：部署 `daemonset.yaml`（安装 CNI 插件）

完整内容见附录 11.1。作用：在 `kube-system` 部署 DaemonSet `ip-preserving-host-device-cni`，跑在全部节点（含 CPU 池），特权模式安装 CNI 插件 `ip-preserving-host-device`（镜像 `ghcr.io/digitalocean-packages/ip-preserving-host-device:v1.0.0`），把宿主机上的 RDMA 网卡透传进 Pod 并保留原始 IP。

```bash
kubectl apply -f daemonset.yaml
```

预期输出：

```
$ kubectl apply -f daemonset.yaml
daemonset.apps/ip-preserving-host-device-cni created

$ kubectl get ds -n kube-system ip-preserving-host-device-cni
NAME                            DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   AGE
ip-preserving-host-device-cni   3         3         3       3            3           15s
```

**检查点**：确认 CNI 二进制真的落地到了宿主机上（而不只是 Pod 状态显示 Running）：

```
root@rs-gpu-pool-3xnkkk:/opt/cni/bin# ls -la
...
-rwxr-xr-x 1 root root  5159967 Apr 24 10:38 host-device               # 标准 host-device 插件
-rwxr-xr-x 1 root root  4006048 Sep 27 15:33 ip-preserving-host-device  # 本次 DaemonSet 装的插件，时间戳与 apply 时刻一致
-rwxr-xr-x 1 root root 50876101 Sep 26 00:03 multus-shim
...
```

`ip-preserving-host-device` 二进制确认存在，跟标准 `host-device` 插件放在同一个目录（`/opt/cni/bin`，供 Multus 调用），时间戳与本次 `apply` 的时间点一致，证明 DaemonSet 的 `install` 容器确实把插件正确分发到了节点上。

### 1.4 步骤二：部署 `network-attachments.yaml`（定义每张 RDMA NIC 对应的 NAD）

完整内容见附录 11.2。作用：`NetworkAttachmentDefinition`（NAD）是 Multus CNI 定义"附加网络"的配置对象——Multus 读取这些定义，据此决定如何创建对应的网络接口、并把它挂载到 Pod 上（Pod 只需要在 annotation 里引用某个 NAD 的名字，如 `roce-net-fabric0@pod-fabric0`，具体怎么创建接口的细节都封装在 NAD 里）。这个文件定义了 16 个 NAD（`roce-net-fabric0` ~ `roce-net-fabric15`），每个对应宿主机上一张物理 RDMA 网卡（`fabric0`~`fabric15`），类型均为 `ip-preserving-host-device`。

```bash
kubectl apply -f network-attachments.yaml
```

预期输出：

```
$ kubectl apply -f network-attachments.yaml
networkattachmentdefinition.k8s.cni.cncf.io/roce-net-fabric0 created
... (fabric1 ~ fabric15 依次创建，共 16 条)

$ kubectl get network-attachment-definitions -A | wc -l
17   # 16 条 NAD + 1 行表头
```

### 1.5 GPU 与 RDMA NIC 的物理亲和关系（PCI 拓扑）

在正式跑实验之前，值得先搞清楚一件事：**你要用的这几张 RDMA NIC，物理上离哪块 GPU 最近？** 这决定了后面多 GPU 测试的拓扑瓶颈在哪。不依赖 `nvidia-smi topo -m` 这种上层抽象，直接用最底层的 PCI 总线拓扑（`lspci`）推导。以节点 `rs-gpu-pool-3xnkkk` 为例：

**第一步：拿到每个 GPU 的 PCI Bus ID**

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

**第二步：拿到每张 mlx5 网卡的 PCI Bus ID**（`ibv_devinfo`/`rdma link` 只给设备名如 `mlx5_0`，要落到具体 PCI 地址需要看 sysfs 的 `device` 软链接）

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

**第三步：用 `lspci -tv` 打印完整 PCIe 总线树**，看 GPU 和哪些网卡共享同一个 Root Complex：

```
root@rs-gpu-pool-3xnkkk:~# lspci -tv
-+-[0000:00]-+-00.0  Intel Corporation 82G33/G31/P35/P31 Express DRAM Controller
 |           +-... (QEMU 虚拟外设 / ICH9 南桥外设，与本次分析无关，省略)
 +-[0000:80]---00.0-[81-86]----00.0-[82-86]--+-00.0-[83-85]----00.0-[84-85]----00.0-[85]----00.0  NVIDIA Corporation GB110 [B300 SXM6 AC]   # GPU0
 |                                           \-01.0-[86]--+-00.0  Mellanox ConnectX Family mlx5Gen Virtual Function   # mlx5_0
 |                                                        \-00.1  Mellanox ConnectX Family mlx5Gen Virtual Function   # mlx5_1
 +-[0000:88]---00.0-[89-8e]----00.0-[8a-8e]--+-00.0-[8b-8d]----00.0-[8c-8d]----00.0-[8d]----00.0  NVIDIA Corporation GB110 [B300 SXM6 AC]   # GPU1
 |                                           \-01.0-[8e]--+-00.0  Mellanox ConnectX Family mlx5Gen Virtual Function   # mlx5_2
 |                                                        \-00.1  Mellanox ConnectX Family mlx5Gen Virtual Function   # mlx5_3
 ... (GPU2~GPU7 与 mlx5_4~mlx5_15 的结构完全一样，依次对应，此处省略)
 \-[0000:c0]---00.0-[c1-c3]----00.0-[c2-c3]----00.0-[c3]--+-00.0  Mellanox MT2910 Family [ConnectX-7]
                                                          +-00.1  Mellanox MT2910 Family [ConnectX-7]
                                                          +-00.2  Mellanox MT2910 Family [ConnectX-7]
                                                          \-00.3  Mellanox MT2910 Family [ConnectX-7]   # 4 个额外端口，不与任何 GPU 共享 Root Complex
```

树状结构非常清晰：每个 GPU 和它"最近"的两张网卡，是从**同一个 PCIe Root Complex**（如 `0000:80`）分出两条支路——一条经过若干级 PCIe Switch 连到 GPU，另一条直接连到同一张物理 ConnectX 卡切出来的两个 VF（网卡）。汇总成表：

| GPU | GPU PCI Bus | 同根（最近）的 NIC | NIC PCI Bus |
|---|---|---|---|
| GPU0 | 85:00.0 | mlx5_0 / mlx5_1 | 86:00.0 / 86:00.1 |
| GPU1 | 8D:00.0 | mlx5_2 / mlx5_3 | 8e:00.0 / 8e:00.1 |
| GPU2 | 95:00.0 | mlx5_4 / mlx5_5 | 96:00.0 / 96:00.1 |
| GPU3 | 9D:00.0 | mlx5_6 / mlx5_7 | 9e:00.0 / 9e:00.1 |
| GPU4 | A5:00.0 | mlx5_8 / mlx5_9 | a6:00.0 / a6:00.1 |
| GPU5 | AD:00.0 | mlx5_10 / mlx5_11 | ae:00.0 / ae:00.1 |
| GPU6 | B5:00.0 | mlx5_12 / mlx5_13 | b6:00.0 / b6:00.1 |
| GPU7 | BD:00.0 | mlx5_14 / mlx5_15 | be:00.0 / be:00.1 |

规律：`GPU[n]` 对应 `mlx5_[2n]` 和 `mlx5_[2n+1]`，`nvidia-smi topo -m` 给出的距离矩阵与此完全一致（同组 = `PXB`，跨组 = `SYS`）。

**这就是"rail-optimized"拓扑的物理基础：每个 GPU 都被绑定了一对物理上最近的 RDMA 网卡**，数据搬运时可以只经过本地 PCIe Switch，不用跨 NUMA/穿 CPU Host Bridge。**关键含义：本文档所有实验用的 `mlx5_0`/`mlx5_1`（`fabric0`/`fabric1`）在物理上只跟 GPU0 同根**，跟 GPU1~7 不共享 Root Complex——这个事实在第 6 节 AllReduce 测试的瓶颈分析中很关键，如果你打算做多 GPU 测试，先按这个方法确认好拓扑再解读结果。

### 1.6 部署前基线：检查主机侧网卡状态

部署 CNI（1.3/1.4 节）不会立刻移动任何网卡——只有当 Pod 真正引用 NAD 时才会触发。所以现在（还没部署 Pod）先记录一下主机侧的基线状态，方便后面对比：

```
$ ssh root@138.68.59.208 "ip -br link show | grep -i fabric"
fabric0 ~ fabric14  UP  (缺 fabric15)
vrf-fabric0 ~ vrf-fabric14  UP

$ ssh root@134.209.13.101 "ip -br link show | grep -i fabric"
fabric0 ~ fabric15  UP  (16 张网卡齐全)
vrf-fabric0 ~ vrf-fabric15  UP
```

**接口/VRF/路由关系**：主机上 `fabric0`/`fabric1` 分别被 enslave 到独立的 `vrf-fabric0`/`vrf-fabric1`（各自一张单独的路由表），IP 分别是：

| 节点 | fabric0 (rail0) | fabric1 (rail1) |
|---|---|---|
| rs-gpu-pool-3xnkkk（134.209.13.101） | `fd02:0:0:58::1` | `fd02:0:0:59::1` |
| rs-gpu-pool-3xnkk5（138.68.59.208） | `fd02:0:0:178::1` | `fd02:0:0:179::1` |

每条 rail 的 VRF 路由表里除了本地 `/64` 直连路由，还有一条 `fd02::/32 via <本 rail 网关> onlink` 的默认路由（网关地址是该 rail 在 leaf 交换机上对应端口的地址，例如 `fabric0` 网关是 `fd02:0:0:58::2`），所有不在本机 `/64` 子网内的 `fd02::/32` 流量都要先送到这个网关（leaf 交换机），再由交换机转发。**两条 rail 各自的网关是独立的、不同的接口**，同一台主机上的 `vrf-fabric0` 和 `vrf-fabric1` 之间没有本地路由捷径——哪怕 fabric0 和 fabric1 物理上插在同一台服务器上，跨 rail 通信也必须先出网线到 leaf 交换机绕一圈再回来。

**小实验：用 ping 的 TTL 验证"leaf 环回"行为**（默认发出 TTL=64）：

```
# 跨节点、同 rail：node2(134.209.13.101).fabric0 -> node1(138.68.59.208).fabric0
$ ssh root@134.209.13.101 "ping -6 -I fabric0 -c 4 fd02:0:0:178::1"
64 bytes from fd02:0:0:178::1: icmp_seq=1 ttl=62 ...   # 少 2 跳（比如 leaf+spine，或两级 leaf）
rtt avg 0.129 ms

# 跨节点、同 rail：node2.fabric1 -> node1.fabric1
$ ssh root@134.209.13.101 "ping -6 -I fabric1 -c 4 fd02:0:0:179::1"
64 bytes from fd02:0:0:179::1: icmp_seq=1 ttl=62 ...   # 同样少 2 跳
rtt avg 0.116 ms

# 同节点、跨 rail（在 rs-gpu-pool-3xnkkk 上）：fabric0 -> fabric1
$ ssh root@134.209.13.101 "ping -6 -I fabric0 -c 4 fd02:0:0:59::1"
64 bytes from fd02:0:0:59::1: icmp_seq=1 ttl=63 ...    # 只少 1 跳
rtt avg 0.102 ms

# 同节点、跨 rail（在 rs-gpu-pool-3xnkk5 上）：fabric0 -> fabric1
$ ssh root@138.68.59.208 "ping -6 -I fabric0 -c 4 fd02:0:0:179::1"
64 bytes from fd02:0:0:179::1: icmp_seq=1 ttl=63 ...   # 只少 1 跳
rtt avg 0.081 ms
```

四组全部 0% 丢包。**TTL 的差异证实了"leaf 环回"的猜测**：跨节点同 rail（TTL 少 2）经过了 2 段路由跳；而同一台机器上 fabric0 ping fabric1（TTL 只少 1）只经过 1 段路由跳——数据包从 `fabric0` 出网线到 leaf 交换机，交换机直接把它转发回同一台机器的 `fabric1` 端口，不需要再经过第二级（spine）。即使物理上是同一台服务器，两条 rail 在网络层面也被当成两个完全独立的、只能通过外部交换机互通的网段。

**反向验证：不指定 `-I` 绑定出口设备，直接 ping 远端地址会失败**，进一步证明上面这些路由确实只存在于对应 rail 的 VRF 表里，不在默认 `main` 表：

```
$ ssh root@134.209.13.101 "ping -6 -c 4 fd02:0:0:178::1"     # 不带 -I fabric0
ping: connect: Network is unreachable
```

原因：`ip -6 rule show` 里 `1000: from all lookup [l3mdev-table]` 这条 l3mdev 规则要靠**已经确定的出口设备**才能定位到它所属的 VRF 表；不指定 `-I`（不绑定设备）时，内核走的是默认的 `32766: from all lookup main` 规则，而 `main` 表里根本没有 `fd02::/32` 这条路由（它只存在于 `vrf-fabric0`/`vrf-fabric1` 各自的表里），所以直接报 `Network is unreachable`。**记住这个结论——后面第 4 节 `ib_write_bw` 会踩到同一个坑。**

**RDMA 设备层基线**：以上都是网络层（netdevice/IP/路由）的信息，再用几个 RDMA/verbs 专用工具从设备层面确认一下 `fabric0`/`fabric1`（`mlx5_0`/`mlx5_1`）跟其它 14 张卡的状态（以 `rs-gpu-pool-3xnkkk` 为例）：

```
$ rdma link | grep -i fabric
link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev fabric0
link mlx5_1/1 state ACTIVE physical_state LINK_UP netdev fabric1
... (mlx5_2~mlx5_15 依次列出，均 ACTIVE/LINK_UP，netdev 分别是 fabric2~fabric15)

$ ibv_devices
    device          	   node GUID
    ------          	----------------
    mlx5_0          	4ebb47fffe67f96a
    mlx5_1          	4ebb47fffe67f96b
    ... (mlx5_2~mlx5_15，此外还有 mlx5_16~mlx5_19 这 4 个额外端口，见下方 ibdev2netdev 的说明)

$ ibv_devinfo -v | grep -E "hca_id|GID\["
hca_id:	mlx5_0
			GID[  0]:		fe80:0000:0000:0000:4cbb:47ff:fe67:f96a, RoCE v1
			GID[  1]:		fe80::4cbb:47ff:fe67:f96a, RoCE v2
			GID[  2]:		fd02:0000:0000:0058:0000:0000:0000:0001, RoCE v1
			GID[  3]:		fd02:0:0:58::1, RoCE v2      # 本文档实验一直用的 GID index 3
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
... (fabric1 的 v1/v2 各 2 条，同样的结构)

$ ibdev2netdev
mlx5_0 port 1 ==> fabric0 (Up)
mlx5_1 port 1 ==> fabric1 (Up)
mlx5_2 port 1 ==> fabric2 (Up)
... (mlx5_3~mlx5_15 都是 fabricN (Up))
mlx5_16 port 1 ==> ibp195s0f0 (Down)
mlx5_17 port 1 ==> ibp195s0f1 (Down)
mlx5_18 port 1 ==> ibp195s0f2 (Down)
mlx5_19 port 1 ==> ibp195s0f3 (Down)
```

`ibdev2netdev` 这里正好印证了第 8.4 节调试 MPI Operator 时踩到的 `NCCL_IB_HCA` 前缀匹配坑：`mlx5_16`~`mlx5_19` 这 4 个额外端口本来就是 **Down 状态、没有关联到任何 `fabricN` netdev**（对应 `ibdev2netdev` 里的 `ibp195s0f0~3`），跟 `mlx5_0`~`mlx5_15` 完全不是一路的东西——这也是为什么当年 NCCL 一旦把它们误匹配进设备列表，`ibv_create_ah` 立刻报 `No such device`。

---

<a id="sec-2"></a>
## 2. 部署测试负载并验证
[↑ 回到目录](#toc)

### 2.1 workload 结构（`test-workload.yaml`）

完整内容见附录 11.3。这是本文档所有实验共用的测试负载：

- `StatefulSet` 名为 `8-gpu-2-fabric-pod`，2 个副本，目标是 2 台 GPU 节点各起一个 Pod。
- Pod annotation `k8s.v1.cni.cncf.io/networks` 挂载 `roce-net-fabric0`（命名为 `pod-fabric0`）和 `roce-net-fabric1`（`pod-fabric1`）。
- `nodeSelector: doks.digitalocean.com/gpu-brand: nvidia` + 容忍 `nvidia.com/gpu:NoSchedule` 污点，确保调度到 GPU 节点。
- 镜像 `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel`（PyTorch 2.13.0+cu130，NCCL 2.29.7），启动时装网络诊断工具（`iproute2`/`ibverbs-utils`/`infiniband-diags` 等）后 `sleep infinity`，供手动 exec 调试。
- 资源：每 Pod `nvidia.com/gpu: 8`、`rdma/fabric0: 100`、`rdma/fabric1: 100`；128Gi 内存型 `/dev/shm`（NCCL 用）+ 挂载宿主机 HuggingFace 缓存目录。

### 2.2 部署

```bash
kubectl apply -f test-workload.yaml
```

预期输出：

```
$ kubectl apply -f test-workload.yaml
statefulset.apps/8-gpu-2-fabric-pod created

$ kubectl get pods -l app=8-gpu-2-fabric-pod -o wide
NAME                   READY   STATUS    IP             NODE
8-gpu-2-fabric-pod-0   1/1     Running   10.121.3.77    rs-gpu-pool-3xnkkk  (134.209.13.101)
8-gpu-2-fabric-pod-1   1/1     Running   10.121.2.170   rs-gpu-pool-3xnkk5  (138.68.59.208)
```

### 2.3 验证：Pod 内部状态

```
$ kubectl exec 8-gpu-2-fabric-pod-0 -- ip -br link
pod-fabric0      UP  4e:bb:47:67:f9:6a  <BROADCAST,MULTICAST,UP,LOWER_UP>
pod-fabric1      UP  4e:bb:47:67:f9:6b  <BROADCAST,MULTICAST,UP,LOWER_UP>
eth0@if63        UP  ...

$ kubectl exec 8-gpu-2-fabric-pod-0 -- nvidia-smi -L
GPU 0~7: NVIDIA B300 SXM6 AC  (8 张全部可见)
```

**检查点**：`pod-fabric0`/`pod-fabric1` 的 MAC 地址应该跟对应主机上原 `fabric0`/`fabric1` 完全一致——这确认是同一张物理网卡被移入 Pod netns（而非虚拟设备），`ip-preserving-host-device` 的直通特性生效。

Pod 内 fabric 地址（后续实验反复用到，建议记下来）：

| Pod | pod-fabric0 (rail0/mlx5_0) | pod-fabric1 (rail1/mlx5_1) | eth0 (Pod 默认网络) |
|---|---|---|---|
| `8-gpu-2-fabric-pod-0`（node `rs-gpu-pool-3xnkkk`） | `fd02:0:0:58::1` | `fd02:0:0:59::1` | `10.121.3.77` |
| `8-gpu-2-fabric-pod-1`（node `rs-gpu-pool-3xnkk5`） | `fd02:0:0:178::1` | `fd02:0:0:179::1` | `10.121.2.170` |

RDMA 设备对应：`mlx5_0` ↔ `pod-fabric0`（rail0），`mlx5_1` ↔ `pod-fabric1`（rail1）。GID[3]（RoCEv2）与上述地址一致，可用 `kubectl exec ... -- ibv_devinfo -v | grep -E "hca_id|GID\["` 自行核对。

**VRF 与路由表结构**：跟第 1.6 节主机侧观察到的现象完全一样，Pod 内 `pod-fabric0`/`pod-fabric1` 各自被 enslave 到独立的 VRF（`vrf100`/`vrf101`，对应路由表 100/101）：

```
$ kubectl exec 8-gpu-2-fabric-pod-0 -- ip -6 route show           # 默认 main 表：几乎是空的
fe80::/64 dev eth0 proto kernel metric 256 pref medium

$ kubectl exec 8-gpu-2-fabric-pod-0 -- ip -6 route show vrf vrf100  # pod-fabric0 对应的 VRF 路由表
fd02:0:0:58::/64 dev pod-fabric0 proto kernel metric 256 pref medium
fd02::/32 via fd02:0:0:58::2 dev pod-fabric0 proto static metric 1024 onlink pref medium   # 跨节点路由，只在这张 VRF 表里
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

**记住这个结论**：跨节点路由只存在于对应的 VRF 路由表（100/101）里，不在默认 `main` 路由表。第 4 节 `ib_write_bw` 会因为不知道这一点而踩坑。

### 2.4 部署后：主机侧网卡与路由是否移除

```
$ ssh root@134.209.13.101 "ip -br link show | grep -iE 'fabric0|fabric1\b'"   # pod-0 所在节点
vrf-fabric0 / vrf-fabric1  UP   (fabric0/fabric1 本体已不在链路列表里)

$ ssh root@138.68.59.208  "ip -br link show | grep -iE 'fabric0|fabric1\b'"   # pod-1 所在节点
vrf-fabric0 / vrf-fabric1  UP   (同上)

# 路由表也一并清空：
$ ssh root@138.68.59.208 "ip route show | grep -i fabric"        # main 路由表：无输出
$ ssh root@138.68.59.208 "ip route show vrf vrf-fabric0"          # VRF 自己的路由表：无输出
$ ssh root@138.68.59.208 "ip route show vrf vrf-fabric1"          # 无输出
（134.209.13.101 结果相同）
```

**结论：两台主机上原本的 `fabric0`/`fabric1` 物理网卡接口和对应路由都已从主机默认网络命名空间中完全移出，只保留对应的 `vrf-fabric0`/`vrf-fabric1`（VRF 主设备本身未被移走，但已经是空壳——没有 slave 设备、没有路由）。** 原因：`fd02:0:0:xx::/64 dev fabric0` 这些路由挂在物理设备 `fabric0` 上，设备被 CNI 插件整体移到 Pod netns 后，路由跟着设备一起被移走。其余未被 Pod 使用的 `fabric2`~`fabric14/15` 仍留在主机上，未受影响，符合 `ip-preserving-host-device` 插件的预期行为。

### 2.5 补充实验：主机侧"消失"只是网络层，RDMA verbs 层的 GID 其实全局共享

上面 2.4 节的结论容易让人误以为 `mlx5_0`/`mlx5_1` 这两张卡在主机上"完全不存在了"。进一步用三种不同工具在主机上查这两张卡的 GID，会发现结果并不一致：

| 工具 | 结果 | 原因 |
|---|---|---|
| 裸读 `/sys/class/infiniband/mlx5_0/ports/1/gids/*` | 全零 / 报错 | 逻辑是"先看有没有关联的 netdevice"，`mlx5_0`/`mlx5_1` 的 netdevice 已经被移进 Pod，主机看不到 |
| `show_gids` | 完全不列出 `mlx5_0`/`mlx5_1` | 同上，只跳过没有关联 netdevice 的卡 |
| `ibv_devinfo -v` | **非空**，且跟 Pod 内查到的值完全一致（如 `fd02:0:0:58::1`） | 走 verbs API 直接查硬件全局 GID 表，不关心当前 netns |

根因：`rdma system show` 显示这台机器是 `netns shared` 模式，即 RDMA 子系统在内核里**不按网络命名空间隔离**，全局唯一一份，因此 `ibv_devinfo -v` 能绕开 netns 边界查到硬件表里真实生效的 GID。

**结论（澄清"主机能不能用这张卡"）**：
1. 主机上"看得到 GID"是 **Pod 在用**这张卡产生的状态被内核共享暴露出来的，不代表主机本来就有这个能力；如果 Pod 没给网卡配 IP，这个 GID 条目根本不存在。
2. 主机不能真正用这两张卡做网络通信——RDMA/RoCE 通信需要本地 netns 里有对应的 netdevice/IP/路由，这些东西已经完整转移进 Pod，主机 `ip link` 里看不到 `fabric0`/`fabric1`（跟 2.4 节的结论一致）。
3. 两个隔离层次不同：**网络层**（netdevice/IP/路由）被 Pod 独占，主机完全拿不到；**RDMA 设备层**（verbs/uverbs、硬件 GID 表）在 `netns shared` 模式下全局共享，主机能看到但不应该、也不能安全地绕过去使用——正常结论是**这两张卡的使用权完全属于 Pod，主机看得到但用不了**。
4. 补充：`mlx5_0`/`mlx5_1` 是从物理功能（PF，`c3:00.x`，ConnectX-7）切出来的 SR-IOV VF，被整体分配给 Pod，不是独立物理卡。

（下面第 3~6 节的所有实验都在这个已经部署好的 workload 上进行。做完全部实验后，第 7 节会再做一次删除，验证网卡能否正确归还主机。）

---

<a id="sec-3"></a>
## 3. 实验一：连通性测试（同 rail / 跨 rail Ping）
[↑ 回到目录](#toc)

背景：GPU fabric 是 **rail-optimized** 设计（`fabric0`=rail0/`mlx5_0`，`fabric1`=rail1/`mlx5_1`），这种设计允许跨 rail 通信，需要验证不同节点间"错配 rail"（如 pod0 的 rail0 对 pod1 的 rail1）也能正常通信，而不仅仅是同 rail 对同 rail。

在两个 Pod 都 Running 之后，用 2.3 节记录的 fabric 地址做 4 组测试：

```bash
# 1. 同 rail: pod0 fabric0(rail0) -> pod1 fabric0(rail0)
kubectl exec 8-gpu-2-fabric-pod-0 -- ping -6 -I pod-fabric0 -c 4 fd02:0:0:178::1

# 2. 同 rail: pod0 fabric1(rail1) -> pod1 fabric1(rail1)
kubectl exec 8-gpu-2-fabric-pod-0 -- ping -6 -I pod-fabric1 -c 4 fd02:0:0:179::1

# 3. 跨 rail: pod0 fabric0(rail0) -> pod1 fabric1(rail1)
kubectl exec 8-gpu-2-fabric-pod-0 -- ping -6 -I pod-fabric0 -c 4 fd02:0:0:179::1

# 4. 跨 rail: pod0 fabric1(rail1) -> pod1 fabric0(rail0)
kubectl exec 8-gpu-2-fabric-pod-0 -- ping -6 -I pod-fabric1 -c 4 fd02:0:0:178::1
```

预期结果（4 组全部 0% 丢包、延迟一致 ~0.1ms）：

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

**结论：跨 rail 的 L3 连通性完全正常，0% 丢包，延迟与同 rail 一致（~0.1ms）**，底层交换网络对跨 rail 流量做了正确路由/转发，两条 rail 之间没有网络隔离问题。

---

<a id="sec-4"></a>
## 4. 实验二：RDMA 带宽测试（`ib_write_bw`）
[↑ 回到目录](#toc)

### 4.1 会踩的坑：直接用 fabric 网卡 IPv6 地址做 OOB

第一反应可能是直接拿 fabric 地址跑：

```bash
kubectl exec 8-gpu-2-fabric-pod-0 -- ib_write_bw -d mlx5_0 --ipv6-addr fd02:0:0:178::1
```

会得到：

```
Network is unreachable / Couldn't connect
```

根因跟第 1.6/2.3 节讲的一样：`pod-fabric0`/`pod-fabric1` 被 enslave 到各自的 VRF，跨节点路由 `fd02::/32` 只存在于对应 VRF 的路由表里，不在容器默认的 `main` 路由表。`ping -I <iface>` 能成功是因为显式绑定了出口设备绕开了路由歧义，而 `ib_write_bw` 没有等价的"绑定设备"选项。尝试过的绕过方法均失败：`ip vrf exec`（容器 cgroup 只读）、`ip -6 rule add`（缺 `NET_ADMIN`）、`ib_write_bw -R` rdma_cm 模式（同样走内核路由表，一样受限）。

### 4.2 解法：OOB 走 eth0，RDMA 数据面走 fabric NIC 的 GID

`ib_write_bw` 的连接目标地址只是用来交换 QP 元数据的**带外（OOB）控制通道**，实际 RDMA 数据搬运走的是 `-d` 指定设备的 GID（RoCEv2 GRH 寻址），二者可以分离。让 OOB 走 Pod 默认网络 `eth0`（有正常的 `main` 路由表 default route），RDMA 数据面继续走 fabric 网卡的 GID，即可绕开 4.1 的限制：

```bash
# server（pod1）
kubectl exec 8-gpu-2-fabric-pod-1 -- ib_write_bw -d <mlx5_x> -x 3

# client（pod0），目标写 pod1 的 eth0 IP（OOB）
kubectl exec 8-gpu-2-fabric-pod-0 -- ib_write_bw -d <mlx5_y> -x 3 10.121.2.170
```

按这个模式跑 4 种 rail 组合（RDMA_Write BW，65536B，5000 iterations，单 QP）：

| # | client (pod0) | server (pod1) | 类型 | BW peak (MB/s) | BW avg (MB/s) | MsgRate (Mpps) |
|---|---|---|---|---|---|---|
| 1 | mlx5_0 (rail0) | mlx5_0 (rail0) | 同 rail | 43834.02 | 6789.66 | 0.1086 |
| 2 | mlx5_1 (rail1) | mlx5_0 (rail0) | **跨 rail** | 44117.65 | 3913.35 | 0.0626 |
| 3 | mlx5_0 (rail0) | mlx5_1 (rail1) | **跨 rail** | 44117.65 | 19508.49 | 0.3121 |
| 4 | mlx5_1 (rail1) | mlx5_1 (rail1) | 同 rail | 43885.31 | 15975.77 | 0.2556 |

示例输出（#1，同 rail 基线）：

```
$ kubectl exec 8-gpu-2-fabric-pod-1 -- ib_write_bw -d mlx5_0 -x 3        # server, 后台运行
$ kubectl exec 8-gpu-2-fabric-pod-0 -- ib_write_bw -d mlx5_0 -x 3 10.121.2.170

 Device : mlx5_0   Link type : Ethernet   GID index : 3
 local address:  GID: 253:02:00:00:00:00:00:88:00:00:00:00:00:00:00:01
 remote address: GID: 253:02:00:00:00:00:01:120:00:00:00:00:00:00:00:01
 #bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
 65536      5000             43834.02            6789.66		   0.108635
```

**结论：4 种 rail 组合（同 rail x2、跨 rail x2）RDMA_Write 全部成功建链并传输数据，证明跨 rail 场景下 RDMA 数据面同样可以正常工作。** 各组 BW average 数值波动较大（3913~19508 MB/s），这是因为测试用的是默认参数（单 QP、单一 65536B 消息大小、仅 5000 次迭代，未加 `-a`/`-D` 做完整扫描），本轮目的是验证连通性/功能而非严谨测最大带宽，数据仅供参考。

---

<a id="sec-5"></a>
## 5. 实验三：PyTorch GPU-to-GPU RDMA（单 NIC）
[↑ 回到目录](#toc)

目标：写一个简单脚本，2 个 Pod 之间只用 1 张 RDMA NIC 做 GPU-to-GPU 通信，测 1/2/4/8 GB。

### 5.1 思路

`torch.distributed` + NCCL backend，`world_size=2`，rank0/rank1 各对应一个 Pod；每个 size 先 warmup 3 次 `dist.send`/`dist.recv`，再正式跑 5 次取平均耗时算带宽。

- OOB（进程组 rendezvous）走 Pod 默认网络 `eth0`（`NCCL_SOCKET_IFNAME=eth0`），`MASTER_ADDR` 必须填 **rank0** 所在 Pod 的 eth0 IP（c10d 的 TCPStore server 由 rank0 创建）。
- GPU-to-GPU 数据面强制只用一张 RDMA NIC：`NCCL_IB_HCA=mlx5_0`，`NCCL_IB_GID_INDEX=3`（RoCEv2 GID）。

完整代码见附录 11.4（`gpu_rdma_bw_test.py`）。

### 5.2 运行命令

```bash
# rank1（Pod 8-gpu-2-fabric-pod-1，非 master）
kubectl exec 8-gpu-2-fabric-pod-1 -- bash -lc "\
  MASTER_ADDR=10.121.3.77 MASTER_PORT=29500 RANK=1 WORLD_SIZE=2 \
  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
  python3 -u /root/gpu_rdma_bw_test.py > /root/rank1.log 2>&1"

# rank0（Pod 8-gpu-2-fabric-pod-0，master，MASTER_ADDR 必须是自己的 eth0 IP）
kubectl exec 8-gpu-2-fabric-pod-0 -- bash -lc "\
  MASTER_ADDR=10.121.3.77 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 \
  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
  python3 -u /root/gpu_rdma_bw_test.py > /root/rank0.log 2>&1"
```

> ⚠️ **踩坑提醒**：`dist.init_process_group` 用 `env://` 方式时，TCPStore server 由 `RANK=0` 创建并监听，`MASTER_ADDR` 必须是 rank0 所在 Pod 的可达 IP。如果填成了 rank1 的 IP，两个进程都会去连一个没人监听的地址，两边都会卡死在 `init_process_group`，且没有任何报错或超时提示（需要手动 kill 才能发现）。排查这类卡死问题的技巧：脚本里加分阶段 `print(..., flush=True)` + `python3 -u` 无缓冲执行 + 输出重定向到文件，能一眼看出卡在哪一步（rendezvous / NCCL init / send/recv）。

### 5.3 关键日志（确认走单张 RDMA NIC + GPUDirect RDMA）

```
NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth0
NCCL INFO Bootstrap: Using eth0:10.121.3.77<0>              # OOB 走 eth0
NCCL INFO NCCL_IB_HCA set to mlx5_0                          # 数据面限定单张 NIC
NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [RO]; OOB eth0:10.121.3.77<0>
NCCL INFO DMA-BUF is available on GPU device 0
NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0/GDRDMA   # GPUDirect RDMA，走 IB/0 = mlx5_0
NCCL INFO Connected all rings, use ring PXN 0 GDR 1
```

### 5.4 测试结果

| 数据量 | 平均耗时 | BW average | BW (Gb/s) |
|---|---|---|---|
| 1 GB | 22.84 ms | 47.00 GB/s | 376.01 |
| 2 GB | 44.69 ms | 48.05 GB/s | 384.43 |
| 4 GB | 89.63 ms | 47.92 GB/s | 383.36 |
| 8 GB | 177.32 ms | 48.44 GB/s | 387.55 |

**结论：单张 RDMA NIC（`mlx5_0`/`fabric0`/rail0）在 4 个数据量级下均能稳定跑出 ~47-48 GB/s（约 380 Gb/s）的 GPU-to-GPU 带宽，数值随数据量增大几乎不变，说明已经打满该 NIC 链路带宽上限，没有随包大小出现明显的启动开销或抖动。**

---

<a id="sec-6"></a>
## 6. 实验四：PyTorch AllReduce（双 NIC，8 GPU/Pod）
[↑ 回到目录](#toc)

目标：2 个 Pod 之间用 `mlx5_0`+`mlx5_1` 两张 RDMA NIC，每个 Pod 用满 8 张 GPU，做 AllReduce，测 1/2/4/8 GB。

### 6.1 思路与运行方式

- `world_size=16`（2 Pod × 8 GPU），用 `torch.distributed` + NCCL backend，每个 rank 对 1/2/4/8 GB 的 float32 tensor 做 3 次 warmup + 5 次 `dist.all_reduce(SUM)`，按 nccl-tests 惯例同时报 `algbw`(=size/time) 和 `busbw`(=algbw×2×(n-1)/n，16 卡 ring all-reduce 的有效总线带宽)。
- 用 `torchrun` 启动（`--nnodes=2 --nproc_per_node=8`），两个 Pod 分别跑对应 `node_rank` 的脚本，`MASTER_ADDR` 固定填 node_rank=0（pod0）的 eth0 IP `10.121.3.77`。
- 关键环境变量：`NCCL_SOCKET_IFNAME=eth0`（OOB）、`NCCL_IB_HCA=mlx5_0,mlx5_1`（数据面限定这两张对齐的 NIC）、`NCCL_IB_GID_INDEX=3`、`NCCL_CROSS_NIC=0|1`（本节重点对比项）。

完整代码见附录 11.5（`allreduce_bw_test.py` + `run_allreduce_pod0.sh` + `run_allreduce_pod1.sh`）。

### 6.2 结果：`NCCL_CROSS_NIC=0`（rail-optimized 默认，按 rail 严格对齐）

```
NCCL INFO NCCL_IB_HCA set to mlx5_0,mlx5_1
NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [1]mlx5_1:1/RoCE [RO]; OOB eth0:10.121.3.77<0>
```

| 数据量 | 平均耗时 | algbw | busbw | busbw (Gb/s) |
|---|---|---|---|---|
| 1 GB | 424.55 ms | 2.53 GB/s | 4.74 GB/s | 37.94 |
| 2 GB | 833.67 ms | 2.58 GB/s | 4.83 GB/s | 38.64 |
| 4 GB | 1649.85 ms | 2.60 GB/s | 4.88 GB/s | 39.05 |
| 8 GB | 3297.98 ms | 2.60 GB/s | 4.88 GB/s | 39.07 |

### 6.3 结果：`NCCL_CROSS_NIC=1`（允许 channel 间自由切换 NIC，对比项）

| 数据量 | 平均耗时 | algbw | busbw | busbw (Gb/s) |
|---|---|---|---|---|
| 1 GB | 418.29 ms | 2.57 GB/s | 4.81 GB/s | 38.51 |
| 2 GB | 826.20 ms | 2.60 GB/s | 4.87 GB/s | 38.99 |
| 4 GB | 1640.47 ms | 2.62 GB/s | 4.91 GB/s | 39.27 |
| 8 GB | 3277.36 ms | 2.62 GB/s | 4.91 GB/s | 39.31 |

**`CROSS_NIC=0` 和 `CROSS_NIC=1` 结果几乎完全一样（busbw 都在 ~4.8-4.9 GB/s），说明这个参数在本次场景下不是瓶颈所在。**

### 6.4 关键发现：瓶颈是拓扑结构（8 GPU 共享 2 NIC，且这 2 NIC 只归属 GPU0），不是 NCCL_CROSS_NIC

从 NCCL 日志看，跨节点的网络 channel 只出现在 **rank0↔rank8、rank1↔rank8/rank9** 这类"网关 GPU"之间（例如 `Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/2/GDRDMA`），其余 6 张 GPU 的跨节点流量都是先通过 `P2P/CUMEM`（NVLink）汇聚到这 1-2 张"网关 GPU"，再通过 RDMA NIC 出网。

**根因（呼应第 1.5 节 PCI 拓扑）**：这套硬件是严格的 rail-aligned 设计，`GPU[n]` 只和 `mlx5_[2n]`/`mlx5_[2n+1]` 同一个 PCIe Root Complex（`PXB` 距离），到其它 NIC 都要跨 NUMA 走 `SYS` 距离。也就是说 **`mlx5_0`/`mlx5_1` 在物理上只和 GPU0 同根**，跟 GPU1~7 完全不共享 Root Complex。所以本次测试里，GPU0 是唯一真正"零跳"直连这两张 NIC 的 GPU，其余 7 张 GPU 必须先经 NVLink 跳到 GPU0 所在的 PCIe 域，再由 GPU0 转发出网——标准 rail-optimized（1 GPU : 1 对专属 NIC）拓扑，在本次只用 `mlx5_0`/`mlx5_1` 两张网卡服务全部 8 张 GPU 的场景下，退化成了「7 张 GPU 靠 NVLink 转发 + 1 张 GPU 独占出网带宽」的结构。

这解释了为什么本节 AllReduce 的 busbw（~4.9 GB/s）远低于第 5 节单 NIC 点对点测试的 ~48 GB/s：AllReduce 的总线带宽受限于「NVLink 汇聚到 GPU0 + 仅 GPU0 出网」这条串行路径，而不是 NIC 本身的物理带宽上限，`NCCL_CROSS_NIC` 调不调整都改变不了这个结构性瓶颈（该参数只影响 NCCL 是否允许一个 channel 内切换 NIC，不能让 GPU1~7 凭空获得直连 NIC 的物理通路）。

**结论：**
1. 两张 RDMA NIC（`mlx5_0`/`mlx5_1`）确实都被 NCCL 识别并注册使用（日志证据充分），16 卡 AllReduce 功能验证通过，4 个数据量级全部成功。
2. 但受限于本次实验环境「一台机器坏了 NIC，只能用 2 张对齐的 NIC 服务 8 张 GPU」（见第 0.3 节），且这 2 张 NIC 在 PCIe 拓扑上只跟 GPU0 同根，AllReduce 的总线带宽被压低到 ~4.9 GB/s，**这个数字不代表 RDMA 硬件或 rail-optimized 方案本身的性能上限**，只反映了"标准 8 GPU:8 NIC（每 GPU 一对专属 NIC）拓扑退化为 8 GPU:2 NIC（且这 2 张只归属 GPU0）"后的必然结果。
3. `NCCL_CROSS_NIC=0` vs `1` 在本场景下没有实质差异，不需要为了这个瓶颈去调整该参数——真正的解法是恢复坏掉的 NIC，让每张 GPU 都能用上自己物理同根的那一对网卡。

如果你的环境每张 GPU 都能配到自己同根的那对 NIC（标准 1:1 rail 映射），预期 busbw 应该显著高于这里的 ~4.9 GB/s——建议在那种环境下重跑一次本节测试作对比。

---

<a id="sec-7"></a>
## 7. 清理环境
[↑ 回到目录](#toc)

做完第 2~6 节的所有实验后，删除测试负载，验证网卡能否正确归还主机。这一步也是第 8 节 MPI Operator 实验的前提——`test-workload.yaml` 的 2 个 Pod 各占满 8 张 GPU，跟 MPIJob 的 Worker 资源请求冲突，必须先清理腾出 GPU。

```bash
kubectl delete -f test-workload.yaml
```

```
$ kubectl delete -f test-workload.yaml
statefulset.apps "8-gpu-2-fabric-pod" deleted

$ kubectl get statefulset,pods -l app=8-gpu-2-fabric-pod
No resources found in default namespace.
```

**验证两台主机上 `fabric0`/`fabric1` 是否恢复：**

```
# 接口恢复，重新挂回对应 VRF（对比第 2.4 节：删除前这里完全看不到 fabric0/fabric1）
$ ssh root@134.209.13.101 "ip -d link show fabric0"
4: fabric0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 4200 ... master vrf-fabric0 state UP ...
    vrf_slave table 100 ... parentbus pci parentdev 0000:86:00.0
    altname enp134s0f0

# IP 地址保留——跟 Pod 里用的完全一样（IP-preserving 双向都生效：进 Pod 保留、还主机也保留）
$ ssh root@134.209.13.101 "ip -6 addr show fabric0"
inet6 fd02:0:0:58::1/64 scope global

# VRF 路由表恢复，包括跨节点的 fd02::/32 路由
$ ssh root@134.209.13.101 "ip -6 route show vrf vrf-fabric0"
fd02:0:0:58::/64 dev fabric0 proto kernel metric 256 pref medium
fd02::/32 via fd02:0:0:58::2 dev fabric0 proto static metric 1024 onlink pref medium
fe80::/64 dev fabric0 proto kernel metric 256 pref medium
multicast ff00::/8 dev fabric0 proto kernel metric 256 pref medium

# 134.209.13.101 上 fabric1 / 138.68.59.208 上 fabric0、fabric1 的结果与上面完全对称（IP 分别恢复为
# fd02:0:0:59::1 / fd02:0:0:178::1 / fd02:0:0:179::1，各自 VRF 路由表也全部恢复）
```

**结论：删除 `test-workload.yaml` 后，`fabric0`/`fabric1` 被 CNI 插件干净地从 Pod netns 归还到主机默认网络命名空间，接口、VRF 挂载（`master vrf-fabric0`/`vrf-fabric1`）、IP 地址、路由（含跨节点的 `fd02::/32` 路由）全部恢复到第 1.6 节部署前的状态，没有残留或损坏。这也验证了 `ip-preserving-host-device` 插件"归还时保留 IP"的设计在双向（分配进 Pod、回收回主机）都正确生效。**

如果只是想清理 CNI 本身（不打算再用这套方案），可以再执行 `kubectl delete -f network-attachments.yaml` 和 `kubectl delete -f daemonset.yaml`。

---

<a id="sec-8"></a>
## 8. 实验五：MPI Operator AllReduce（nccl-tests，双 NIC，8 GPU/Pod）
[↑ 回到目录](#toc)

在第 7 节清理掉 `test-workload.yaml`、腾出 GPU 之后，接着做一次独立实验：不再手写 PyTorch 脚本，改用社区标准的 [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator) + [`nccl-tests`](https://github.com/NVIDIA/nccl-tests) 官方 `all_reduce_perf` 二进制，跑一次跟第 6 节同样规模（2 Pod × 8 GPU，2 张 RDMA NIC）的 AllReduce 测试。这个实验用的是 MPIJob 自己创建的 Pod（不依赖 `test-workload.yaml`），跟前面章节的资源完全独立。

### 8.1 部署 MPI Operator

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

### 8.2 MPIJob 定义与代码审查

拿到一份现成的 `allreduce-2-b300x8-2-fabric.yaml`（完整最终版见附录 11.6），核心结构：

- `slotsPerWorker: 8` + `Worker.replicas: 2` → 总共 16 个 slot（对应 16 张 GPU）。
- `Launcher` 用 `mpirun` 通过 SSH（端口 2222）远程拉起 `Worker` 上的 `/opt/nccl-tests/build/all_reduce_perf`。
- `Worker` 挂载 `roce-net-fabric0@net0`、`roce-net-fabric1@net1`（跟前面章节的 `pod-fabric0/1` 是同一套 NAD，只是接口改名叫 `net0/net1`），资源请求 `nvidia.com/gpu: 8` + `rdma/fabric0: 100` + `rdma/fabric1: 100`。
- 镜像用 `ghcr.io/coreweave/nccl-tests:13.2.0-devel-ubuntu24.04-nccl2.29.7-1-7112046`（官方预编译好 `nccl-tests` + MPI 的镜像）。

### 8.3 部署、查看结果与清理

```bash
kubectl apply -f allreduce-2-b300x8-2-fabric.yaml
```

**查看 MPIJob / Pod 状态**：

```
$ kubectl get mpijob
NAME                AGE
mpi-multus-b300x8   19s

$ kubectl get pods -o wide | grep mpi-multus-b300x8
mpi-multus-b300x8-launcher-xxxxx   1/1     Running   0   19s   10.121.0.xx    rs-cpu-pool-3y0p31
mpi-multus-b300x8-worker-0         1/1     Running   0   19s   10.121.3.xx    rs-gpu-pool-3xnkkk
mpi-multus-b300x8-worker-1         1/1     Running   0   19s   10.121.2.xx    rs-gpu-pool-3xnkk5
```

**查看测试结果**：`all_reduce_perf` 的输出全部打在 `Launcher` Pod 的日志里（`Worker` 只是被 SSH 远程拉起执行，输出通过 `mpirun` 汇总回 `Launcher`），所以只需要看 `Launcher`：

```bash
kubectl logs -f mpi-multus-b300x8-launcher-xxxxx   # 实时跟随
kubectl logs mpi-multus-b300x8-launcher-xxxxx      # 跑完之后一次性看完整日志
```

`Launcher` 容器执行完 `mpirun` 就会退出，对应 Pod 状态会变成 `Completed`（`0/1 Completed`），这是正常现象，不是报错；如果看到 `mpirun detected that one or more processes exited with non-zero status` 或 `Test NCCL failure` 才是真的失败。`runPolicy.cleanPodPolicy: Running` 这个配置会让 `Worker` Pod 在作业结束后自动被清理掉（变成 `Terminating`），`Launcher` 的 `Completed` Pod 会保留，方便事后用 `kubectl logs` 回看结果。

**清理测试任务**：

```bash
kubectl delete -f allreduce-2-b300x8-2-fabric.yaml
```

删除 `MPIJob` 资源会把 `Launcher`（即使已经是 `Completed`）和所有 `Worker` Pod 一起删掉。想重跑一次（比如改完 `NCCL_CROSS_NIC` 之类的参数），必须先 `delete` 干净再 `apply`，不能对同名 `MPIJob` 直接重新 `apply`（`Launcher`/`Worker` 是一次性跑到完成的 Pod，不会自动重启复用）。

### 8.4 调试过程：两个独立的报错，两个不同层面的根因

部署之后遇到了两轮完全不同的报错，分别出在 MPI 层和 NCCL 层：

#### 第一轮：MPI/UCX 层 —— `uct_iface_open(tcp/net0) failed: Input/output error`

```
[mpi-multus-b300x8-worker-1:996  :0]      ucp_worker.c:1415 UCX  ERROR uct_iface_open(tcp/net0) failed: Input/output error
[mpi-multus-b300x8-worker-1:01000] pml_ucx.c:314  Error: Failed to create UCP worker
...
mpirun noticed that process rank 12 ... exited on signal 11 (Segmentation fault).
```

- **根因**：Open MPI 默认会用 UCX 作为 PML，UCX 库初始化时会枚举容器里所有网络接口，包括 `net0`（`fabric0`/`mlx5_0`）。但这张网卡按第 1/2 节反复验证的结论——enslave 到独立 VRF，没有配成普通 TCP socket 能用的接口——UCX 对它开 TCP transport 直接报 I/O 错误，进而整个 UCP worker 创建失败，最终段错误崩溃。
- **排查中证明"猜错了"的尝试**（记录下来避免以后重复踩）：单独加 `-mca pml ^ucx`（禁用 UCX 作为 PML）**没用**——报错完全一样。原因：这只是告诉 Open MPI "消息传递别选 UCX"，并不能阻止 UCX 库自己的设备探测逻辑在初始化时运行，这个探测不受 PML 选择开关控制。
- **真正生效的修复**：`-x UCX_NET_DEVICES=eth0` —— 这是 UCX 库自己的环境变量，直接告诉它"只看得到 eth0"，从根源上不让它去碰 `net0`/`net1`，不依赖 Open MPI 间接开关。

#### 第二轮：NCCL 层 —— `ibv_create_ah failed: No such device`

修好第一轮之后，MPI 层通过了，但 NCCL 初始化阶段又报：

```
[5] ibvwrap.cc:205 NCCL WARN Call to ibv_create_ah failed with error No such device
[5] devx_utils.cc:260 NCCL WARN devx_utils.cc:260 Call to wrap_ibv_create_ah(&ah, pd, &attr) failed: 2
mpi-multus-b300x8-worker-0: Test NCCL failure all_reduce.cu:451 'unhandled system error...'
```

固定出现在 local rank 5/6/7（即 GPU5/6/7）。排查过程：
- 先怀疑 `NCCL_NET_DISABLE_INTRA=1`（强制节点内通信不走网络）会不会让没有直连 NIC 的 GPU（回顾第 1.5 节 PCI 拓扑：只有 GPU0 跟 `mlx5_0`/`mlx5_1` 同根）被迫尝试走网络——去掉这个变量重跑，**报错完全没变**，排除。
- 打开 `NCCL_DEBUG=INFO` 看详细日志，找到真正线索：

  ```
  NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [1]mlx5_1:1/RoCE [2]mlx5_10:1/RoCE [3]mlx5_11:1/RoCE
                      [4]mlx5_12:1/RoCE [5]mlx5_13:1/RoCE [6]mlx5_14:1/RoCE [7]mlx5_15:1/RoCE
                      [8]mlx5_16:1/IB/SHARP [9]mlx5_17:1/IB/SHARP [10]mlx5_18:1/IB/SHARP [11]mlx5_19:1/IB/SHARP
  ```

- **根因**：`NCCL_IB_HCA=mlx5_0,mlx5_1` 本意只用这两张卡，但 NCCL 对这个变量的名字匹配默认是**前缀匹配**，`"mlx5_1"` 恰好是 `mlx5_10`~`mlx5_19`（4 个跟 GPU 不同根、Pod 也没被授权访问的 ConnectX-7 管理端口）的字符串前缀，全被误匹配进设备列表。某些 GPU 的 channel 被分配到这些"幽灵设备"上，建地址句柄（AH）时自然找不到设备。
- **修复**：用 `=` 强制精确匹配，且**只在整个列表最前面加一个 `=`**（不是每个设备名前各加一个）：

  ```
  NCCL_IB_HCA==mlx5_0,mlx5_1
  ```

  第一次尝试写成 `=mlx5_0,=mlx5_1`（每项各加 `=`）不报错，但日志显示所有 rank 都只注册了 `mlx5_0`，`mlx5_1` 被丢掉了——说明这个精确匹配标志是"整体列表加一次"的语义，不是"逐项加"。改成单个前导 `=` 后，两张卡才都正确出现在 `NET/IB : Using [0]mlx5_0:1/RoCE [1]mlx5_1:1/RoCE` 里。

### 8.5 最终结果

修好以上问题、且保留 `NCCL_NET_DISABLE_INTRA=1`（确认跟本次故障无关，重新打开后性能更好）之后，测试稳定跑通，`Out of bounds values: 0 OK`。

**`NCCL_CROSS_NIC=0`**：

| size | busbw (GB/s) |
|---|---|
| 8.4 MB | 36.15 |
| 134 MB | 84.88 |
| 2 GB | 89.69 |

`Avg bus bandwidth: 26.8761 GB/s`

**`NCCL_CROSS_NIC=1`（对比项）**：

| size | busbw (GB/s) |
|---|---|
| 8.4 MB | 37.61 |
| 134 MB | 84.68 |
| 2 GB | 89.77 |

`Avg bus bandwidth: 27.0751 GB/s`

两组几乎完全一样，跟第 6.3/6.4 节 PyTorch AllReduce 测试的结论一致：**`NCCL_CROSS_NIC` 在这套"8 GPU 共享 2 NIC、且这 2 NIC 只归属 GPU0"的拓扑下不是瓶颈**，调不调整都一样。

有意思的是，这次用 nccl-tests 官方二进制跑出来的 busbw（~27-90 GB/s，视 size 而定）明显高于第 6 节 PyTorch 脚本跑出来的数字（~4.9 GB/s）。合理的解释是两者的 warmup/迭代次数、消息大小扫描方式、以及 nccl-tests 官方实现对多 size 的处理方式不同，具体差异未继续深挖——**再次强调本文档第 0.3 节的前提：所有性能数字都不具备严格参考价值，只用于验证功能连通性**，不同测试工具之间的数字不能直接跨表比较。

### 8.6 拓扑深挖：16 张 GPU 到底是怎么组环、怎么用这 2 张 NIC 的

前面第 1.6/6.4 节已经从 PCI 拓扑角度推断出"只有 GPU0 跟 `mlx5_0`/`mlx5_1` 同根"，这里借着 `NCCL_DEBUG=INFO` 的完整日志，实际验证一下 16 张 GPU 之间的数据到底怎么走。

**不是"每个 Pod 各自一个 ring、再单独桥接"，而是一个横跨 16 张 GPU 的单一大环。** NCCL 打印出的完整环序（`Channel 00/08`）：

```
0  7  6  5  4  3  2  1  8  15  14  13  12  11  10  9   （9 之后绕回 0，闭环）
```

拆解开看：

**1. 节点内部分（NVLink，`P2P/CUMEM`）**：
- worker-0 内部：`0→7→6→5→4→3→2→1`（7 段 NVLink 直连）
- worker-1 内部：`8→15→14→13→12→11→10→9`（同样 7 段 NVLink 直连）

**2. 跨节点部分（RDMA，`GDRDMA`）——整个环只在两个点跨节点**：边 A `1[1]→8[0]`，边 B `9[1]→0[0]`。日志证据：

```
worker-0 [1] Channel 00/0 : 1[1] -> 8[0] [send] via NET/SPCX/2(0)/GDRDMA
worker-1 [0] Channel 00/0 : 1[1] -> 8[0] [receive] via NET/SPCX/2/GDRDMA
worker-1 [1] Channel 00/0 : 9[1] -> 0[0] [send] via NET/SPCX/2(8)/GDRDMA
worker-0 [0] Channel 00/0 : 8[0] -> 0[0] [receive] via NET/SPCX/2/GDRDMA
```

**关键细节**：这两条跨节点边表面上是 GPU1（rank1、rank9）在发起，但它们自己并不直连 NIC——日志里的 `(0)` 和 `(8)` 是 NCCL 的 **PXN（PCI×NVLink）转发标注**：GPU1 先把数据通过 NVLink 转给同节点的 GPU0（`(0)` = 转发给本地 rank0，`(8)` = 转发给本地 rank8），再由 **GPU0 拿着物理网卡真正把数据发出去**。只有 GPU0↔GPU0 这一对（rank0↔rank8）是真正直连 NIC、日志里没有代理标注的。

**结论**：16 卡组成一个环，环内 7 段走 NVLink，环在两个节点边界处各断开一次形成 2 条跨节点边；这 2 条边表面上分别由 GPU1 发起，但两条边的数据最终都要先经 NVLink "借道"本节点的 GPU0，再由 GPU0 通过这 2 张 RDMA NIC（在这套 NCCL 构建里合并显示成一个虚拟设备索引 `NET/SPCX/2`）真正发到对端——跟第 6.4 节"GPU0 是唯一真正的出网网关"这个结论完全吻合，这里只是多了一层"具体怎么借道"的实际日志证据。

### 8.7 本节踩坑小结（并入第 9 节速查表）

| 现象 | 试过但无效的方案 | 真正有效的修复 |
|---|---|---|
| UCX 报 `uct_iface_open(tcp/net0) failed` | `-mca pml ^ucx` | `-x UCX_NET_DEVICES=eth0` |
| NCCL 报 `ibv_create_ah failed: No such device`（固定在没有直连 NIC 的 GPU 上） | 去掉 `NCCL_NET_DISABLE_INTRA=1` | `NCCL_IB_HCA` 用单个前导 `=` 强制整体精确匹配：`NCCL_IB_HCA==mlx5_0,mlx5_1` |

---

<a id="sec-9"></a>
## 9. 踩坑速查表
[↑ 回到目录](#toc)

| 现象 | 根因 | 解法 / 参考章节 |
|---|---|---|
| Pod 内 `ib_write_bw` 直接用 fabric 地址报 `Network is unreachable` | fabric 网卡 enslave 到独立 VRF，跨节点路由只在 VRF 表里，不在 `main` 表 | 4.1/4.2 节：OOB 走 `eth0`，数据面走 `-d mlx5_x` 的 GID |
| 主机/Pod 上不带 `-I`/不绑设备直接 ping 远端地址失败 | 同上，`l3mdev-table` 规则要靠出口设备才能生效 | 1.6 节 |
| PyTorch `dist.init_process_group` 卡死不报错 | `MASTER_ADDR` 填成了非 rank0 的 IP，TCPStore server 没建在预期地址上 | 5.2 节：`MASTER_ADDR` 必须是 rank0 所在 Pod 的 IP |
| 主机上 `show_gids`/裸读 sysfs 查不到已分配给 Pod 的网卡 GID，但 `ibv_devinfo -v` 能查到 | 网络层（netdevice/IP/路由）被 Pod 独占；RDMA verbs 层在 `netns shared` 模式下全局共享 | 2.5 节 |
| 多 GPU AllReduce 带宽远低于单 NIC 点对点测试 | NIC:GPU 比例不是标准 1:1（本例 2:8），且这些 NIC 物理上只跟其中 1 张 GPU 同根 PCIe Root Complex | 1.6/6.4 节 |
| 同一台主机上两条 rail（如 fabric0/fabric1）互相 ping，TTL 显示经过了 1 跳 | 两条 rail 各属独立 VRF，本地没有路由捷径，必须出网线到 leaf 交换机绕一圈再回来 | 1.6 节 |
| MPI Operator 里 `mpirun` 一提交就报 `not enough slots` | `-np` 数量跟 `slotsPerWorker × Worker replicas` 不匹配（例如从 4 节点配置改成 2 节点忘了同步改 `-np`） | 把 `-np` 改成等于 `slotsPerWorker × Worker replicas` |
| MPI Operator 里 UCX 报 `uct_iface_open(tcp/net0) failed`，进程段错误 | UCX 库自身设备探测会枚举 VRF 隔离的 fabric 接口，不受 `-mca pml ^ucx` 控制 | 8.4 节：`-x UCX_NET_DEVICES=eth0` |
| MPI Operator 里 NCCL 报 `ibv_create_ah failed: No such device`，固定在某几个 GPU 上 | `NCCL_IB_HCA=mlx5_0,mlx5_1` 前缀匹配误把 `mlx5_10`~`mlx5_19` 也匹配进设备列表 | 8.4 节：改成单个前导 `=` 精确匹配 `NCCL_IB_HCA==mlx5_0,mlx5_1` |

---

<a id="sec-10"></a>
## 10. 待办 / 下一步
[↑ 回到目录](#toc)

- **换到真正的生产环境测试，而不是本文档这套模拟环境**：本文档所有性能数字都受限于第 0.3 节说明的模拟环境限制（坏了一个 NIC、只能凑出 2 张对齐的 rail），不具备参考价值，需要在真实的 rail-optimized 集群上重新跑一遍才能拿到有意义的数据。
- **大规模测试**：更多节点、用满每台机器全部的 RDMA NIC（而不是像本文档这样只用 2/16 张），验证标准 1:1（甚至更多）rail 映射下的真实吞吐，也能测出跨节点、跨 rail 在更大规模下的扩展性。
- **覆盖更多通信框架**，不只局限于 NCCL：
  - Mooncake Transfer Engine
  - NIXL / UCX
  - NCCL（本文档已覆盖，后续在真实环境下复测）
- **上层分布式推理场景的应用测试**，而不只是底层通信原语的连通性/带宽验证：
  - PD 分离（Prefill-Decode disaggregation）
  - KV Cache Sharing
  - Wide EP（专家并行）

---

<a id="sec-11"></a>
## 11. 附录：完整 YAML 与脚本代码
[↑ 回到目录](#toc)

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

### 11.4 `gpu_rdma_bw_test.py`（第 5 节：单 NIC GPU-to-GPU 测试）

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

### 11.5 第 6 节：AllReduce 测试代码

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

#### `run_allreduce_pod0.sh`（在 `8-gpu-2-fabric-pod-0` 上运行，`node_rank=0`）

```bash
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
```

#### `run_allreduce_pod1.sh`（在 `8-gpu-2-fabric-pod-1` 上运行，`node_rank=1`）

```bash
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
```

（对比 `NCCL_CROSS_NIC=1` 时，两个脚本里把 `NCCL_CROSS_NIC=0` 改成 `1`、`MASTER_PORT` 换一个新端口即可，其余不变。）

### 11.6 第 8 节：MPI Operator 最终版 `allreduce-2-b300x8-2-fabric.yaml`

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

（对比 `NCCL_CROSS_NIC=1` 时，把 `NCCL_CROSS_NIC=0` 改成 `1` 即可，其余不变。）
