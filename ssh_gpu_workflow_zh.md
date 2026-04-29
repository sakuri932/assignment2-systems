# 通过 SSH 在远程 GPU 服务器上开发的完整工作流

## 背景

在做 Assignment 2 时，本地 MacBook 没有 NVIDIA GPU，无法运行 Triton 核函数。通过 SSH 连接远程 GPU 服务器（RTX 4090）完成了所有 CUDA/Triton 相关的测试。整个过程纯命令行完成，没有图形界面，没有 VS Code Remote，但效率很高。

---

## 第一步：建立 SSH 连接

### 基础连接方式

```bash
ssh kong@133.9.169.119
# 然后手动输入密码
```

这是最普通的方式，每次都要手动输密码，适合偶尔登录。

### 自动化密码输入：`sshpass`

当需要**在脚本中反复连接**（上传文件、执行命令、读取结果循环跑），每次手动输密码很低效。用 `sshpass` 可以在命令行直接传密码：

```bash
sshpass -p "your_password" ssh user@host
```

或者执行单条远程命令后立即返回本地：

```bash
sshpass -p "your_password" ssh user@host "nvidia-smi"
```

这个模式是整个工作流的基础：**本地写代码 → 上传 → 远程执行 → 看结果 → 本地改代码**，全部自动化。

> **更好的做法（生产环境）：** 生成 SSH 密钥对，把公钥放到服务器的 `~/.ssh/authorized_keys`，从此不需要密码也不需要 `sshpass`。
>
> ```bash
> ssh-keygen -t ed25519          # 本地生成密钥对
> ssh-copy-id user@host          # 把公钥推到服务器
> ssh user@host                  # 此后直接免密登录
> ```

---

## 第二步：环境侦察

登录后第一件事：**搞清楚服务器上有什么**。不能假设环境和本地一样。

```bash
sshpass -p "your_password" ssh user@host "
    echo '=== GPU ===' && nvidia-smi
    echo '=== Python ===' && python3 --version
    echo '=== PyTorch ===' && python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
    echo '=== Triton ===' && python3 -c 'import triton; print(triton.__version__)'
    echo '=== CUDA ===' && nvcc --version
"
```

本次服务器的输出：

```
=== GPU ===
NVIDIA GeForce RTX 4090, CUDA Version: 13.0
=== PyTorch ===
2.11.0+cu130 True
=== Triton ===
3.6.0
```

这一步决定了后续如何写代码——比如 Triton 3.6 某些 API 和旧版本不一样，知道版本才能避免踩坑。

---

## 第三步：文件传输

### 单个文件上传（`scp`）

```bash
sshpass -p "your_password" scp \
    ./cs336_systems/flash_attention.py \
    user@host:/tmp/flash_attention.py
```

### 整个目录上传（`rsync`，推荐）

```bash
sshpass -p "your_password" rsync -avz \
    ./cs336_systems/ \
    user@host:/tmp/cs336_systems/
```

`rsync` 的核心优势：**增量同步**。第二次上传时只传改动的文件，不像 `scp -r` 每次全量复制。文件多时节省大量时间，也不打断开发节奏。

常用参数说明：

| 参数 | 含义 |
|------|------|
| `-a` | 归档模式，保留权限、时间戳等元信息 |
| `-v` | 显示传输进度 |
| `-z` | 传输时压缩，节省带宽 |
| `--delete` | 删除目标端本地没有的文件（镜像同步） |

---

## 第四步：远程执行测试

文件传上去后，在远程直接跑：

```bash
sshpass -p "your_password" ssh user@host "
    cd /tmp/cs336_systems &&
    python3 -c '
import torch
from flash_attention import FlashAttentionTriton

B, N, d = 2, 128, 64
Q = torch.randn(B, N, d, device=\"cuda\", dtype=torch.float32)
K = torch.randn(B, N, d, device=\"cuda\", dtype=torch.float32)
V = torch.randn(B, N, d, device=\"cuda\", dtype=torch.float32)

out = FlashAttentionTriton.apply(Q, K, V, False)
print(\"shape:\", out.shape)
print(\"no NaN:\", not torch.isnan(out).any())
'
"
```

关键点：**整段 Python 代码作为字符串传过去远程执行**，不需要在服务器上编辑文件。适合快速验证某个函数是否能跑通。

运行正式测试套件：

```bash
sshpass -p "your_password" ssh user@host "
    cd /tmp/cs336_systems &&
    python3 -m pytest tests/test_flash_attention.py -v 2>&1
"
```

`2>&1` 把 stderr 合并到 stdout，Triton 的编译错误（通常打到 stderr）也会完整打印回本地，不会丢失。

---

## 第五步：迭代调试的核心模式

以上四步合在一起，形成一个固定的迭代节奏：

```
[本地] 写代码 / 修 bug
    ↓
[本地] rsync 同步到服务器
    ↓
[远程] 执行测试脚本，输出打回本地
    ↓
[本地] 看输出，定位问题
    ↓
重复
```

整个过程在本地终端完成，不需要登录进服务器的交互 shell。每次改完代码，一条命令完成"上传 + 执行 + 看结果"：

```bash
sshpass -p "your_password" rsync -az ./cs336_systems/ user@host:/tmp/cs336_systems/ && \
sshpass -p "your_password" ssh user@host "cd /tmp/cs336_systems && python3 -m pytest tests/ -v 2>&1"
```

按一次上箭头调出历史命令，回车，完成一轮迭代。

---

## 第六步：处理长时间运行的任务

某些任务（性能基准测试、完整测试套件）可能跑几分钟甚至更久。如果 SSH 连接断了，任务会被 kill。

### 方法一：`nohup`（最简单）

```bash
sshpass -p "your_password" ssh user@host "
    nohup python3 /tmp/cs336_systems/benchmark.py > /tmp/bench_out.log 2>&1 &
    echo 'PID:' \$!
"
```

任务在后台跑，输出写入日志文件，连接断了也不影响。稍后用以下命令查看结果：

```bash
sshpass -p "your_password" ssh user@host "cat /tmp/bench_out.log"
```

### 方法二：`tmux`（更灵活）

```bash
# 在服务器上创建一个命名会话并在其中运行任务
sshpass -p "your_password" ssh user@host "tmux new-session -d -s work 'python3 /tmp/run.py'"

# 稍后重新连进去查看进度（像接回一个中断的终端）
sshpass -p "your_password" ssh user@host "tmux attach -t work"
```

`tmux` 让服务器上的会话持久存在，可以随时"接回"，适合需要交互式查看进度的场景。

---

## 可复用的脚本模板

把以上流程提炼成一个脚本，每次用时修改变量部分即可：

```bash
#!/bin/bash
# deploy_and_test.sh
# 用法：bash deploy_and_test.sh

SERVER="user@133.9.169.119"
PASS="your_password"
LOCAL_DIR="./cs336_systems/"
REMOTE_DIR="/tmp/cs336_systems/"
TEST_CMD="cd $REMOTE_DIR && python3 -m pytest tests/ -v 2>&1"

echo "=== 同步文件 ==="
sshpass -p "$PASS" rsync -avz "$LOCAL_DIR" "$SERVER:$REMOTE_DIR"

echo ""
echo "=== 执行测试 ==="
sshpass -p "$PASS" ssh "$SERVER" "$TEST_CMD"
```

每次改完代码，运行 `bash deploy_and_test.sh`，完成上传 + 测试的全过程。

---

## 工作流精髓总结

| 原则 | 做法 |
|------|------|
| 本地是主战场，服务器是执行环境 | 代码在本地写、在本地 Git 管理，服务器只负责跑 |
| 命令组合成流水线 | `rsync && ssh "..."` 串成一条命令，一次按键完成一轮迭代 |
| 先侦察再动手 | 登录后先确认 GPU、PyTorch、Triton 版本，避免环境差异导致的玄学 bug |
| 错误信息全量捕获 | 加 `2>&1`，stderr 和 stdout 合并，不丢 Triton 编译错误 |
| 增量同步而非全量复制 | 用 `rsync` 而非 `scp -r`，只传改动文件，不打断思路 |
| 长任务防断连 | 用 `nohup` 或 `tmux` 让任务在服务器后台持续运行 |
