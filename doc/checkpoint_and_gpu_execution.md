# Checkpoint 与双卡启动维护

`src/checkpoint_retention.py` 继承 MMEngine 原生 CheckpointHook，仅调整本地文件的发布与删除顺序。checkpoint 先写同目录临时文件、flush/fsync、原子替换；原生删除请求在该次保存全部成功后才执行。模型、optimizer、scheduler、message_hub 和 EMA 仍由原生 Runner/EMAHook 保存，AMP 如启用则保留原生 wrapper 的状态。本轮实际训练使用普通 OptimWrapper，不声称启用了 AMP。

每个训练臂/种子保留一个 latest；开发配置另按原 `r005/AP75`、greater 保留 best。确认阶段没有合法选优验证集，仍不设 best，固定最终 epoch 不变。结果 JSON 明确引用的 checkpoint 受到保护；引用损坏时暂缓删除，但不停止已成功保存的训练。恢复时不扫描或批量删除旧积存，后者只通过既有清理器处理。

`src/training_resources.py` 直接调用已安装的 `cqc_run.resources`。每次新入口选择两张卡，先要求每卡至少 8192 MiB 空闲，再按不同计算 PID 数升序、剩余显存降序排序；UUID 用于 CUDA 可见性，物理序号另行记录。8192 MiB 沿用原准入阈值，按保守 6144 MiB 配额加 2048 MiB 余量解释，6144 MiB 不是声称实测的峰值。未知进程信息不能当作空闲，不足两张卡不会退回单卡。全局 batch、采样器、EMA、优化器及科学预算不变。

2026-09-16 在 46 的 d3 环境用 CPU 执行：

```sh
python -m unittest src.test_checkpoint_retention src.test_training_resources -v
```

9 项检查通过，包括真实 Runner/EMAHook 序列化、优化器恢复、保存失败保留旧文件、AP75 best、显式下游依赖、旧积存不在恢复入口被扫除，以及公共 GPU 选择器的 PID 去重、零 PID 优先、显存过滤和 UUID 映射。另对 HRSC/DOTA × 四臂展开配置比较：除 checkpoint hook 与其导入之外，训练、数据和评测配置一致；开发展开配置仍以 AP75 选优。

实际已保存的 DOTA M1 seed17 epoch5、HRSC B2 seed17 epoch35、开发 B1 epoch36 均可在 CPU 完整读取，包含 264 项优化器状态、两个 scheduler、EMA、epoch/iter 与 message_hub。这只是恢复材料核验，不是新的科学评测。

部署不会改变已加载的 Python 对象。核验时当前双卡 rank PID 为 686643、686644；旧调度 PID 为 72604；既有等待入口 PID 1085730 会在旧调度自然结束后重新加载磁盘入口。新 hook 由后续训练子进程读取，新 GPU 选择逻辑由后续重新加载的调度入口读取。本轮没有重启、迁移或热改这些进程。

## 旧积存的恢复与清理

本次只考虑 HRSC/B2 seed17 的 epoch1–34、HRSC/M1 seed17 的 epoch1–35、DOTA/B2 seed17 的 epoch1–34，共103个文件。三组末态、现有结果引用的文件、全部取消种子、当前训练组、旧 unsharded 产物及开发 best/latest 均保留。候选的物理 inode 均不同，合计 16,450,464,061 bytes。

三份 `configs/r005_legacy_*.py` 原样来自保留 checkpoint 内的完整实际配置，包含真实优化器、scheduler、原生 EMA、数据、seed、分布式采样及历史保存策略。它们只用于按需恢复历史中间产物，不被当前训练入口引用。特别是两组 B2 原训练为35 epoch，不能用当前36 epoch预算替代其历史来源。`configs/r005_checkpoint_rebuild.json` 保存配置正文校验值、各组保留末态校验值、精确 epoch 清单和代码来源。

恢复命令使用原生 torchrun，每组实际两卡，并调用同一公共 GPU 选择器。仅在确需重建时执行：

```sh
/home/rspip/miniconda3/envs/d3/bin/python src/rebuild_r005_intermediates.py --config configs/r005_checkpoint_rebuild.json
```

恢复先使用新的工作目录，保留所请求的全部历史中间输出，再只恢复原位置缺失的指定文件，不覆盖任何原有末态、best或结果。需要不同恢复工作目录时传 `--output-root runs/r005/另一个未使用目录`。随机计算重训不保证字节一致，不可把重建权重伪称为原统计使用的权重。加 `--dry-run` 只检查配置并打印原生命令，本轮没有运行上述恢复训练。

`configs/r005.recovery.json` 使用既有 cqc-run 的恢复格式，旧描述保留于 Git 历史。代码没有消费它的旧自定义字段。只有 Home Git 已推送、输入存在、精确目标不属于保留依赖且清理器确认零进程引用后，才执行 `cqc-run run cleanup --project ... --recovery configs/r005.recovery.json --target ... --reason ...`；以清理器回写的逐文件记录为准。

2026-09-16 10:59 UTC，使用已发布的重复 `--target` 接口，先对103个精确目标全部 dry-run，再一次批量清理。删除记录提交为 `163ae51e100c178c677ffc249f7b30798c4f9b4f`，释放16,450,464,061 bytes。全项目 r005 的 PTH 清单由291个变为188个；所有非目标文件的 inode、大小、mtime 均保持不变，既有进程启动 ticks 不变。剩余文件包含受保留约束的取消种子产物、正在运行组及公共初始化缓存，不能把根目录总数解释成单臂保留上限。
