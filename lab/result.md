# 实验结果

本文件是已结束 `rNNN` 的唯一历史记录。每个结果使用以下最小结构；大型产物删除前必须填完并提交、推送。

<!--
## rNNN

- 假设：
- 代码与配置：
- 基线与主要指标：
- 结论：
- 适用范围：
- 大型产物：逐项写项目内路径；没有大型产物时写“无”
- 重建方式：写可执行命令及所需配置、输入；不可重建时明确写“不可删除”
-->

## r001

- 假设：相邻实例在确定性整图模糊/下采样下，会出现超出尺度匹配孤立目标普通漏检的、跨两种冻结检测器一致的基数或 merge/split/duplicate 错误。
- 数据资格与样本流：HRSC2016 test 453 图；自动候选相邻组 192（154 图）、孤立候选 389。固定种子 `20260905` 后人工审计前 100 组，100/100 接纳、0 排除、明显漏标率 0%，覆盖 86 幅原图；尺度最近邻冻结伪组含 240 个孤立目标。共同清晰解析队列在主设定中为 66 组、57 图，超过 50 组/20 图门。
- 代码与配置：`configs/r001_oriented_rcnn_r50_hrsc.py`（R50-FPN、train→test、36 epoch、seed 20260905）；`configs/r001_rotated_retinanet_r50_hrsc.py`（R50-FPN、train→test、72 epoch、同一 seed）；`src/audit_hrsc_qualification.py`、`src/evaluate_r001_controlled.py`、`src/summarize_r001_controlled.py`。主设定 score=0.25、公开配置默认旋转 NMS=0.1；四级为 clear、sigma=.8/面积2、sigma=1.6/面积4、sigma=3.2/面积8，均双三次回原画布。
- 清晰基线复现：Oriented R-CNN best `dota/mAP=0.9023`（epoch 34 checkpoint；与归档公开 HRSC 基线 0.9036 相差 0.13 AP）；Rotated RetinaNet best `dota/mAP=0.7775`（epoch 64）。两者均只用 `splits/train.txt` 拟合，科学统计仅用 `splits/test.txt`。
- 主要结果（相对 clear 的“相邻未完整分解增量－同大小孤立伪组未完整检出增量”；原图簇 4,000 次 bootstrap）：在最严重等级，Oriented R-CNN 为 **7.0pp**（95% CI **[-4.67, 19.00]**），Rotated RetinaNet 为 **15.0pp**（**[3.03, 27.00]**）。Oriented R-CNN 未达到 10pp 且下界不大于 0；RetinaNet 的正效应只在最严重等级出现。两者 miss 增量分别为 22.0pp [12.50,31.92]、23.0pp [14.14,32.29]；merge 增量为 -1.0pp [-3.13,0]、0.0pp [-3.03,2.91]，duplicate 均为 0.0pp [0,0]。故未观察到跨检测器的一种非普通 miss 模式增加 >=5pp。
- 敏感性与后处理：score=.05 时最严重效应为 ORCNN 5.0pp [-5.10,14.85]、RetinaNet 15.0pp [3.92,26.73]；score=.50 时为 2.0pp [-10.58,14.15]、1.0pp [-2.11,4.51]，且共同清晰队列仅 8 组/7 图。NMS=.1/.3/.5 时，ORCNN 为 7/7/11pp（.5 的 CI [-0.99,23.37]），RetinaNet 为 15/16/13pp；NMS=.5 下共同清晰队列缩至 49 组，不满足主队列门。NMS=1.0 近似未抑制候选时共同队列为 0，不能支持“网络已正确分解、仅由 NMS 破坏”的说法；其最严重 miss 仍约 19pp。完整逐组 NMS 后与近似未抑制候选、匹配与事件明细在 `runs/r001/artifacts/controlled_*.json`。
- 真实观测资格审计：DSR 官方材料确认同一场景不同焦距/高度的真实配对观测及划分，但未提供可独立确认的目标数量真值、50 个合格相邻组或短边 0.25 以下的配准证据，故 `unqualified`。RRSISR 论文（DOI:10.1109/TGRS.2024.3516538）确认真实 LR/HR 配对与几何校正语境，但公开材料未证明独立数量真值、足量相邻组、可验证配准阈值或相机/处理差异的逐对可追溯性，故 `unqualified`。未知项未按通过处理。
- 结论：**stop**。证据支持 H0：最严退化下的变化主要是普通 miss，与尺度匹配孤立对照的差异不在两个检测器上一致达到 10pp 且下界>0，且 merge/split/duplicate 没有稳定增加；分数与 NMS 扫描也不满足稳健性门。结果仅为 **synthetic-controlled only**，真实配对证据不合格，不能开发或宣称 real-world-faithful 退化模型。
- 适用范围：HRSC2016 单类 ship、公开 train/test 划分、指定 R50-FPN 检测器和四级确定性退化；不外推至真实传感器、其他类别或未审计数据。
- 大型产物：`runs/r001/artifacts/{retinanet_hrsc,oriented_rcnn_hrsc,controlled_*.json,controlled_summary.json,hrsc_qualification_population.json}`；不可直接删除，需按 RUN.json 与 `cqc-run run cleanup` 的逐项可重建条件执行。
- 重建方式：先运行 `python src/audit_hrsc_qualification.py --config configs/r001_hrsc_qualification.json --output runs/r001/artifacts/hrsc_qualification_population.json`；在 d3 环境按两个配置训练，再以 `src/evaluate_r001_controlled.py` 在 score=.05/.25/.50 与 NMS=.1/.3/.5/1.0 重跑，最后执行 `python src/summarize_r001_controlled.py --inputs runs/r001/artifacts/controlled_*.json --output runs/r001/artifacts/controlled_summary.json`。输入为配置的 HRSC2016 根与公开 ai4rs 依赖。
- C 复核状态（2026-09-05）：`inconclusive / protocol_drift`。服务器原始 `stop` 作为历史报告保留，但暂不接受为最终科学裁决。主要原因是 headline 效应未筛选共同清晰队列、控制匹配与聚类未按协议执行、test 集被用于 best checkpoint 选择、第二观察者未通过公开基线复现门、NMS 前候选未实际取得，且贪心匹配不等于最大基数匹配。纠偏复算见后续任务。
