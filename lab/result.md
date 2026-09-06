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
- C 复核状态（2026-09-05，数据/权重再核对后修订）：`inconclusive / protocol_drift`。服务器原始 `stop` 作为历史报告保留，但暂不接受为最终科学裁决。硬性原因收敛为 headline 效应未筛选共同清晰队列、控制匹配与依赖聚类未按协议执行、NMS 前候选未实际取得，且贪心匹配不等于最大基数匹配。此前把 best checkpoint/test 评估单列为硬阻断过重：官方 HRSC 也采用 `trainval -> test` 并在训练流程中评估 test；r001 实为 `train -> test`，与官方 RetinaNet 参考训练集不同，故 7.05 AP 差不能单独判复现失败。主机 `pth_data` 已确认存在 valid 的 HRSC Oriented R-CNN 3x 权重（AP50 0.9036），没有 HRSC RetinaNet 条目；纠偏复算优先复用前者和 MMRotate 官方 RetinaNet 权重。纠偏复算见后续任务。

## r002

- 假设：不改变 r001 的样本、四级退化、检测器类别或继续门，只纠正共同清晰队列、全局协变量匹配、最大基数匹配、真实 NMS 前候选和依赖 bootstrap 后重新裁决。
- 回归证据：`cross_edge_maximum_cardinality`、`common_queue_isolation`、`shared_control_dependency`、`same_raw_candidates_across_nms`、`smd_gate` 五项反例均通过，记录在 `runs/r002/artifacts/r002_result.json`。
- 观察者资格：Oriented R-CNN 的指定 valid 归档权重为 `trainval.txt -> test.txt`、报告 AP50=0.9036；当前唯一 ai4rs/d3 推理栈在 `torch.load` 时失败，原始错误为 `ModuleNotFoundError: numpy._core`。尝试兼容模块别名后触发 NumPy C-ABI `SystemError`，不能把该权重安全加载。候选 qhl 环境有 NumPy 1.26.4，但没有 mmengine/mmrotate/mmdet，不能形成兼容观察者。官方 RetinaNet v0.1.0 `trainval.txt -> test.txt` 权重下载、SHA-256 `ee4f18af46cb4057dbbc84c5753b86058bc75eb91c04ba7af3f8ee01d9dd1142`，已在 d3 通过 state-dict、`le90`、标准预处理及单图推理预检；但两观察者门要求同时满足。
- 样本流：继承已审计相邻组 100（86 图）；排除相邻组源图后的孤立候选为 356。因 ORCNN 观察者兼容门失败，未生成任何退化候选、控制匹配、共同队列或效应统计；这不是零效应。
- 运行与 provenance：`cqc-run` 的 `runs/r002/RUN.json` 为 COMPLETE，源提交 `abbe2b0f5a36fa66c628065ac7df98ef83b8a762`，GPU=1（GPU 2）。`r002_result.json` SHA-256 为 `fd741b0ec852f725eb42aab42554df402d2382d9a747b96d360e08aa56ee0824`；`raw_candidates.json.gz` SHA-256 为 `e7209ad33bf906434561100540f07d0f992cfad2994b3209166eaae94579ba87`，内容明确为 `not_generated`，而非以 NMS=1.0 冒充 NMS 前候选。
- 结论：**inconclusive**。r002 的五项证据链回归通过，但指定的两观察者之一未通过加载兼容性门；协议要求此时不得登记 `stop` 或 `continue`，也不得用 r001 的不同训练 split 权重替换归档观察者以产生统计数字。r001 的原始数字仍为历史、不能用于本轮科学裁决。
- 适用范围：本结论仅说明当前 d3/ai4rs 栈无法把主机登记的 ORCNN checkpoint 作为冻结观察者安全重建；不反映相邻实例假设真伪。
- 大型产物：`runs/r002/artifacts/{r002_result.json,raw_candidates.json.gz}`，均具备本轮可执行重建入口但在结论和远端保存前不得清理。
- 重建方式：在 d3 环境执行 `python src/run_r002_controlled.py --config configs/r002_controlled.json --output-dir runs/r002/artifacts`；该命令会先执行五个回归反例，再严格尝试指定 checkpoint，若仍无法兼容则可复现本轮 `inconclusive` 产物。
- B 复核补充（2026-09-06）：保留本轮 `inconclusive` 和全部历史产物，但撤回“当前环境无法安全重建归档 ORCNN 参数”的持续阻塞判断。本次在原环境用局部 NumPy metadata Unpickler 成功读取 348 个有限参数张量，归档配置模型 `strict=True` 加载全部匹配；尚未做前向与 clear AP50。另以实际生产汇总函数复现“应为 100pp 却得到 NaN”，并核实 ORCNN 没有现 hook 所需接口、RetinaNet identity hook 会遗漏坐标恢复。原五项自测 True 不证明整条证据链通过。审计源码、输入校验值与论文路线见 `lab/discussion.md` 2026-09-06 节；复现命令为 `python src/audit_controlled_evidence.py --config configs/r002_controlled.json --source src/run_r002_controlled.py --load-observer`。本补充没有产生新的科学效应统计。

## r003

- 假设：在冻结的 HRSC2016 相邻目标组、四级确定性整图退化中，两个冻结观察者会出现超过匹配孤立对照的实例分解特异错误。
- 代码与配置：`configs/r003_controlled.json`、`src/run_r003_evidence.py`。ORCNN 以局部 NumPy metadata unpickler 严格加载；RetinaNet 使用登记的 MMRotate 官方权重。ORCNN/Retina 的生产候选回放在非单位缩放图 `100000001.bmp` 上分别为 22/22、2/2，逐框最大绝对差均为 0。
- 清晰观察者核验：标准推理覆盖 HRSC test 453 图。修正报表函数后得到内部 greedy-AP50 计算 ORCNN=0.9585、Retina=0.1554（`runs/r003/artifacts/clear_ap50.json`）。这不是官方 DOTA evaluator 的已验证 AP50；尤其 Retina 值与登记官方 84.80 明显不符，故不能把它登记为有效的双观察者标准 AP50。
- 样本和平衡：100/100 逐组盲审接纳、0 排除；86 个相邻源图，排除后孤立池 356，匹配 240 个唯一控制目标/源图。四协变量 SMD 匹配前为 [0.2072,0.0551,0.1049,0.0086]，匹配后为 [0.0488,0.0498,0.0252,0.0460]，均 <=0.1。两个观察者的 clear score=.25/NMS=.1 共同队列却只有 27 组、23 图，低于预定的至少 50 组/20 图门；依赖连通分量 23 个，大小为 72 或 144 条条件记录。
- 条件性信号（非主结论）：在这 27 组的 score=.25/NMS=.1 下，ORCNN 四级效应为 0.0 [0,0]、3.704 [0,11.111]、0.0 [-15.385,13.793]、-7.407 [-26.087,10.0] pp；Retina 为 0.0 [0,0]、7.407 [0,19.231]、11.111 [-8.008,32.146]、14.815 [-7.692,36.0] pp。最严重级 Retina miss 增量 40.741 [22.572,60.0] pp、ORCNN miss 增量 14.815 [3.704,27.586] pp；两者 duplicate 均为 0。孤立 recall 最严重级 ORCNN 84.568 [72.024,95.679]%、Retina 65.741 [52.665,77.778]%。这些数字仅是未达主队列门的诊断，不能用于继续或停止裁决。
- 敏感性与缺口：最严重级主分数/NMS 的 27 组效应如上；ORCNN 在 score .05/.25/.50 与 NMS .1/.3/.5 范围为 -18.519 至 0 pp，Retina 为 0 至 18.519 pp，未显示双观察者稳定方向。运行产物未保存 NMS 前 matching coverage/最终 NMS 覆盖损失，也未把规定的生产回归反例写成可审计结果；这两项不能事后以近似数替代。
- 结论：**inconclusive**。控制平衡、候选回放和四级条件统计已生成，但有效双观察者标准 AP50 未建立、共同 clear 队列只有 27<50，且 NMS 前覆盖损失和生产反例输出缺失。因此证据既不满足 `continue`，也不是有效证据下的 `stop`；不得据此进入独立数据或真实配对桥接路线。
- 大型产物：`runs/r003/artifacts/{acceptance.json,clear_ap50.json,raw_candidates.json.gz,r003_result.json}`；`r003_result.json` 约 141 MiB、候选约 162 MiB。均只能依 RUN.json 的可重建条件由 `cqc-run run cleanup` 清理。
- 重建方式：`/home/rspip/miniconda3/envs/d3/bin/python src/run_r003_evidence.py --config configs/r003_controlled.json --out runs/r003/artifacts`；clear 补算为 `python src/evaluate_r003_clear_ap.py --config configs/r003_controlled.json --out runs/r003/artifacts/clear_ap50.json`。
- B 实测复核（2026-09-06）：本轮不能验收为科学输出完整。RetinaNet 使用 MMRotate 0.x 官方权重，却套用 1.x `FakeRotatedAnchorGenerator(angle_version='le90')`；官方实现明确要求加载旧权重时该锚框参数为 Python `None`，bbox coder 仍为 `le90`。只修正锚框参数，同一批 test 前 20 图的 score=.25/IoU≥.5 匹配数由 10/25 升至 23/25。随后完整 453 图、1228 个 GT 经 `DOTAMetric` 实际调用的 `eval_rbbox_map(use_07_metric=True, iou_thr=.5)` 复测：ORCNN AP50=**0.903558**，RetinaNet AP50=**0.847564**，对应参考分别为 0.9036、0.848；两者 state-dict 均严格加载。原 0.9585/0.1554 是自定义积分指标，不是可比的官方 AP50。
- 清晰队列修正：沿用原 100 组名单，标准 clear score=.25/NMS=.1 下 ORCNN 可完整分解 96 组，正确锚框的 RetinaNet 为 75 组，共同队列 **74 组/63 图**。这是匹配控制和共同支持筛选之前的清晰资格计数，不能保证最终 SMD 合格。撤回“当前 HRSC 清晰队列不足”的持续阻塞判断；原 27 组效应涉及错误 RetinaNet 观察者，不能作为修正后统计或失败证据。
- 复核代码与产物：源提交 `8bad173f69dbd58d39ede54fa7b4d560e8470ea9`，配置 `configs/retinanet_hrsc_legacy_inference.py`，审计 `src/audit_hrsc_observers.py`。产物位于 `runs/r003/artifacts/observer_review/{observer_review.json,orcnn_clear_predictions.json.gz,retina_clear_predictions.json.gz}`；review JSON SHA-256=`ebf7ad43828f4c705163ef2ca75d10e5b7c7cfc8485a27405d840d81fbddeb77`。未改写旧结果或重新打开已结束运行。本补充为清晰观察者复核，不含修正后的退化效应，也没有再次独立核验人工接纳记录。
- 复核重建：在原兼容推理依赖中运行 `python src/audit_hrsc_observers.py --config configs/r003_controlled.json --retina-config configs/retinanet_hrsc_legacy_inference.py --out runs/r003/artifacts/observer_review_rebuild`，输出目录必须尚不存在。仅做清晰推理，实际使用一张卡；其余卡已有负载，本次小规模复核无需新训练。补充产物尚未纳入原 RUN.json 的可清理输出，不可删除。
- 当前科学裁决仍为 **inconclusive**，但理由已收敛为：正确 RetinaNet 的退化候选、基于正确 clear 置信度的控制匹配和最终主队列统计尚待补算，同时原要求的 NMS 前覆盖损失、生产反例与主队列 SMD 尚待补齐；不再把已解决的观察者 AP 或 27 组计数当作当前阻塞。

## r004

- 假设：在正确冻结观察者下，四级整图模糊/采样退化会对相邻舰船组造成超过唯一匹配孤立对照的、跨观察者一致的实例分解特异风险。
- 观察者与实现：继承已实测官方 VOC07 AP50：归档 ORCNN **90.3558%**、正确旧版锚框语义 RetinaNet **84.7564%**（均为 trainval→test 冻结权重）。RetinaNet 只将 anchor `angle_version` 设为 Python `None`，bbox coder 保持 `le90`；两权重 strict load。候选为原生坐标恢复、score floor .05 后最终 NMS/max_per_img 前的候选；ORCNN 仍已受原生 RPN/RoI 预筛选，Retina 仍受原生 dense-head 预筛选。
- 接纳、队列与平衡：沿用 100 个既有组的 clear 审阅依据（逐项路径在 `acceptance.json`，未查看退化输出）；100 接纳、0 排除。正确 clear 下共同完整分解为 74 组/63 图；排除相邻源图后孤立池 356，清晰协变量驱动的不放回匹配取得 172 个唯一控制目标/源图。最终也是 74 组/63 图、63 个依赖分量（每分量 1--3 组）。四协变量 SMD 从 [0.1826,0.0216,0.2924,0.1295] 降至最终 [0.0496,0.0075,0.0367,0.0478]，均 <=.1；达到 >=50 组、>=20 图门。
- 主设定（score=.25/NMS=.1，4,000 次共享原图分量 bootstrap）：ORCNN 的四级主效应依次为 0.000 [0,0]、4.730 [0,10.256]、8.446 [2.055,15.064]、**3.266 [-4.445,11.302] pp**；RetinaNet 为 0.000 [0,0]、-1.689 [-5.556,2.597]、5.631 [-0.866,12.928]、**14.302 [3.444,25.321] pp**。最严重级孤立目标 recall 为 ORCNN 86.628% [81.034,91.813]、RetinaNet 76.163% [69.186,82.759]，均高于 50%。但 ORCNN 未达到 10pp、CI 跨零。
- 事件与归因：最严重级的 ORCNN/Retina miss 增量为 3.266 [-4.445,11.302]/14.302 [3.444,25.321] pp；merge 为 -1.351 [-4.286,0]/-1.351 [-4.286,0] pp，duplicate 均 0。因此正向变化主要是普通 miss，未出现满足条件的 merge 或 duplicate。主设定 NMS 前→NMS 后未截断→最终的匹配数（相邻/控制）分别为：clear ORCNN 172/171→172/170→172/170，最严重 ORCNN 156/153→156/149→156/149；clear Retina 172/163→172/158→172/158，最严重 Retina 142/135→141/131→141/131。故最终输出截断损失为 0；NMS 损失存在但不构成跨观察者的实例分解机制证据。
- 敏感性与验证：`r004_result.json` 包含四级×双观察者×3 score×3 NMS 的 **72** 个点估计和 95% CI（5328 条逐组条件记录）；所有敏感性使用同一 74 组固定队列。实际生产反例均为 True：交叉边最大基数两对、单侧四级记录 100pp、共享控制依赖/拒绝复用、同一候选的后处理一致性、SMD 失衡识别。完整合格集即这份已固定 74 组，不以退化结果二次筛选。
- 结论：**stop**。本轮为有效的正确观察者条件实验，但继续条件不成立：在任何同一非饱和等级，两个观察者没有同时达到 >=10pp 且下界>0；同一种 merge/duplicate 也没有 >=5pp 的一致增量。区间仍容许 ORCNN 存在有意义正效应，故这不是接受零假设；而是停止当前“实例分解特异性→任务忠实退化”投入路线，不进入独立数据、真实配对或生成器开发。
- 大型产物：`runs/r004/artifacts/{acceptance.json,raw_candidates.json.gz,r004_result.json}`，均由本次 `RUN.json` 声明为可重建；在远端结论保存前不得清理。
- 重建方式：`/home/rspip/miniconda3/envs/d3/bin/python src/run_r004_evidence.py --config configs/r004_controlled.json --out runs/r004/artifacts`；输入为 HRSC2016、两个配置登记的 `pth_data` 权重和 r003 observer review。
- 方案交接补充（2026-09-06）：只读核实 46 上 `runs/r003/RUN.json` 已为 `COMPLETE`，现有 `cqc-run run start` 只接受 `PLANNED`/`FAILED`，且要求 `run_id` 与唯一任务槽精确一致。因此修正前次“同编号追加补算”的执行建议：保留已结束记录与全部旧产物，将用户本次要求的后续补充实验登记为唯一 **r004**，科学问题、样本来源和原继续条件不变。这不是把 FAILED 的工程重试拆成新任务；r004 内工程失败仍须在同号修复继续。
- 本次交付的是 r004 科学方案与 `configs/r004_controlled.json`，尚未启动新退化推理，也未产生新效应。配置固定正确的 RetinaNet 适配、候选上限、继承资产、四级退化和统计口径；其中 `candidate_group_limit` 仅代表候选名单上限，不代表接纳成功。旧 `src/run_r003_evidence.py` 不能直接消费该协议完成任务，SERVER 须补齐相应实现后经现有运行器执行，新增产物由自己的 RUN.json 精确声明。旧缓存只继承 ORCNN 且补齐新控制输入，错误 RetinaNet 分支不得继承。原观察者审计补充产物依旧不因此获得删除授权。
