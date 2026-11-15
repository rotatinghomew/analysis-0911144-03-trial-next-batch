# 受可行域与在途试验约束的工业试制序贯选点

本项目保存业务资料的修订版本，提供资料校验、原始记录查询和现有口径的概览核算。操作人员通过本地 HTTP 接口导入资料，分析客户端按版本读取，避免后来的补录覆盖先前资料。

## 环境与运行

使用 Python 3.11，数值计算依赖固定在 requirements.txt。容器使用固定摘要的基础镜像。准备阶段可安装依赖，准备完成后测试与演示只访问本机；无需数据库服务或业务账号。

```sh
docker build -t trial_next_batch:local .
docker run --rm --network none trial_next_batch:local python -m unittest discover -s tests -v
docker run --rm --network none trial_next_batch:local python -m trial_next_batch demo
```

也可在 Python 3.11 虚拟环境中安装 requirements.txt 后运行相同的测试和模块命令。demo 在临时目录建库，启动真实本地 HTTP 服务，导入资料，再通过客户端读取结果，退出时关闭服务。运行过程没有固定等待或远程请求。

## 常驻资料接口

```sh
python -m trial_next_batch serve --database /tmp/trial_next_batch.sqlite3 --load fixtures/demo.json
```

默认监听 127.0.0.1:8080；--host 和 --port 可调整。服务用于可信本地环境，当前不提供公网认证与授权。数据保存在指定 SQLite 文件，重启后仍可按 revision 取回。API 格式见 docs/interfaces.md，业务数据语义见 docs/data-contract.md。

## 下一批试制建议

```sh
curl -s -X POST 'http://127.0.0.1:8080/plans' -H 'Content-Type: application/json' \
  -d '{"request": {"n_trials": 2, "cost_budget": 40, "strength_floor": 42, "loss_ceiling": 0.135}}'
```

建议只使用指定时点可见的完整记录；在途试验不进入拟合，失败试验只作为该区域不可行的证据，同炉次记录按共同误差处理。强度与损耗各自建模，不以任意权重合并成单一分数；有信息价值的边界组合会被标为探索，前沿上的组合会被标为确认。证据不足、成本无法覆盖或底线无解时返回 `no_recommendation` 或 `exploratory_only` 并给出原因。每次建议连同资料版本号与摘要保存，后续导入不会改写历史建议。完整字段与状态见 docs/planning.md。

## 目录与现有能力

`trial_next_batch/domain.py` 负责资料规则和已有概览；`repository.py` 负责不可变资料版本与导入回执；`service.py` 是服务边界；`client.py` 是独立 HTTP 消费者。fixtures/demo.json 是便于逐项核对的小样本，fixtures/history.json 是同口径的本地合成历史资料。测试覆盖领域规则、版本稳定性及真实 HTTP 往返。

现有概览的解释边界跟随输出的 meaning 或 visibility 字段。原始资料和描述性结果不自动构成推断、优化或对外承诺；调用者应保留版本号和摘要。

`planner.py` 的建议同样有解释边界：预测可行性不等于安全认证，`confirm` 与 `explore` 只是本次试验的用途标签，不是对配方优劣的最终结论。
