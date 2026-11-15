# 本地接口约定

所有接口使用 UTF-8 JSON，错误响应为带 error 字段的对象。GET /health 返回进程健康状态。POST /imports 接收 request_key 与 data，data 遵循业务资料契约；新导入返回 201，完全相同的请求重放返回 200。相同请求号对应不同资料返回 409；非法资料返回 400，不留下部分版本。

GET /revisions 列出 revision 和内容摘要。GET /records 返回业务原始记录；GET /summary 返回已有概览，两者都返回 revision、digest 和 result。可加 revision 查询参数固定历史资料，不提供时读取最新资料；资料不存在为 404。摘要对对象键顺序不敏感，对数组顺序敏感；资料内容相同可共享版本，导入请求回执相互独立。

未知路径为 404。现有客户端依赖这些路径和包装字段；后续增加业务分析接口时保留既有行为，不能把资料导入成功等同于业务分析已经形成结论。每个分析产物应声明所用数据版本，不能在客户端读取时偷偷替换成最新资料。

## 试制建议接口

`POST /plans?revision=N` 接收 `{"request": {...}}`，在固定资料版本上计算下一批试制建议并保存快照，返回 201：

```json
{"revision": 1, "digest": "……", "plan": {"plan_id": 1, "created_at": "……", "revision": 1,
  "data_digest": "……", "request_digest": "……", "engine_version": "1.0.0"},
 "result": {"status": "recommended", "data_basis": {}, "suggestions": [], "reasons": []}}
```

- 不传 `revision` 时使用最新资料，`at` 缺省取该资料的最大 `visible_at`；传入 `revision` 时严格使用该版本，不会在计算中途换成最新资料。
- 请求参数非法或含未知字段返回 400；资料版本不存在返回 404。
- `GET /plans` 列出已保存的建议（可加 `revision` 过滤）；`GET /plans/{plan_id}` 返回当时保存的完整结果。历史建议不会因后续导入而改变。
- 业务语义、状态取值与模型边界见 docs/planning.md。
