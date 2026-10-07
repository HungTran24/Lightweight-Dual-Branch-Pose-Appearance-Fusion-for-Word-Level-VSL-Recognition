---
title: RTMW extraction manifest report
description: Deterministic RTMW extraction manifest inventory and sharding summary for smoke and full runs
status: ready
update_at: 29-08-26
---

# RTMW extraction manifest report

## Contract

- Hugging Face dataset: `hungchill24/vsl400-front-view`
- Hugging Face revision: `df5509963c62b2adaa18525d2990c8b30a100829`
- Known decode defect: `019442`

## Manifest summaries

| Mode | Total videos | Shards | Shard target | Inventory SHA256 | Known defect present |
|---|---:|---:|---:|---|---|
| `smoke` | 10 | 2 | 5 | `bfbbf82f3d0c382c...` | YES |
| `full` | 24,753 | 50 | 500 | `869cb8f9fd52f28e...` | YES |
