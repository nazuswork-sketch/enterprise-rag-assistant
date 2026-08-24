# DevOps Incident Response & Deployment Runbook

## 1. Production Incident Severities
- **SEV-1 (Critical Outage)**: Core user flow degraded or offline. Response SLA: 5 minutes. PagerDuty escalation to on-call lead.
- **SEV-2 (Major Issue)**: Non-critical feature broken or latency > 2000ms. Response SLA: 15 minutes.
- **SEV-3 (Minor)**: Internal tool failure or cosmetic bug. Response SLA: 2 hours.

## 2. Rollback Procedure for Kubernetes Clusters
If an automated Canary deployment fails health checks:
1. Identify the previous stable release tag via `git tag --sort=-creatordate`.
2. Run the rollback helm command: `helm rollback enterprise-core-service [REVISION_NUMBER] -n production`.
3. Flush Redis cluster cache: `redis-cli -h prod-redis.internal -a $REDIS_PASS FLUSHDB`.
4. Notify the `#incident-room` channel on Slack with incident summary and root cause.

## 3. Database Migration Standards
- All database migrations must be backwards-compatible (expand-and-contract pattern).
- Zero-downtime policy: Never run destructive `ALTER TABLE DROP COLUMN` without a prior 2-week deprecation window.
- Maximum lock timeout on PostgreSQL: `SET lock_timeout = '4s';`.
