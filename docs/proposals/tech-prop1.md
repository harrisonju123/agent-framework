# Passive Intent Signals Reverse ETL - Tech Spec

## Overview

Sync passive upsell signals from Snowflake-exported Parquet files in S3 to MariaDB. Data Engineering exports company-level signals daily; mauvelous-hippo ingests them for direct DB queries by downstream services.

**Key Changes from Original Broker Proposal:**
- File format: Parquet (not gzipped CSV)
- S3 path: `passive_upsell_signals/dt={YYYY-MM-DD}/data.parquet`
- Data scope: Company-level signals (not member-level broker data)
- Purpose: Upsell intent indicators for downstream service queries

---

## Architecture

```
Snowflake → S3 (Parquet) → Temporal Workflow → MariaDB
                                ↓
                    Other services query DB directly
```

**Trigger:** Temporal Scheduled Workflow (daily cron, e.g., 6 AM UTC)

---

## Data Models

### Table: `company_upsell_signals`

| Column | Type | Description |
|--------|------|-------------|
| id | BIGINT UNSIGNED | Auto-increment PK |
| cdms_canonical_company_id | VARCHAR(64) | **UNIQUE** - Primary lookup key |
| legacy_company_id | VARCHAR(64) | e.g., "peo_146083" |
| core_product | VARCHAR(16) | "peo", "payroll", etc. |
| debut_product_tier | VARCHAR(32) | |
| current_product_tier | VARCHAR(32) | |
| is_active_customer | BOOLEAN | |
| initial_billed_ee_count | INT | |
| most_recent_paid_ee_count | INT | |
| ee_growth | INT | |
| ee_growth_rate | DECIMAL(10,4) | |
| count_icp_members_active | INT | |
| count_icp_members_all_time | INT | |
| count_eor_members_active | INT | |
| count_eor_members_all_time | INT | |
| count_remote_employees_active_since_premiere | INT | |
| last_remote_employee_invited_at | TIMESTAMP | Nullable |
| count_full_time_employees_active_since_premiere | INT | |
| last_full_time_employee_invited_at | TIMESTAMP | Nullable |
| last_hi_app_started_at | TIMESTAMP | Nullable |
| num_days_since_hi_app_started | INT | |
| **has_signal_active_icp_without_eor** | BOOLEAN | Signal flag |
| **has_signal_past_icp_without_eor** | BOOLEAN | Signal flag |
| **has_signal_two_plus_remote_employees** | BOOLEAN | Signal flag |
| **has_signal_payroll_growth_25_pct** | BOOLEAN | Signal flag |
| **has_signal_peo_growth_10_pct** | BOOLEAN | Signal flag |
| **has_signal_peo_basic_five_plus_hires** | BOOLEAN | Signal flag |
| **has_signal_hi_app_started_not_submitted** | BOOLEAN | Signal flag |
| model_updated_at | TIMESTAMP | When Snowflake model ran |
| created_at | TIMESTAMP | Record creation |
| updated_at | TIMESTAMP | Record update |

**Indexes:**
- `UNIQUE idx_cdms_company_id (cdms_canonical_company_id)` - Primary lookup
- `INDEX idx_legacy_company (legacy_company_id)` - Legacy ID queries
- `INDEX idx_core_product (core_product)` - Product segmentation
- `INDEX idx_model_updated (model_updated_at)` - Freshness queries

### Table: `upsell_signal_sync_metadata`

| Column | Type | Description |
|--------|------|-------------|
| id | BIGINT UNSIGNED | Auto-increment PK |
| sync_date | DATE | **UNIQUE** - Date partition |
| s3_key | VARCHAR(512) | Full S3 key |
| workflow_id | VARCHAR(256) | Temporal workflow ID |
| status | ENUM | 'started', 'completed', 'failed' |
| rows_processed | INT | Total rows in file |
| rows_inserted | INT | New records |
| rows_updated | INT | Existing records updated |
| error_message | TEXT | Nullable - error details |
| started_at | TIMESTAMP | |
| completed_at | TIMESTAMP | Nullable |

---

## Implementation Plan

### 1. Add Parquet Dependency

**File:** `go.mod`

Add: `github.com/parquet-go/parquet-go` (maintained fork with Go generics support)

### 2. Database Migrations

**File:** `migrations/000002_create_company_upsell_signals.up.sql`
- Create `company_upsell_signals` table with indexes

**File:** `migrations/000002_create_company_upsell_signals.down.sql`
- Drop table

**File:** `migrations/000003_create_upsell_signal_sync_metadata.up.sql`
- Create `upsell_signal_sync_metadata` table

**File:** `migrations/000003_create_upsell_signal_sync_metadata.down.sql`
- Drop table

### 3. Repository Layer

**File:** `internal/database/upsell_signals.go` (NEW)

```go
type UpsellSignalsRepository interface {
    UpsertSignalsBatch(ctx context.Context, signals []CompanyUpsellSignal) (inserted, updated int, err error)
    GetSyncMetadata(ctx context.Context, syncDate string) (*SyncMetadata, error)
    CreateSyncMetadata(ctx context.Context, meta *SyncMetadata) error
    UpdateSyncMetadata(ctx context.Context, id int64, status string, metrics SyncMetrics) error
}
```

### 4. Parquet Parser

**File:** `internal/parquet/upsell_signals.go` (NEW)

```go
// ParquetRow maps to parquet schema
type ParquetRow struct {
    CDMSCanonicalCompanyID string `parquet:"CDMS_CANONICAL_COMPANY_ID"`
    CompanyID              string `parquet:"COMPANY_ID"`
    // ... all columns
}

// ReadBatch reads rows [offset, offset+limit) from parquet file
func ReadBatch(reader io.ReaderAt, size int64, offset, limit int) ([]ParquetRow, error)

// GetRowCount returns total rows in parquet file
func GetRowCount(reader io.ReaderAt, size int64) (int64, error)
```

### 5. Temporal Activities

**File:** `internal/temporal/activities/upsell_signals.go` (NEW)

Activities to implement:
1. `CheckSyncMetadata` - Check if date already synced
2. `CreateSyncMetadata` - Record sync start
3. `GetParquetRowCount` - Get total rows for batching
4. `ReadParquetBatch` - Read batch of rows from S3
5. `UpsertSignalsBatch` - Upsert batch to DB
6. `UpdateSyncMetadataCompleted` - Mark sync complete
7. `UpdateSyncMetadataFailed` - Mark sync failed with error

**File:** `internal/temporal/activities/activities.go`
- Add `listUpsellSignalActivities()` method
- Register in `ListAllActivities()`

### 6. Temporal Workflow

**File:** `internal/temporal/workflows/sync_upsell_signals.go` (NEW)

```go
type SyncCompanyUpsellSignalsInput struct {
    SyncDate    string // "2026-01-30", defaults to yesterday
    ForceResync bool   // Re-process even if completed
}

type SyncCompanyUpsellSignalsResult struct {
    SyncDate      string
    RowsProcessed int
    RowsInserted  int
    RowsUpdated   int
    S3Key         string
}
```

**Workflow Logic:**
1. Check sync metadata (skip if already completed and !ForceResync)
2. Create sync metadata (status: started)
3. Get parquet row count
4. Loop: ReadParquetBatch → UpsertSignalsBatch (batch_size: 500)
5. Update sync metadata (status: completed)

**Workflow ID Pattern:** `sync-upsell-signals-{YYYY-MM-DD}`

### 7. Temporal Schedule

**File:** `internal/temporal/schedules.go` (or via Temporal UI)

- Schedule ID: `sync-upsell-signals-daily`
- Cron: `0 6 * * *` (6 AM UTC daily)
- Workflow: `SyncCompanyUpsellSignals`
- Input: `{ "SyncDate": "" }` (empty = yesterday)

### 8. Wire Dependencies

**File:** `internal/temporal/activities/activities.go`
- Add S3 service, DB repository to activities Client

**File:** `internal/config/config.go`
- Add S3 bucket config for upsell signals

---

## S3 Configuration

**Bucket:** `mauvelous-hippo-snowflake-nonprod-20260115212105210100000001`
**Path Pattern:** `passive_upsell_signals/dt={YYYY-MM-DD}/data.parquet`

**IAM Permissions Required:**
- `s3:GetObject` on `passive_upsell_signals/*`

---

## Idempotency

1. **Workflow ID**: `sync-upsell-signals-{date}` prevents duplicate executions
2. **Sync Metadata**: Check `status = 'completed'` before processing
3. **DB Upsert**: `INSERT ... ON DUPLICATE KEY UPDATE` on `cdms_canonical_company_id`

---

## Verification Plan

1. **Unit Tests:**
   - Parquet parser with sample file
   - Repository upsert logic
   - Activity params/results serialization

2. **Integration Tests:**
   - End-to-end workflow with mocked S3
   - Idempotency: run twice, verify no duplicates

3. **Manual Testing:**
   - Deploy to staging
   - Trigger workflow via Temporal UI with specific date
   - Query `company_upsell_signals` table
   - Verify `upsell_signal_sync_metadata` records

4. **Metrics to Verify:**
   - Rows processed matches source file
   - No errors in sync metadata
   - Query performance on indexed columns

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `go.mod` | Add parquet-go dependency |
| `migrations/000002_create_company_upsell_signals.up.sql` | CREATE |
| `migrations/000002_create_company_upsell_signals.down.sql` | CREATE |
| `migrations/000003_create_upsell_signal_sync_metadata.up.sql` | CREATE |
| `migrations/000003_create_upsell_signal_sync_metadata.down.sql` | CREATE |
| `internal/database/upsell_signals.go` | CREATE |
| `internal/parquet/upsell_signals.go` | CREATE |
| `internal/temporal/activities/upsell_signals.go` | CREATE |
| `internal/temporal/activities/activities.go` | MODIFY - register activities |
| `internal/temporal/workflows/sync_upsell_signals.go` | CREATE |
| `internal/config/config.go` | MODIFY - add S3 bucket config |

---

## Estimation (for Jira tickets)

| Task | Size |
|------|------|
| Database migrations | S |
| Repository layer + tests | M |
| Parquet parser + tests | M |
| Temporal activities + tests | M |
| Temporal workflow + tests | M |
| Wire dependencies | S |
| Temporal schedule setup | S |
| End-to-end testing | M |

**Total:** ~1.5 sprints (M-L)
