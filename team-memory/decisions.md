# Architectural Decisions

> ADR-style log: what we chose, why, and when. Cascade reads this to avoid suggesting paths the team has already considered and rejected.

## Decision template

```
## [YYYY-MM-DD] Decision title
**Context**: What was the situation that required a decision?
**Decision**: What did we decide?
**Why**: What were the reasons? What alternatives did we reject and why?
**Implications**: What does this mean for future work?
```

## Example

## [2026-01-15] PostgreSQL over MongoDB for primary datastore
**Context**: Needed a primary database for the user-facing application.
**Decision**: PostgreSQL.
**Why**: Strong relational guarantees, mature ecosystem, team familiarity. Rejected MongoDB due to schema flexibility being a non-feature for this use case (we DO want schema enforcement), and operational complexity.
**Implications**: All new persistence work uses Postgres. No new MongoDB introductions. Migrations via Alembic.

---

*Add your team's decisions below. The AI will respect them when proposing solutions.*
