# Prior Work

> Brief summaries of recently shipped stories. Cascade reads this to avoid duplicate work and to understand the existing surface area of the codebase.

## Format

```
## [YYYY-MM-DD] Story title
**Summary**: What was built (1-2 sentences).
**Files touched**: Primary modules / endpoints / components.
**Notes**: Anything future work should know.
```

## Example

## [2026-04-22] Pagination on /api/users
**Summary**: Added cursor-based pagination to the users list endpoint with `?limit=` and `?after=` query params.
**Files touched**: `src/api/routes/users.py`, `src/models/user.py`.
**Notes**: Default limit is 50, max is 200. If you need to extend pagination to other endpoints, follow the same cursor pattern (don't introduce offset-based).

---

*Add summaries of shipped stories below — keep them brief, link out to PRs for details.*
