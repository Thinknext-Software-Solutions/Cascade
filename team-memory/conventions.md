# Coding Conventions

> Tell your team's AI sessions what your team's conventions are. Be specific. The more concrete examples, the better the AI's output.

## Naming

Add your team's naming patterns. Examples:
- Python: snake_case for functions and variables, PascalCase for classes
- Files: kebab-case for Python modules? snake_case? Be explicit.
- Database: singular or plural table names?
- API endpoints: REST conventions? GraphQL?

## File layout

Where do new files go? Examples:
- New API endpoint → `src/api/routes/`
- New database model → `src/models/`
- New test → `tests/{matching path as src/}`

## Style preferences

- Line length: 100? 120? 80?
- Quotes: single or double?
- Type hints: required everywhere? only public functions?
- Comments: minimal? extensive? what's the philosophy?

## Patterns we use

- Dependency injection style?
- Error handling pattern (exceptions vs Result types)?
- Logging conventions?
- Configuration management (env vars, config files, etc.)?

## Patterns we avoid

- Anti-patterns specific to your team
- Libraries you've intentionally chosen NOT to use
- Approaches that have burned you before

---

*This is your team's file — fill it in with real specifics. The AI's output will reflect exactly what's here.*
