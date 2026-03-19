# Coding Style Preferences

## Language Preferences
- **TypeScript**: Strict mode enabled, no implicit any
- **Python**: Type hints required for all functions

## Code Organization
- Prefer functional composition over class inheritance
- Keep functions small and focused (single responsibility)
- Use immutable data structures where possible

## Naming Conventions
- Variables: camelCase
- Constants: UPPER_SNAKE_CASE
- Types: PascalCase
- Private members: _prefix

## Anti-Patterns to Avoid
- Mutable state in TypeScript classes
- God functions that do too much
- Nested callbacks (use async/await)
