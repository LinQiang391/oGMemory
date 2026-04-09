# Debugging Workflow Pattern

## Problem Analysis Phase
1. **Reproduce the issue** - Get reliable reproduction steps
2. **Isolate variables** - Identify what changed between working and broken states
3. **Check data flow** - Trace how data moves through the system

## Diagnostic Logging
4. **Add logs at boundaries** - Entry/exit of functions, before/after state changes
5. **Log critical values** - Input parameters, return values, state mutations
6. **Use structured logging** - JSON format for easier parsing

## Hypothesis Testing
7. **Form hypothesis** - Based on symptoms, guess the root cause
8. **Add verification** - Log or assert that validates/refutes hypothesis
9. **Minimize changes** - One change at a time, verify effect

## Common Pitfalls
- Don't change too many things at once
- Don't skip to the solution without understanding the problem
- Don't ignore error messages or stack traces
- Remember: correlation is not causation
