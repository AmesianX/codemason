---
name: test-driven-development
description: TDD — enforce RED → GREEN → REFACTOR, write the test before the code.
---

# Test-Driven Development

Write the test first. Watch it fail. Make it pass with the least code. Then
refactor. Skipping the RED step means you never proved the test can fail — so it
proves nothing.

## The Iron Law

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST.
```

## When to use

New features, bug fixes (the regression test IS the RED step), refactors that
need a safety net, and any change whose behavior you can state as an assertion.
Skip only for pure exploration/spikes (throwaway code) and trivial,
untestable glue.

## The Red-Green-Refactor cycle

### RED — write a failing test

Write the smallest test that captures ONE piece of desired behavior. Run it with
`Bash` and **watch it fail for the right reason** (assertion failure, not an
import error or typo). A test that errors instead of failing is not yet a valid
RED.

```bash
# run just the new test
pytest tests/test_thing.py::test_new_behavior -q     # python
npm test -- -t "new behavior"                        # node
```

### GREEN — make it pass

Write the **least** code that makes the test pass — even something blunt. Do not
add behavior the test doesn't demand. Run the one test (GREEN), then run the
FULL suite to catch regressions.

```bash
pytest tests/test_thing.py::test_new_behavior -q && pytest -q
```

### REFACTOR — clean up under green

With the suite green, improve names, remove duplication, and simplify — re-running
the suite after each change. Never refactor with a red bar.

Then repeat for the next slice of behavior.

## Vertical slices, not horizontal

Build one thin end-to-end behavior at a time (input → logic → output for a
single case), not "all the models, then all the services, then all the tests".
Each cycle should leave the system working for one more real case.

## Common rationalizations (all wrong)

- "I'll add tests after" — you won't, and you'll have proved nothing.
- "This is too simple to test" — then the test is trivial to write.
- "I already know it works" — then RED passing first try tells you the test is
  broken; that's information.

## Verification checklist

Every new behavior has a test written *before* it; you saw each test fail before
making it pass; the full suite is green; no production code exists that no test
exercises.
