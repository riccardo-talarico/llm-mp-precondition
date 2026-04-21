IDENTIFY_SECTIONS_PROMPT = """
You are a Concurrency Architect and Static Analysis expert. 
Your goal is to map the Synchronization Surfaces of a Go program. 
A synchronization surface is any point where the execution of one goroutine depends on, or influences, the state of another.

Code:
{code}

Here there is a list of the concurrency primitives used in the code.
Primitives:
{primitives}
You must use the provided tool/schema to format your response. Output ONLY valid JSON matching the schema, and no other text.
"""


UNDERSTAND_SECTION_PROMPT = """
You are a Senior Concurrency Architect. 
Your task is to reverse-engineer a specific "Concurrency Section" to understand its architectural intent and operational guarantees.

Full snippet of code:
{code}

Target Section:
{section}

1. Functional Intent (The "Why"):
Define the high-level goal of this section. Is it a Worker Pool, a Fan-In aggregator, a Circuit Breaker, a Resource Gatekeeper, or a Lazy Initializer? 
Explain what problem this specific arrangement of primitives is trying to solve.

2. Functioning (The "How"):
Trace the lifecycle of the state within this section:
Activation: What triggers the concurrency? (e.g., a method call, a channel signal, or an init).
Orchestration: How do the goroutines coordinate? Describe the "Handshake" between the primitives.
Termination: How does this section reach a quiet state? Identify the cleanup mechanism (e.g., context cancellation, channel closing, or WaitGroup.Wait).

Possibly including Structural Invariants & Properties:
The "laws" that must remain true for this code to be safe. Examples include:

Ownership: Which goroutine "owns" the channel or the data?
Cardinality Invariants: (e.g., "Exactly one writer, multiple readers" or "The number of active workers never exceeds the channel buffer").
Blocking Guarantees: Is the section designed to be non-blocking for the caller, or is it a synchronous "Wait-until-done" pattern?
Re-entrancy: Can this section be safely executed multiple times concurrently?

Constraint: Focus on the structural logic. If the business logic is "calculating taxes," ignore the math and focus on how the "tax calculation workers" are managed.
You must use the provided tool/schema to format your response. Output ONLY valid JSON matching the schema, and no other text.
"""

GENERATE_TRACE_FROM_IDEA_PROMPT = """
You are Concurrency Stress-Tester and Go Runtime Emulator. Your goal is, given an analysis of a go snippet of code and 
an idea of a possible bug, to create a problematic trace for that bug.
Describe the interleaving logic and generate a strict chronological timeline of actions (list of actions).
Use the string: `G[N]: [Action]` for an action.
Example:
- G1: mu.Lock()
- G2: mu.Lock() (blocked)
- G1: ch <- val (blocked)

Code: 
{code}

Idea:
{trace_idea}
You must use the provided tool/schema to format your response. Output ONLY valid JSON matching the schema, and no other text.
"""



CHECK_BALANCE_PROMPT = """
You are a Go programming expert and concurrency auditor.
TASK:
Analyze the usage of Go concurrency primitives in the provided code and detect potential imbalances.
Check dual operations: for each channel where are the send and where are the receives? Are they balanced? 
Coudl they be unbalanced in some execution paths? Is the imbalance a problem?
The same for mutexes (lock and unlock)
Reason also on waitgroups, where are the add and waits? Are they balanced?

Code:
{code}

Primitives:
{primitives}

CONSTRAINTS:
- Identify mismatches between:
  - channel sends and receives
  - mutex locks and unlocks
  - waitgroup adds and waits
- Consider possible execution paths, (including possible ways in which select statements could ruin the balance)
- Report only concrete issues or potential imbalances
- Each problem must be a short sentence (max 15 words)
- If no issues are found:
  - has_issues = false
  - problems = empty list
- Do not include explanations outside the JSON
- Do not infer primitives not listed in the input

STEPS:
1. Identify all occurrences of each primitive in the code
2. Count relevant operations (send/receive, lock/unlock, add/wait)
3. Evaluate balance across possible execution paths
4. Report results per primitive
You must use the provided tool/schema to format your response. Output ONLY valid JSON matching the schema, and no other text.
"""

GENERATE_TRACE_IDEAS_PROMPT = """
You are a Concurrency Stress-Tester and Go Runtime Emulator. Your goal is to identify potential concurrency bugs in a go snippet of code.
Provide a list of ideas of potential issues.
Code:
{code}

Use this analysis as a support:
{sections}
{understanding}
{balanceanalysis}

CONSTRAINTS:
- Generate at most 5 ideas
- Each idea must describe a plausible concurrency issue, even if uncertain
- Base ideas only on the provided inputs
- Prioritize sections and the "balance" analysis when identifying issues
- description must be max 20 words
- location must reference a relevant section
- Do not repeat the same issue type for the same location
- Do not include explanations outside the structured output

STEPS:
1. Review sections to locate concurrency interactions
2. Use understanding to interpret behavior
3. Use balanceanalysis to detect imbalance-related risks
4. Generate diverse potential issues
You must use the provided tool/schema to format your response. Output ONLY valid JSON matching the schema, and no other text.
"""


GENERATE_TRACES_PROMPT = """
You are a Concurrency Stress-Tester and Go Runtime Emulator. Your goal is to identify execution interleavings in Go code that result in deadlocks, data races, or logical hangs.

### TASK
Examine the provided Go code and its primitives. Hypothesize specific, execution sequences (traces) where concurrent operations overlap in a way that breaks the program.
Ignore the high-level business logic; focus only on how the concurrency primitives can interact in the worst possible order.

For each trace:
**Step 1: interleaving_logic**
Describe the logical sequence of events. Explain how the goroutines interact, which one holds a resource, and why another becomes blocked. Focus on the "Conflict Set" (e.g., Goroutine A waits for a channel that Goroutine B will never send to).

**Step 2: sequence**
Generate a strict chronological timeline of actions (list of actions). Use the string: `G[N]: [Action]` for an action.
Example:
- G1: mu.Lock()
- G2: mu.Lock() (blocked)
- G1: ch <- val (blocked)

Your goal is to find a specific order of actions that causes a concurrency bug.
Generate up to five problematic traces.

Code:
{code}

Primitives:
{primitives}
"""

VERIFY_TRACE_PROMPT = """
You are a rigorous Go program verifier. You are given a Go program and a "Hypothesized Problematic Trace."

Your task is to determine if this trace is actually **reachable** given the program's structural constraints. 

Analyze the following:
1. **Control Flow**: Do the 'if' conditions, 'for' loop boundaries, or 'return' statements allow the goroutines to reach the states described in the trace?
2. **Initialization**: Are the channels and mutexes properly initialized before the actions in the trace occur?
3. **Capacity**: Does a channel's buffer size (if any) prevent the specific blocking sequence proposed?
4. **Lifetime**: Does one goroutine's parent function exit and terminate the child before the trace can complete?

If the trace is reachable, confirm it and explain how. If it is impossible, explain exactly which structural constraint (e.g., "the 'if' on line 22 prevents G2 from ever reaching the Lock call") makes it so.
Code:
{code}
Trace:
{trace}
Do not escape single quotes (e.g., use ', not \')
You must use the provided tool/schema to format your response. Output ONLY valid JSON matching the schema, and no other text.
"""

CLASSIFICATION_PROMPT = """You are a Go Concurrency Expert and Bug classificator. You are given Go snippets of codes and an identified problematic trace that causes a bug
You goal is to classify the bug illustrated by the trace through the following classification.
Bug Classification Hierarchy: classify it into exactly one class, type and subtype based on these definitions:
Class 1: Blocking Bugs, types:
1.1. Resource Deadlock: Goroutines block waiting for a synchronization resource (lock) held by another. Subtypes:
1.1.1 Double locking: A single goroutine attempts to acquire a lock it already holds, causing it to block itself
1.1.2. AB-BA deadlock: Multiple goroutines acquire multiple locks in conflicting orders (e.g., G1: Lock A then B; G2: Lock B then A
1.1.3. RWR deadlock: Involving sync.RWMutex. A specific 'Write-after-Read' priority block. Occurs when a recursive Read lock is attempted while a Write lock is already pending, 
causing the second Read to block behind the Write, which is blocked by the first Read. 
1.2. Communication Deadlock: Goroutines block waiting for a message/signal from another. Subtypes:
1.2.1. Channel: Sending/Receiving on a channel where no counterpart is available to complete the handoff (e.g., unbuffered channel leaks.
1.2.2. Condition Variable: Misuse of sync.Cond (e.g., Wait() is called but Signal() or Broadcast() is never triggered due to logic errors).
1.2.3. WaitGroup: Calling Wait() on a sync.WaitGroup where the internal counter never reaches zero due to missing Done() calls.
1.2.4. Channel & Context: Complex communication blocks involving the interaction of channels with context.
1.2.5. Channel & Condition Variable: Complex communication blocks involving the interaction of channels with condition variables.
1.3. Mixed Deadlock: A deadlock cycle specifically requiring at least one shared-memory primitive (Lock/WaitGroup) AND at least one communication primitive (Channel). 
Example: G1 holds Mutex A and waits for Channel B; G2 sends to Channel B but is blocked waiting for Mutex A. Subtypes:
1.3.1. Channel & Lock: A cycle where a goroutine holds a lock while waiting for a channel operation, while the counterpart for that channel operation is waiting for the same lock.
1.3.2. Channel & waitGroup: A cycle where a channel operation is blocked by a WaitGroup.Wait(), or a WaitGroup.Done() is blocked by a channel operation.
Class 2: Nonblocking bugs, types:
2.1 Traditional: traditional concurrency issues that are also found in other languages. Subtypes:
2.1.1 Data race: Concurrent access to a memory location where at least one access is a write.
2.1.2 Order violation: A bug where Goroutine G1 relies on a state or value from Goroutine G2, 
but executes before Goroutine G2 has completed the necessary work (e.g., accessing a resource before it is initialized or after it is closed).
2.2 Go-Specific: Non-blocking bugs that arise from unique Go language features or its standard library. Subtypes:
2.2.1 Anonymous Function: Data races caused by variables implicitly shared between a parent and child goroutine within an anonymous function.
2.2.2 Misuse channel: Issues like setting a channel to nil while other goroutines are communicating on it, which can trigger data races.
2.2.3 Testing Library: Bug caused by the misuse of the Testing library of Go.

Logic Steps:
1. Analyze the trace
2. Determine if the bug cause a total/partial halt (blocking) or incorrect execution/data (nonblocking)
3. Identify the root cause primitive (e.g. is it a channel?)
4. Classify it according to the hierarchy

Code: 
{code}
Trace: {trace}, trace eval: {trace_eval}
You must use the provided tool/schema to format your response. Output ONLY valid JSON matching the schema, and no other text.
"""


SINGLE_PROMPT_DETECTION_AND_CLASSIFICATION = """Role: You are a Go Concurrency Expert and Program Verifier. Your task is to analyze Go code snippets to identify and classify bugs (if there are any) according to the following hierarchy:
Bug Classification Hierarchy: If a bug is detected, classify it into exactly one class, type and subtype based on these definitions:
Class 1: Blocking Bugs, types:
1.1. Resource Deadlock: Goroutines block waiting for a synchronization resource (lock) held by another. Subtypes:
1.1.1 Double locking: A single goroutine attempts to acquire a lock it already holds, causing it to block itself
1.1.2. AB-BA deadlock: Multiple goroutines acquire multiple locks in conflicting orders (e.g., G1: Lock A then B; G2: Lock B then A
1.1.3. RWR deadlock: Involving sync.RWMutex. A specific 'Write-after-Read' priority block. Occurs when a recursive Read lock is attempted while a Write lock is already pending, 
causing the second Read to block behind the Write, which is blocked by the first Read. 
1.2. Communication Deadlock: Goroutines block waiting for a message/signal from another. Subtypes:
1.2.1. Channel: Sending/Receiving on a channel where no counterpart is available to complete the handoff (e.g., unbuffered channel leaks.
1.2.2. Condition Variable: Misuse of sync.Cond (e.g., Wait() is called but Signal() or Broadcast() is never triggered due to logic errors).
1.2.3. WaitGroup: Calling Wait() on a sync.WaitGroup where the internal counter never reaches zero due to missing Done() calls.
1.2.4. Channel & Context: Complex communication blocks involving the interaction of channels with context.
1.2.5. Channel & Condition Variable: Complex communication blocks involving the interaction of channels with condition variables.
1.3. Mixed Deadlock: A deadlock cycle specifically requiring at least one shared-memory primitive (Lock/WaitGroup) AND at least one communication primitive (Channel). 
Example: G1 holds Mutex A and waits for Channel B; G2 sends to Channel B but is blocked waiting for Mutex A. Subtypes:
1.3.1. Channel & Lock: A cycle where a goroutine holds a lock while waiting for a channel operation, while the counterpart for that channel operation is waiting for the same lock.
1.3.2. Channel & waitGroup: A cycle where a channel operation is blocked by a WaitGroup.Wait(), or a WaitGroup.Done() is blocked by a channel operation.
Class 2: Nonblocking bugs, types:
2.1 Traditional: traditional concurrency issues that are also found in other languages. Subtypes:
2.1.1 Data race: Concurrent access to a memory location where at least one access is a write.
2.1.2 Order violation: A bug where Goroutine G1 relies on a state or value from Goroutine G2, 
but executes before Goroutine G2 has completed the necessary work (e.g., accessing a resource before it is initialized or after it is closed).
2.2 Go-Specific: Non-blocking bugs that arise from unique Go language features or its standard library. Subtypes:
2.2.1 Anonymous Function: Data races caused by variables implicitly shared between a parent and child goroutine within an anonymous function.
2.2.2 Misuse channel: Issues like setting a channel to nil while other goroutines are communicating on it, which can trigger data races.
2.2.3 Testing Library: Bug caused by the misuse of the Testing library of Go.
Assume Partial Context: If the snippet is missing a main function, assume the provided functions are called in a way that triggers the concurrency logic shown.
If no bug is found then insert None in all the fields.
You must use the provided tool/schema to format your response. Output ONLY valid JSON matching the schema, and no other text.
""" 