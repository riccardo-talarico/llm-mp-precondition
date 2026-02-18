IDENTIFY_CONCURRENCY_PROMPT = """
You are a Go static analysis assistant. Your goal is to extract the concurrency "topology" of the provided code. 

Focus exclusively on these primitives: 
- Channels (buffered/unbuffered, size, direction)
- sync.Mutex and sync.RWMutex
- sync.WaitGroup
- sync.Cond
- select statements and goroutine spawns (go func())

For every primitive found, you must provide:
1. **Type**: The exact Go type or primitive name.
2. **Function**: The name of the function where the primitive is declared or used.
3. **Scope/Context**: A brief description of where it resides (e.g., "inside a for-loop," "guarded by an if-statement," "inside a select case").

Output the results as a clear, structured list. Do not analyze bugs yet; just map the primitives.
"""

GENERATE_TRACE_PROMPT = """
You are a concurrency adversary. Using the provided map of Go primitives, your task is to hypothesize a "Problematic Trace" that results in a blocking bug (deadlock, leak, or hang).

Ignore the high-level business logic; focus only on how these primitives can interact in the worst possible order.

**Step 1: Interleaving Description**
Describe the logical sequence of events. Explain how the goroutines interact, which one holds a resource, and why another becomes blocked. Focus on the "Conflict Set" (e.g., Goroutine A waits for a channel that Goroutine B will never send to).

**Step 2: Textual Timeline**
Generate a strict chronological timeline of actions. Use the notation: `G[N]: [Action]`.
Example:
- G1: mu.Lock()
- G2: mu.Lock() (blocked)
- G1: ch <- val (blocked)
- Result: Deadlock.

Your goal is to find a specific order of actions that forces a VIOLATION of "Partial-Deadlock Freedom."

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

If the trace is reachable, confirm the bug. If it is impossible, explain exactly which structural constraint (e.g., "the 'if' on line 22 prevents G2 from ever reaching the Lock call") makes it so.
Trace:
{trace}
"""

CLASSIFICATION_PROMPT = """You are a Go Concurrency Expert and Bug classificator. You are given Go snippets of codes and an identified problematic trace that causes a bug
You goal is to classify the bug illustrated by the trace through the GoBench paper classification.
'Bug Classification Hierarchy (GoBench paper classification): classify it into exactly one subtype and subsubtype based on these definitions
'1. Resource Deadlock: Goroutines block waiting for a synchronization resource (lock) held by another. Subsubtypes
'1.1. Double Locking: A single goroutine attempts to acquire a lock it already holds, causing it to block itself
'1.2. AB-BA Deadlock: Multiple goroutines acquire multiple locks in conflicting orders (e.g., G1: Lock A then B; G2: Lock B then A
'1.3. RWR Deadlock: Involving sync.RWMutex. A pending Write lock request takes priority, blocking subsequent Read requests even if the current lock is a Read lock, potentially creating a cycle if the current Reader waits for a new Reader.
'2. Communication Deadlock: Goroutines block waiting for a message/signal from another. Subsubtypes:
'2.1. Channel: Sending/Receiving on a channel where no counterpart is available to complete the handoff (e.g., unbuffered channel leaks.
'2.2. Condition Variable: Misuse of sync.Cond (e.g., Wait() is called but Signal() or Broadcast() is never triggered due to logic errors).
'2.3. WaitGroup: Calling Wait() on a sync.WaitGroup where the internal counter never reaches zero due to missing Done() calls.
'2.4. Channel & Context: Complex communication blocks involving the interaction of channels with context.
'2.5. Channel & Condition Variable: Complex communication blocks involving the interaction of channels with condition variables.
'3. Mixed Deadlock: A cycle created by mixing message-passing and shared-memory synchronization. Subsubtypes:
'3.1. Channel & Lock: A cycle where a goroutine holds a lock while waiting for a channel operation, while the counterpart for that channel operation is waiting for the same lock.
'3.2. Channel & WaitGroup: A cycle where a channel operation is blocked by a WaitGroup.Wait(), or a WaitGroup.Done() is blocked by a channel operation.
Trace: {trace}"""