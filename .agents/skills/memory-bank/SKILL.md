---
name: memory-bank
description: Initializes, reads, and synchronizes the persistent markdown memory bank logs for this project workspace. Use when starting a project or logging macro progress.
---

# Skill: Memory Bank Management

This skill allows the agent to handle lifecycle actions for the workspace's persistent memory tracking files.

## When to Use This Skill
- When the user explicitly states "initialize memory bank" or "update memory bank".
- When stabilizing the environment state after a major feature migration or refactor sequence.

## How to Use It

### On Command: "initialize memory bank"
1. Perform a Pre-Flight validation scan using filesystem or custom memory tools.
2. If the workspace registry directory does not exist, initialize it.
3. Generate skeletal markdown files for the core structure (`projectbrief.md`, `productContext.md`, `systemPatterns.md`, `techContext.md`, `activeContext.md`, `progress.md`).
4. Perform an initial codebase structural sweep to populate the `techContext.md` with accurate language and framework parameters.

### On Command: "update memory bank"
1. Perform a full re-read of all active memory files to align the current state.
2. Scan recent workspace logs, modifications, or implementation artifacts.
3. Rewrite the relevant updates into the local files following the reverse-order rule (`progress.md` -> `activeContext.md` -> others).
4. Validate changes with the user if structural pattern discrepancies are discovered.