# Git Workflow Guide

This document outlines a simple git workflow for safely experimenting with the DemandPack and site-to-region model without losing working versions.

## Initial Setup

To initialise git in this folder:

```bash
git init
git add .
git commit -m "Initial DemandPack baseline"
```

## Making Checkpoints

Before asking Cursor for big changes, create a checkpoint:

```bash
git add .
git commit -m "Checkpoint before changing DemandPack logic"
```

Use descriptive commit messages that explain what state you're preserving.

## Undoing Bad Experiments

To undo a bad experiment and go back to the last commit:

```bash
git restore .
```

This will discard all uncommitted changes and restore files to the last commit state.

## Working with Branches

Create a temporary branch for a new idea:

```bash
git checkout -b demandpack-weekly-regimes
# work + commits on the branch
git checkout main
git merge demandpack-weekly-regimes
```

This allows you to experiment safely on a branch, then merge back to main when ready.

## Tagging Milestones

To tag a PoC milestone:

```bash
git tag demandpack-poc-v1
```

Tags are useful for marking important versions (e.g., "baseline", "poc-v1", "pre-electrification") that you might want to reference later.

## Viewing History

To see your commit history:

```bash
git log --oneline
```

To see what files changed in a commit:

```bash
git show <commit-hash>
```

