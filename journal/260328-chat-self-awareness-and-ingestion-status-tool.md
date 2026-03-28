# Chat Self-Awareness and Ingestion Status Tool

## Problem

The chat agent couldn't answer meta questions about itself — "are all sources indexed?",
"what's the most recent journal entry?" — even though the data was available. It would
respond with "I cannot confirm" or "I would need to use tools" instead of using the
inventory data already in its context.

## Changes

### Enriched Chat System Prompt

The system prompt now includes per-source indexing stats (file count, chunk count,
last-indexed timestamp) alongside the document inventory. It also includes root_docs
and engineering_team categories that were previously missing from the inventory.

The prompt instructions were rewritten to tell the agent to answer confidently from
the inventory data rather than hedging or deferring to tools it can't use.

All stats are queried live from the database on every chat request — nothing is static.

### New `ingestion_status` MCP Tool

Added a new MCP tool that compares configured sources against indexed sources:
- Lists all configured source names from the YAML config
- Lists all indexed sources with file counts, chunk counts, last indexed time
- Flags any sources that are configured but not yet indexed
- Returns a `fully_indexed` boolean

This is useful for MCP clients (not the chat endpoint) that want to check indexing health.

### Duplicate Source Name Validation

Added earlier in this session — `_parse_sources()` now raises `ValueError` if two sources
share the same name, preventing silent data loss from config copy-paste errors.
