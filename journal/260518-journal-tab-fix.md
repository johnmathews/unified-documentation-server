# 260518 — Fix empty Journal tab in webapp

## The bug

The webapp's Journal page rendered zero entries on a fresh deployment, even
though `journal/` directories full of `*.md` files were being ingested
normally. The filter is client-side: the page asks `/api/sources/tree` and
keeps files whose `type === "journal"`. Every file was coming back tagged
`documentation`.

Root cause: `DocTypesConfig`'s defaults shipped with `global_rules = ()`. The
classifier walks per-source rules → global rules → `fallback_type`
(`documentation`). With no global rules and no `document-types.yml` mounted
(the example file is `document-types.example.yml`, gitignored copy is
optional), every parent doc fell through to `documentation`. The Journal
page therefore filtered to an empty set.

Production *had* a `document-types.yml` with `journal/**` rules and worked
fine. Fresh installs and local dev did not, because nobody copied the
example. The example existing as a separate file silently turned what
should have been a sensible default into an opt-in.

## The fix

Baked the example's `global_rules` into `_DEFAULT_GLOBAL_RULES` in
`config.py` and made it the default for `DocTypesConfig.global_rules`. The
rule set covers:

- `**/journal/**` and `journal/**` → `journal`
- `**/prompts/**` and `prompts/**` → `prompt`
- `**/*.lock` → `not-docs`
- `**/.DS_Store` → `not-docs`

Both top-level and nested patterns are needed because `fnmatch`'s `**`
doesn't match an empty path component. An explicit YAML's `global_rules`
still **replaces** the defaults (no merge) — preserves the original
semantics for operators who want a custom rule set.

## Cache invalidation on upgrade

`KnowledgeBase.backfill_types_if_needed` short-circuits when the stored
hash equals the current config hash. Naively, deployments that had stored
the SHA256 of their `document-types.yml` (or the empty string, for "no
YAML") would never re-trigger when the defaults changed — the file content
hadn't moved.

Versioned the cache key: stored value is now `"{version}:{sha256}"` where
`version = _DOC_TYPES_DEFAULTS_VERSION` (currently `"v2"`). Pre-versioned
rows (bare SHA256 or empty string) compare unequal to any combined form,
so every existing deployment reclassifies exactly once after the upgrade
and stores the new combined key. Future changes to the baked-in defaults
just need a version bump.

## Tests

- `test_default_config_classifies_journal_and_prompt_paths` pins the new
  out-of-the-box behaviour.
- `test_yaml_global_rules_replace_defaults` pins replace-not-merge.
- `test_backfill_reclassifies_when_legacy_bare_sha_stored` and
  `test_backfill_reclassifies_when_no_config_after_legacy_empty_hash`
  cover the upgrade path from both pre-versioned forms.
- `test_single_source_tree_includes_type_field` and
  `test_bulk_tree_includes_type_field` are regression tests on the REST
  endpoints — the webapp filters by `type`, so silently dropping it from
  the response would mask future regressions.

## Other

Updated `docs/architecture.md` to describe the baked-in defaults and the
versioned cache key. Updated the comment block at the top of
`config/document-types.example.yml` to point readers at `_DEFAULT_GLOBAL_RULES`
and clarify the replace-not-merge semantics.
