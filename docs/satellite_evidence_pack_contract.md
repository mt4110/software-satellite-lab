# Satellite Evidence Pack Contract

## Name

Use:

```text
Satellite Evidence Pack
```

Avoid implying:

```text
Agent Skill
Plugin
Extension Marketplace
Capability Pack
```

## Definition

```text
A Satellite Evidence Pack is a declarative bundle that transforms local software-work artifacts into normalized evidence, recall instructions, evaluation suggestions, and read-only report widgets.
```

It may suggest. Core records. Human approves. Learning remains preview-only.

## v0 Allowed

- YAML or JSON manifest
- prompt / instruction templates
- recall policies
- evaluation criteria
- report / widget definitions
- schema-defined outputs
- core-owned transforms only

## v0 Denied

- arbitrary Python
- arbitrary JavaScript
- arbitrary shell
- network access
- secrets access
- repo write
- background daemon
- remote install
- auto-update
- marketplace distribution

## Trust Tiers

| Tier | Name | Permission |
|---|---|---|
| T0 | Read-only evidence pack | Can read selected local artifacts and write output artifacts. |
| T1 | Core transform pack | Can use core-owned transforms only. |
| T2 | Command pack | Deferred. Would require explicit allowlist, sandboxing, and tests. |
| T3 | Network pack | Deferred. Not allowed in the first product wave. |

## Pack Audit Must Check

- schema validity
- denied permission requests
- unknown fields
- output schema compatibility
- source path preservation
- learning-candidate inspection bypass attempts
- executable content presence
- remote URL usage

## First Wave Anti-Goals

- no IDE extension
- no marketplace
- no arbitrary plugin runtime
- no cloud server
- no multi-user auth
- no vector search
- no model fine-tune export
- no live provider integration
- no complex dashboard
