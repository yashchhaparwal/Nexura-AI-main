# Changelog

## Unreleased (V3)
- Settings `nexura.commitOnSuccess` (commit a verified run on a new branch, never pushed) and `nexura.budget.*` (request/token
  limits for free-tier quotas; the run pauses and can be resumed). The sidebar log shows the plan's cost estimate, a paused run,
  and the commit result.

## Unreleased (V2)
- Existing-project support: project profile and summary, per-folder approval of extra executables, optional local code index
  (consent first), baseline test run and regression check, multi-hunk patches, per-file approve/reject, index progress in the
  status bar, new commands (Enable Code Indexing, Rebuild Index, Show Project Summary, Edit Project Profile).

## 0.1.0
- First V1 build: plan, build, verify, approvals, diff review, undo, resume.
