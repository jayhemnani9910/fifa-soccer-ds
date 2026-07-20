# GitHub repository settings checklist

Repository settings are external state and cannot be verified from this checkout. A repository
administrator should review the following controls in GitHub after workflow changes.

## Security and analysis

- Enable secret scanning and push protection where the repository plan supports them.
- Enable Dependabot alerts and security updates.
- Keep CodeQL enabled and confirm the `Analyze` check completes on pull requests/default-branch pushes.
- Restrict Actions to required sources and require approval for workflows from untrusted forks.
- Review configured secrets and environments; remove unused credentials and scope each credential
  to the minimum repository/environment permissions.

## Default-branch protection

Create a ruleset for the repository's current default branch rather than hard-coding a branch
name in documentation. Recommended controls:

- Require pull requests and at least one independent approval where staffing permits.
- Dismiss stale approvals after new commits.
- Require conversation resolution.
- Require the current CI checks (`quality`, `security`, `configuration`, and CodeQL `Analyze`).
- Require the branch to be up to date before merge when the project's merge volume permits it.
- Block force pushes and deletion.
- Do not permit routine administrator bypass; use time-bounded emergency exceptions with audit logs.

Check the actual check names in the Branch protection/ruleset UI after the first workflow run,
because GitHub may display workflow/job names differently from YAML job IDs.

## Releases and Pages

- Restrict release/environment deployment approvals to designated maintainers.
- Confirm GitHub Pages publishes only the intended static `docs/` content.
- Review repository social-preview assets and their provenance before publishing them.
- Verify release artifacts are generated from protected tags and do not include datasets, media,
  checkpoints, credentials, MLflow stores, or local environment files.

## Periodic verification

Review these settings at least quarterly and after ownership, plan, workflow, or default-branch
changes. Record the reviewer/date outside the codebase or in a governance issue; this file is a
checklist, not proof that any external control is enabled.
