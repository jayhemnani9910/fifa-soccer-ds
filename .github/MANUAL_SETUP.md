# Manual GitHub Settings Guide

These settings must be configured through the GitHub web UI. They cannot be set via config files or CLI.

---

## 1. Enable Secret Scanning + Push Protection

**Why:** Prevents accidentally pushing API keys, tokens, or passwords to the repo. GitHub scans every push and blocks commits containing known secret patterns.

**Steps:**

1. Go to https://github.com/jayhemnani9910/fifa-soccer-ds/settings/security_analysis
2. Under **"Secret scanning"**:
   - Click **Enable** next to "Secret scanning"
   - Click **Enable** next to "Push protection" (this blocks pushes that contain secrets)
3. Under **"Dependabot"**:
   - Ensure "Dependabot alerts" is **Enabled** (should already be on from our dependabot.yml)
   - Enable "Dependabot security updates" if available

**What happens after:** If you or a contributor tries to push code containing a token (e.g., `GITHUB_TOKEN=ghp_xxxx`), the push will be blocked with a clear message explaining which secret was found.

---

## 2. Enable Copilot Code Review

**Why:** When you or contributors open a PR, GitHub Copilot can automatically review the code, suggest improvements, and flag potential bugs -- before a human reviewer sees it.

**Requirements:** GitHub Copilot must be enabled on your account (free for public repos, or via Copilot subscription).

**Steps:**

1. Go to https://github.com/settings/copilot
2. Under **"Copilot in GitHub.com"**, ensure it's enabled
3. Go to https://github.com/jayhemnani9910/fifa-soccer-ds/settings
4. Scroll to **"Features"** section
5. Look for **"Copilot"** options:
   - Enable "Copilot code review" (if available)
   - This will make Copilot automatically comment on PRs

**Alternative (if not available in settings):**
- On any PR page, you can manually request a Copilot review by clicking the reviewers dropdown and selecting "Copilot"
- The `.github/copilot-instructions.md` file we created will guide Copilot's review based on your project conventions

---

## 3. Set Social Preview Image

**Why:** When someone shares your repo link on Twitter/LinkedIn/Slack, this image appears as the preview card. Without it, GitHub shows a generic gray card.

**Steps:**

1. Go to https://github.com/jayhemnani9910/fifa-soccer-ds/settings
2. Scroll down to **"Social preview"** section
3. Click **"Edit"**
4. Upload the `social-preview.png` image (saved in repo root, 1280x640px)
5. Click **"Save"**

**Verify it works:**
- Paste the repo URL into Twitter's card validator or LinkedIn's post inspector
- You should see your custom image instead of the generic GitHub card

---

## 4. Enable Branch Protection for `master`

**Why:** Prevents direct pushes to master, requiring all changes to go through PRs with passing CI. This is critical once you have contributors.

**Steps:**

1. Go to https://github.com/jayhemnani9910/fifa-soccer-ds/settings/branches
2. Click **"Add branch protection rule"** (or **"Add classic branch protection rule"**)
3. Set **"Branch name pattern"** to: `master`
4. Enable these checkboxes:
   - [x] **Require a pull request before merging**
     - [x] Require approvals: 1 (or 0 if solo project)
   - [x] **Require status checks to pass before merging**
     - Search and add: `lint` and `test` (these are the job names from our CI workflow)
   - [x] **Require branches to be up to date before merging**
   - [x] **Do not allow bypassing the above settings** (optional -- if you want to enforce even for yourself)
5. Click **"Create"**

**Note for solo projects:** If you're the only contributor, you might want to keep "Allow force pushes" enabled for convenience. You can always tighten it later when you have collaborators.

---

## Quick Links

| Setting | URL |
|---------|-----|
| Security & analysis | https://github.com/jayhemnani9910/fifa-soccer-ds/settings/security_analysis |
| Branch protection | https://github.com/jayhemnani9910/fifa-soccer-ds/settings/branches |
| General settings | https://github.com/jayhemnani9910/fifa-soccer-ds/settings |
| Copilot settings | https://github.com/settings/copilot |
| Actions settings | https://github.com/jayhemnani9910/fifa-soccer-ds/settings/actions |
