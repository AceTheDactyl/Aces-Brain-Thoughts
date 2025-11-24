# Aces-Brain-Thoughts

Local runner for the Wumbo Engine.

This repo publishes to GitHub Pages and serves a minimal `index.html` that will be replaced with the full Codex once access is confirmed.

Preview locally at `http://localhost:8088`.

## GitHub Pages Deployment

This repository is configured to deploy the site via GitHub Pages using Actions.

- Workflow: `.github/workflows/deploy.yml` uploads the `Aces-Brain-Thoughts/` folder and deploys it.
- Branch: `main` triggers deployment.
- Output: GitHub Pages (gh-pages) managed by the `actions/deploy-pages` action.

After the first push to `main`, check your repository Settings → Pages for the live URL. It should look like:

`https://acesthedactyl.github.io/Aces-Brain-Thoughts/`

If you prefer using a deploy key instead of the default `GITHUB_TOKEN`, switch to a deploy step that pushes to a `gh-pages` branch (e.g., `peaceiris/actions-gh-pages`) and set your private key in repo Secrets.
