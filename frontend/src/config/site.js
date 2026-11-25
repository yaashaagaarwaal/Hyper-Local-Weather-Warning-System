// Project-level constants for the public home page.
//
// GITHUB_URL: set VITE_GITHUB_URL in frontend/.env(.local) to change it; the
// default is this repository's configured git remote. Contact details are
// deliberately not invented: there is no project mailbox configured, so the
// footer points at the repository and the problem statement instead.
export const SITE = {
  name: "AtmoGuard",
  tagline: "Hyper-Local Weather Warning System",
  problemStatement: "Smart India Hackathon · Problem Statement 26077",
  githubUrl: import.meta.env.VITE_GITHUB_URL || "https://github.com/yaashaagaarwaal/Hyper-Local-Weather-Warning-System",
};
