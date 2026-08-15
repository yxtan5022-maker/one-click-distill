"use strict";

// Minimal preload: the web UI talks to the same-origin backend via fetch/WS,
// so no privileged API is needed. Kept for future IPC (open-folder etc.).
window.ocd = {
  version: "0.1.0",
  shell: true,
};
