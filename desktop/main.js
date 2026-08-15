"use strict";

/**
 * OneClick Distill — Electron shell.
 *
 * Spawns the Python backend (dev: the project venv; packaged: the bundled
 * portable runtime), waits for it to become healthy, then shows the built-in
 * web UI. The backend is the single source of truth (FastAPI + WebSocket),
 * this shell only wraps it.
 *
 *   electron .                      run the desktop app
 *   electron . --screenshot <file>  load the UI, capture a PNG, then quit
 *                                   (used by the automated verification)
 */

const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");
const fs = require("fs");

const PORT = Number(process.env.OCD_PORT || 8080);
const HOST = "127.0.0.1";
const DEV = !app.isPackaged;

let backend = null;
let mainWindow = null;

function projectRoot() {
  return DEV ? path.resolve(__dirname, "..") : process.resourcesPath;
}

function findPython() {
  if (process.env.OCD_PYTHON) return process.env.OCD_PYTHON;
  const root = projectRoot();
  const candidates = DEV
    ? [
        path.join(root, ".venv", "Scripts", "python.exe"),
        path.join(root, ".venv", "bin", "python3"),
        path.join(root, ".venv", "bin", "python"),
      ]
    : [
        path.join(process.resourcesPath, "runtime", "python", "python.exe"),
        path.join(process.resourcesPath, "runtime", "python", "bin", "python3"),
        path.join(process.resourcesPath, "runtime", "python", "bin", "python3.12"),
      ];
  for (const c of candidates) if (fs.existsSync(c)) return c;
  return null;
}

function findLauncher() {
  const root = projectRoot();
  const dev = path.join(root, "server_launcher.py");
  if (fs.existsSync(dev)) return dev;
  const pkg = path.join(process.resourcesPath, "runtime", "launcher.py");
  return fs.existsSync(pkg) ? pkg : null;
}

function waitForHealth(url, timeoutMs) {
  return new Promise((resolve) => {
    const deadline = Date.now() + timeoutMs;
    const attempt = () => {
      const req = http.get(url, (res) => {
        res.resume();
        if (res.statusCode === 200) resolve(true);
        else schedule();
      });
      req.on("error", schedule);
      req.setTimeout(3000, () => req.destroy());
    };
    const schedule = () => {
      if (Date.now() > deadline) resolve(false);
      else setTimeout(attempt, 400);
    };
    attempt();
  });
}

function killTree(pid) {
  if (!pid) return;
  if (process.platform === "win32") {
    try {
      spawn("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true });
    } catch (_) {}
  } else {
    try {
      process.kill(pid, "SIGTERM");
    } catch (_) {}
  }
}

async function startBackend() {
  const py = findPython();
  const launcher = findLauncher();
  if (!py || !launcher) {
    const msg = `无法定位 Python 运行时${py ? "" : "（python）"}${launcher ? "" : "（launcher）"}，请先构建便携运行时或在项目根运行 .venv。`;
    console.error(msg);
    if (!DEV) dialogError(msg);
    return false;
  }
  console.log(`[backend] spawn ${py} ${launcher} ${HOST} ${PORT}`);
  backend = spawn(py, [launcher, HOST, String(PORT)], {
    cwd: DEV ? projectRoot() : process.resourcesPath,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  backend.stdout.on("data", (d) => console.log("[backend]", String(d).trim()));
  backend.stderr.on("data", (d) => console.log("[backend]", String(d).trim()));
  backend.on("exit", (code) => console.log("[backend] exited", code));
  const ok = await waitForHealth(`http://${HOST}:${PORT}/api/health`, 90000);
  console.log(`[backend] health: ${ok ? "ok" : "FAIL"}`);
  return ok;
}

function dialogError(msg) {
  const { dialog } = require("electron");
  dialog.showErrorBox("OneClick Distill 启动失败", msg);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 840,
    minWidth: 900,
    minHeight: 640,
    backgroundColor: "#0f1117",
    autoHideMenuBar: true,
    title: "OneClick Distill",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadURL(`http://${HOST}:${PORT}`);
  return mainWindow;
}

async function captureScreenshot(target) {
  await new Promise((r) => mainWindow.webContents.once("did-finish-load", r));
  // give the UI a beat to fetch hardware + start the metrics chart
  await new Promise((r) => setTimeout(r, 4000));
  const image = await mainWindow.webContents.capturePage();
  fs.writeFileSync(target, image.toPNG());
  console.log(`[shot] saved ${target} (${image.getSize().width}x${image.getSize().height})`);
}

app.whenReady().then(async () => {
  const ok = await startBackend();
  if (!ok) {
    if (DEV) console.error("[backend] failed to become healthy");
    app.exit(1);
    return;
  }
  createWindow();
  mainWindow.once("ready-to-show", () => mainWindow.show());

  const shotArg = process.argv.indexOf("--screenshot");
  if (shotArg >= 0) {
    const target = process.argv[shotArg + 1] || "screenshot.png";
    await captureScreenshot(target);
    app.exit(0);
    return;
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  killTree(backend && backend.pid);
  app.quit();
});

process.on("exit", () => killTree(backend && backend.pid));
