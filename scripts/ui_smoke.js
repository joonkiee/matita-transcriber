// Opens a running Matita in a browser and checks the page works (used by the CI builds).
//   node scripts/ui_smoke.js [msedge|chromium]
// On Windows it also checks the Windows wording and that the Mac-only parts are hidden.
const { chromium } = require("playwright");
(async () => {
  const channel = process.argv[2] === "msedge" ? "msedge" : undefined;
  const browser = await chromium.launch({ channel });
  const page = await browser.newPage({ viewport: { width: 1180, height: 800 } });
  const errors = [];
  page.on("pageerror", e => errors.push(String(e)));
  await page.goto("http://127.0.0.1:8765/");
  await page.waitForTimeout(2500);
  const r = await page.evaluate(() => ({
    title: document.title,
    jobs: document.querySelectorAll("#library .row, #library [data-id]").length,
    vmVisible: !!document.querySelector("#vmOpen") && getComputedStyle(document.querySelector("#vmOpen")).display !== "none",
    text: document.body.innerText,
  }));
  await page.keyboard.press("Control+K");
  await page.waitForTimeout(400);
  const palette = await page.evaluate(() => document.body.innerText.includes("Settings"));
  await page.screenshot({ path: "ui-smoke.png" });
  await browser.close();
  console.log(JSON.stringify({ title: r.title, jobs: r.jobs, vmVisible: r.vmVisible, palette, errors }));
  const win = process.platform === "win32";
  const bad = [];
  if (errors.length) bad.push("page errors: " + errors.join("; "));
  if (!palette) bad.push("the command palette didn't open");
  if (win && r.vmVisible) bad.push("Voice Memos is shown on Windows");
  if (win && /\b(this|your) Mac\b/.test(r.text)) bad.push("the page still says Mac");
  if (bad.length) { console.error(bad.join("\n")); process.exit(1); }
  console.log("ok");
})().catch(e => { console.error(e); process.exit(1); });
