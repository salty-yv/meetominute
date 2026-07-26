const { chromium } = require("playwright");

const baseUrl = process.argv[2] || "http://127.0.0.1:8765";
const outputDir = process.argv[3] || "benchmark-results";

async function inspectPage(page, name) {
  const layout = await page.evaluate(() => ({
    viewportWidth: document.documentElement.clientWidth,
    documentWidth: document.documentElement.scrollWidth,
    bodyWidth: document.body.scrollWidth,
    title: document.title,
  }));
  return {
    name,
    ...layout,
    horizontalOverflow:
      Math.max(layout.documentWidth, layout.bodyWidth) > layout.viewportWidth + 1,
  };
}

(async () => {
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const desktop = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    deviceScaleFactor: 1,
  });
  const page = await desktop.newPage();
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(error.message));

  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.screenshot({
    path: `${outputDir}/ui-home-desktop.png`,
    fullPage: true,
  });
  const results = [await inspectPage(page, "home-desktop")];
  const interactions = {};

  const historySearch = page.locator("[data-history-search]");
  if (await historySearch.count()) {
    await historySearch.fill("不会存在的会议名称");
    interactions.historyFilterShowsEmpty =
      await page.locator("[data-history-empty]").isVisible();
    await historySearch.fill("");
  }

  const meetingLink = page.locator("[data-meeting-row]").first();
  if (await meetingLink.count()) {
    await meetingLink.click();
    await page.waitForLoadState("networkidle");
    await page.screenshot({
      path: `${outputDir}/ui-meeting-desktop.png`,
      fullPage: true,
    });
    results.push(await inspectPage(page, "meeting-desktop"));

    const transcriptSearch = page.locator("[data-transcript-search]");
    if (await transcriptSearch.count()) {
      const firstText = await page
        .locator("[data-transcript-text]")
        .first()
        .inputValue();
      const query = firstText.trim().slice(0, 3);
      if (query) {
        await transcriptSearch.fill(query);
        const visibleCount = await page
          .locator("[data-transcript-segment]:visible")
          .count();
        const totalCount = await page
          .locator("[data-transcript-segment]")
          .count();
        interactions.transcriptFilterWorks =
          visibleCount > 0 && visibleCount <= totalCount;
        await transcriptSearch.fill("");
      }
    }

    const minutes = page.locator("#minutes");
    if (await minutes.count()) {
      await minutes.scrollIntoViewIfNeeded();
      await page.screenshot({
        path: `${outputDir}/ui-minutes-desktop.png`,
        fullPage: false,
      });
    }
  }

  await page.goto(`${baseUrl}/settings/external-llm`, {
    waitUntil: "networkidle",
  });
  await page.screenshot({
    path: `${outputDir}/ui-external-llm-desktop.png`,
    fullPage: true,
  });
  results.push(await inspectPage(page, "external-llm-desktop"));
  const secretToggle = page.locator("[data-toggle-secret]");
  if (await secretToggle.count()) {
    await secretToggle.click();
    interactions.secretVisibilityToggleWorks =
      (await page.locator("[data-secret-input]").getAttribute("type")) ===
      "text";
  }

  await page.goto(`${baseUrl}/diagnostics`, {
    waitUntil: "networkidle",
  });
  await page.screenshot({
    path: `${outputDir}/ui-diagnostics-desktop.png`,
    fullPage: true,
  });
  results.push(await inspectPage(page, "diagnostics-desktop"));
  interactions.diagnosticCardsRendered =
    (await page.locator(".diagnostic-card").count()) === 7;

  for (const [path, name] of [
    ["/calendar", "calendar-desktop"],
    ["/archive", "archive-desktop"],
    ["/trash", "trash-desktop"],
    ["/backups", "backups-desktop"],
  ]) {
    await page.goto(`${baseUrl}${path}`, { waitUntil: "networkidle" });
    await page.screenshot({
      path: `${outputDir}/ui-${name}.png`,
      fullPage: true,
    });
    results.push(await inspectPage(page, name));
  }

  const mobile = await browser.newContext({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 1,
    isMobile: true,
  });
  const mobilePage = await mobile.newPage();
  mobilePage.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  mobilePage.on("pageerror", (error) => consoleErrors.push(error.message));
  await mobilePage.goto(baseUrl, { waitUntil: "networkidle" });
  await mobilePage.screenshot({
    path: `${outputDir}/ui-home-mobile.png`,
    fullPage: true,
  });
  results.push(await inspectPage(mobilePage, "home-mobile"));

  const mobileMeetingLink = mobilePage.locator("[data-meeting-row]").first();
  if (await mobileMeetingLink.count()) {
    await mobileMeetingLink.click();
    await mobilePage.waitForLoadState("networkidle");
    await mobilePage.screenshot({
      path: `${outputDir}/ui-meeting-mobile.png`,
      fullPage: false,
    });
    results.push(await inspectPage(mobilePage, "meeting-mobile"));
  }

  await mobilePage.goto(`${baseUrl}/settings/external-llm`, {
    waitUntil: "networkidle",
  });
  await mobilePage.screenshot({
    path: `${outputDir}/ui-external-llm-mobile.png`,
    fullPage: true,
  });
  results.push(await inspectPage(mobilePage, "external-llm-mobile"));

  await mobilePage.goto(`${baseUrl}/diagnostics`, {
    waitUntil: "networkidle",
  });
  await mobilePage.screenshot({
    path: `${outputDir}/ui-diagnostics-mobile.png`,
    fullPage: true,
  });
  results.push(await inspectPage(mobilePage, "diagnostics-mobile"));

  await mobilePage.goto(`${baseUrl}/backups`, {
    waitUntil: "networkidle",
  });
  await mobilePage.screenshot({
    path: `${outputDir}/ui-backups-mobile.png`,
    fullPage: true,
  });
  results.push(await inspectPage(mobilePage, "backups-mobile"));

  await mobilePage.goto(`${baseUrl}/calendar`, {
    waitUntil: "networkidle",
  });
  await mobilePage.screenshot({
    path: `${outputDir}/ui-calendar-mobile.png`,
    fullPage: true,
  });
  results.push(await inspectPage(mobilePage, "calendar-mobile"));

  await browser.close();
  process.stdout.write(
    JSON.stringify({ results, interactions, consoleErrors }, null, 2),
  );
})().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
