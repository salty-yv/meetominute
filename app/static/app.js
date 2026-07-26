const sleep = (milliseconds) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function showToast(message) {
  const region = document.querySelector("[data-toast-region]");
  if (!region) return;
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  region.appendChild(toast);
  window.requestAnimationFrame(() => toast.classList.add("visible"));
  window.setTimeout(() => {
    toast.classList.remove("visible");
    window.setTimeout(() => toast.remove(), 220);
  }, 2400);
}

async function pollStatus(element) {
  const url = element.dataset.pollUrl;
  if (!url) return;
  await sleep(1800);
  try {
    const response = await fetch(url, {
      headers: { "X-Requested-With": "fetch" },
      cache: "no-store",
    });
    if (!response.ok) return;
    const html = await response.text();
    element.outerHTML = html;
    const next = document.querySelector("#task-status[data-poll-url]");
    if (next) {
      pollStatus(next);
    } else {
      window.location.reload();
    }
  } catch {
    await sleep(3000);
    const current = document.querySelector("#task-status[data-poll-url]");
    if (current) pollStatus(current);
  }
}

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function initUploadForm() {
  const form = document.querySelector("[data-upload-form]");
  if (!form) return;

  const fileInput = form.querySelector("[data-file-input]");
  const fileDrop = form.querySelector("[data-file-drop]");
  const fileLabel = form.querySelector("[data-file-label]");
  const fileMeta = form.querySelector("[data-file-meta]");
  const originalMeta = fileMeta?.textContent || "";

  const updateFile = () => {
    const file = fileInput?.files?.[0];
    if (!fileLabel || !fileMeta || !fileDrop) return;
    fileLabel.textContent = file ? file.name : "拖入录音，或点击选择文件";
    fileMeta.textContent = file
      ? `${formatFileSize(file.size)} · 已准备上传`
      : originalMeta;
    fileDrop.classList.toggle("has-file", Boolean(file));
  };

  fileInput?.addEventListener("change", updateFile);
  ["dragenter", "dragover"].forEach((eventName) => {
    fileDrop?.addEventListener(eventName, () => fileDrop.classList.add("dragging"));
  });
  ["dragleave", "drop"].forEach((eventName) => {
    fileDrop?.addEventListener(eventName, () => fileDrop.classList.remove("dragging"));
  });

  const modeSelect = form.querySelector("[data-mode-select]");
  const modeNote = form.querySelector("[data-mode-note]");
  const notes = {
    local: "录音和逐字稿都保留在本机，适合私密会议。",
    mixed: "录音在本机转写，校正后的逐字稿会发送到配置的云端纪要服务。",
    cloud: "录音和逐字稿会发送到已配置的云端服务，请先确认数据许可。",
  };
  const updateModeNote = () => {
    if (modeNote && modeSelect) {
      const needsExternal =
        modeSelect.value !== "local" &&
        modeSelect.dataset.externalReady !== "true";
      modeNote.textContent = needsExternal
        ? "此模式需要外部 LLM；请先点击下方入口完成配置并测试连接。"
        : notes[modeSelect.value] || "";
    }
  };
  modeSelect?.addEventListener("change", updateModeNote);
  updateModeNote();

  form.addEventListener("submit", () => {
    const button = form.querySelector("[data-submit-button]");
    const label = form.querySelector("[data-button-label]");
    if (button) button.disabled = true;
    if (label) label.textContent = "正在上传并创建任务…";
  });
}

function initHistoryFilters() {
  const search = document.querySelector("[data-history-search]");
  const filter = document.querySelector("[data-history-filter]");
  const rows = Array.from(document.querySelectorAll("[data-meeting-row]"));
  const empty = document.querySelector("[data-history-empty]");
  if (!rows.length || (!search && !filter)) return;

  const update = () => {
    const query = search?.value.trim().toLocaleLowerCase("zh-CN") || "";
    const wantedStatus = filter?.value || "";
    let visible = 0;
    rows.forEach((row) => {
      const titleMatches = !query || row.dataset.title.includes(query);
      const rowStatus = row.dataset.status || "";
      const statusMatches =
        !wantedStatus ||
        rowStatus === wantedStatus ||
        (wantedStatus === "processing" &&
          ["generating_minutes", "canceling"].includes(rowStatus));
      const show = titleMatches && statusMatches;
      row.hidden = !show;
      if (show) visible += 1;
    });
    if (empty) empty.hidden = visible !== 0;
  };

  search?.addEventListener("input", update);
  filter?.addEventListener("change", update);
}

function parseSeekValue(value) {
  if (typeof value !== "string") return Number.NaN;
  if (!value.includes(":")) return Number.parseFloat(value);
  const parts = value.split(":").map((part) => Number.parseFloat(part));
  if (parts.some((part) => Number.isNaN(part))) return Number.NaN;
  return parts.reduce((total, part) => total * 60 + part, 0);
}

function initAudioNavigation() {
  const audio = document.querySelector("#meeting-audio");
  const seekButtons = Array.from(document.querySelectorAll("[data-seek]"));
  const segments = Array.from(document.querySelectorAll("[data-transcript-segment]"));

  seekButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (!audio) {
        showToast("原始录音暂不可用");
        return;
      }
      const target = parseSeekValue(button.dataset.seek || "");
      if (!Number.isFinite(target)) {
        showToast("这条内容没有可用的原文时间");
        return;
      }
      audio.currentTime = Math.max(0, target);
      audio.play().catch(() => {});
      const recording = document.querySelector("#recording");
      if (button.closest("#minutes")) {
        recording?.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  });

  if (!audio || !segments.length) return;
  let activeSegment = null;
  audio.addEventListener("timeupdate", () => {
    const current = audio.currentTime;
    const next = segments.find((segment) => {
      const start = Number.parseFloat(segment.dataset.start || "0");
      const end = Number.parseFloat(segment.dataset.end || String(start + 1));
      return current >= start && current < end;
    });
    if (next === activeSegment) return;
    activeSegment?.classList.remove("is-playing");
    next?.classList.add("is-playing");
    activeSegment = next || null;
  });
}

function resizeTextarea(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.max(72, textarea.scrollHeight)}px`;
}

function initTranscriptTools() {
  const search = document.querySelector("[data-transcript-search]");
  const filter = document.querySelector("[data-speaker-filter]");
  const segments = Array.from(document.querySelectorAll("[data-transcript-segment]"));
  const counter = document.querySelector("[data-visible-segments]");
  const empty = document.querySelector("[data-transcript-empty]");
  if (!segments.length) return;

  const update = () => {
    const query = search?.value.trim().toLocaleLowerCase("zh-CN") || "";
    const wantedSpeaker = filter?.value || "";
    let visible = 0;
    segments.forEach((segment) => {
      const text = segment.querySelector("[data-transcript-text]")?.value
        .toLocaleLowerCase("zh-CN") || "";
      const textMatches = !query || text.includes(query);
      const speakerMatches =
        !wantedSpeaker || segment.dataset.speaker === wantedSpeaker;
      const show = textMatches && speakerMatches;
      segment.hidden = !show;
      if (show) visible += 1;
    });
    if (counter) counter.textContent = String(visible);
    if (empty) empty.hidden = visible !== 0;
  };

  search?.addEventListener("input", update);
  filter?.addEventListener("change", update);
  segments.forEach((segment) => {
    const textarea = segment.querySelector("[data-transcript-text]");
    const speaker = segment.querySelector("[data-segment-speaker]");
    if (textarea) {
      resizeTextarea(textarea);
      textarea.addEventListener("input", () => {
        resizeTextarea(textarea);
        if (search?.value) update();
      });
    }
    speaker?.addEventListener("change", () => {
      segment.dataset.speaker = speaker.value;
      if (filter?.value) update();
    });
  });
}

function initDirtyForms() {
  const forms = Array.from(document.querySelectorAll("[data-dirty-form]"));
  const indicator = document.querySelector("[data-unsaved-indicator]");
  if (!forms.length) return;
  let dirty = false;
  let submitting = false;

  const markDirty = (event) => {
    if (!event.target.name) return;
    dirty = true;
    if (indicator) indicator.hidden = false;
  };

  forms.forEach((form) => {
    form.addEventListener("input", markDirty);
    form.addEventListener("change", markDirty);
    form.addEventListener("submit", () => {
      submitting = true;
      form.querySelectorAll("button[type='submit']").forEach((button) => {
        button.disabled = true;
      });
    });
  });

  window.addEventListener("beforeunload", (event) => {
    if (!dirty || submitting) return;
    event.preventDefault();
    event.returnValue = "";
  });
}

function initWorkspaceNavigation() {
  const sections = Array.from(document.querySelectorAll("[data-workspace-section]"));
  const links = Array.from(document.querySelectorAll("[data-section-link]"));
  if (!sections.length || !links.length || !("IntersectionObserver" in window)) return;

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      links.forEach((link) => {
        link.classList.toggle(
          "active",
          link.dataset.sectionLink === visible.target.id,
        );
      });
    },
    { rootMargin: "-18% 0px -68% 0px", threshold: [0, 0.1, 0.5] },
  );
  sections.forEach((section) => observer.observe(section));
}

function initCopySummary() {
  const button = document.querySelector("[data-copy-summary]");
  if (!button) return;
  button.addEventListener("click", async () => {
    const text = button.dataset.copyText || "";
    try {
      await navigator.clipboard.writeText(text);
      showToast("会议摘要已复制");
    } catch {
      const temporary = document.createElement("textarea");
      temporary.value = text;
      document.body.appendChild(temporary);
      temporary.select();
      document.execCommand("copy");
      temporary.remove();
      showToast("会议摘要已复制");
    }
  });
}

function initDiagnostics() {
  const button = document.querySelector("[data-copy-diagnostics]");
  if (!button) return;

  button.addEventListener("click", async () => {
    const label = button.querySelector("span");
    const previousLabel = label?.textContent || "复制诊断信息";
    button.disabled = true;
    if (label) label.textContent = "正在重新检查…";
    try {
      const response = await fetch(button.dataset.apiUrl || "/api/diagnostics", {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`诊断接口返回 ${response.status}`);
      const payload = await response.json();
      const text = `MeetOminute 运行诊断\n${JSON.stringify(payload, null, 2)}`;
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        const temporary = document.createElement("textarea");
        temporary.value = text;
        document.body.appendChild(temporary);
        temporary.select();
        document.execCommand("copy");
        temporary.remove();
      }
      showToast("诊断信息已复制");
    } catch (error) {
      showToast(`复制失败：${String(error)}`);
    } finally {
      button.disabled = false;
      if (label) label.textContent = previousLabel;
    }
  });
}

function initExternalLLMSettings() {
  const form = document.querySelector("[data-external-llm-form]");
  if (!form) return;

  const secretInput = form.querySelector("[data-secret-input]");
  const toggleSecret = form.querySelector("[data-toggle-secret]");
  toggleSecret?.addEventListener("click", () => {
    if (!secretInput) return;
    const show = secretInput.type === "password";
    secretInput.type = show ? "text" : "password";
    toggleSecret.classList.toggle("showing", show);
  });

  const testButton = form.querySelector("[data-test-external-llm]");
  const result = form.querySelector("[data-connection-result]");
  const resultIcon = result?.querySelector("[data-result-icon]");
  const resultTitle = result?.querySelector("[data-result-title]");
  const resultMessage = result?.querySelector("[data-result-message]");
  const resultLatency = result?.querySelector("[data-result-latency]");

  testButton?.addEventListener("click", async () => {
    if (!form.reportValidity()) return;
    testButton.disabled = true;
    const label = testButton.querySelector("span");
    const previousLabel = label?.textContent || "测试连接";
    if (label) label.textContent = "正在连接…";
    if (result) result.hidden = true;

    try {
      const response = await fetch("/settings/external-llm/test", {
        method: "POST",
        body: new FormData(form),
        headers: { "X-Requested-With": "fetch" },
      });
      const payload = await response.json();
      const status = payload.status || "error";
      if (result) {
        result.hidden = false;
        result.className = `connection-result ${status}`;
      }
      if (resultIcon) {
        resultIcon.textContent =
          status === "success" ? "✓" : status === "warning" ? "!" : "×";
      }
      if (resultTitle) {
        resultTitle.textContent =
          status === "success"
            ? "连接成功"
            : status === "warning"
              ? "连接成功，但需要确认"
              : "连接失败";
      }
      if (resultMessage) resultMessage.textContent = payload.message || "";
      if (resultLatency) {
        resultLatency.textContent = payload.latency_ms
          ? `${payload.latency_ms} ms`
          : "";
      }
    } catch (error) {
      if (result) {
        result.hidden = false;
        result.className = "connection-result error";
      }
      if (resultIcon) resultIcon.textContent = "×";
      if (resultTitle) resultTitle.textContent = "测试请求失败";
      if (resultMessage) resultMessage.textContent = String(error);
      if (resultLatency) resultLatency.textContent = "";
    } finally {
      testButton.disabled = false;
      if (label) label.textContent = previousLabel;
    }
  });
}

function initConfirmActions() {
  document.addEventListener("submit", (event) => {
    const form = event.target.closest?.("[data-confirm-message]");
    if (!form) return;
    const message = form.dataset.confirmMessage || "确定继续吗？";
    if (!window.confirm(message)) event.preventDefault();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const status = document.querySelector("#task-status[data-poll-url]");
  if (status) pollStatus(status);

  initUploadForm();
  initHistoryFilters();
  initAudioNavigation();
  initTranscriptTools();
  initDirtyForms();
  initWorkspaceNavigation();
  initCopySummary();
  initDiagnostics();
  initExternalLLMSettings();
  initConfirmActions();
});
