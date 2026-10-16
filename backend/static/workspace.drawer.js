/* ==========================================================================
   AutoML Workspace — persistent assistant drawer
   A second, lightweight view of the same session and websocket, so the user
   can talk to the agent without losing the page they are on.
   ========================================================================== */

function drawerThread() { return $("drawerChatThread"); }

function drawerRenderHistory(msgs) {
  const thread = drawerThread();
  if (!thread) return;
  thread.innerHTML = "";
  if (!msgs.length) {
    thread.innerHTML = '<div class="msg sys"><div class="sysline">No messages in this session yet.</div></div>';
    return;
  }
  msgs.slice(-40).forEach((m) => {
    if (m.role === "user") {
      thread.insertAdjacentHTML("beforeend", '<div class="msg u"><div class="bubble"></div></div>');
      thread.lastElementChild.firstElementChild.textContent = m.text;
    } else if (m.role === "agent") {
      thread.insertAdjacentHTML("beforeend",
        '<div class="msg a"><div class="bubble"><div class="who"><i aria-hidden="true">◈</i>' +
        esc(AGENTS[m.agent] || "Agent") + '</div><div class="txt">' + richText(m.text) + "</div></div></div>");
    } else {
      thread.insertAdjacentHTML("beforeend", '<div class="msg sys"><div class="sysline">' + esc(m.text) + "</div></div>");
    }
  });
  thread.scrollTop = thread.scrollHeight;
}

async function openChatDrawer() {
  const d = $("chatDrawer");
  d.classList.add("on");
  d.setAttribute("aria-hidden", "false");
  $("scrim").classList.add("on");
  pushOverlay("chatdrawer");
  const thread = drawerThread();
  thread.innerHTML = '<div class="loading-note"><span class="spin" aria-hidden="true"></span>Loading conversation…</div>';
  if (!sel.sessionId) await newSession();
  if (!sel.sessionId) {
    thread.innerHTML = '<div class="msg sys"><div class="sysline">No session is available for this project.</div></div>';
    return;
  }
  try {
    const msgs = (await api("/sessions/" + sel.sessionId + "/messages?limit=50")).messages || [];
    drawerRenderHistory(msgs);
  } catch (e) {
    thread.innerHTML = '<div class="msg sys"><div class="sysline">Could not load history: ' + esc(e.message) + "</div></div>";
  }
  const i = $("drawerChatInput");
  if (i) i.focus();
}

function closeChatDrawer() {
  const d = $("chatDrawer");
  d.classList.remove("on");
  d.setAttribute("aria-hidden", "true");
  popOverlay("chatdrawer");
  if (!overlays.length) $("scrim").classList.remove("on");
}

function drawerSend() {
  const input = $("drawerChatInput");
  const text = (input.value || "").trim();
  if (!text) return;
  if (!sel.projectId) { toast("Select a project first.", "err"); return; }
  const thread = drawerThread();
  thread.insertAdjacentHTML("beforeend", '<div class="msg u"><div class="bubble"></div></div>');
  thread.lastElementChild.firstElementChild.textContent = text;
  thread.scrollTop = thread.scrollHeight;
  input.value = "";
  if (!sendWS({ event: "chat", message: text, project_id: sel.projectId, session_id: sel.sessionId })) {
    thread.insertAdjacentHTML("beforeend",
      '<div class="msg sys"><div class="sysline">Not connected yet — retrying…</div></div>');
  }
}

/** Agent messages arriving over the socket also appear in the open drawer. */
function drawerNote(text, agent) {
  const d = $("chatDrawer");
  if (!d || !d.classList.contains("on")) return;
  const thread = drawerThread();
  if (!thread) return;
  thread.insertAdjacentHTML("beforeend",
    '<div class="msg a"><div class="bubble"><div class="who"><i aria-hidden="true">◈</i>' +
    esc(AGENTS[agent] || "Agent") + '</div><div class="txt">' + richText(text) + "</div></div></div>");
  thread.scrollTop = thread.scrollHeight;
}
