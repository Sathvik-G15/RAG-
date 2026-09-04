const API = "/api";

async function request(path, options = {}) {
  const token = localStorage.getItem("token") || (await login());
  const res = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(options.headers || {}),
    },
  });
  if (res.status === 401) {
    const fresh = await login();
    const retry = await fetch(`${API}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${fresh}`,
      },
    });
    return retry.json();
  }
  return res.json();
}

async function login() {
  const res = await fetch(`${API}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: "doctor", password: "x" }),
  });
  const body = await res.json();
  localStorage.setItem("token", body.access_token);
  return body.access_token;
}

export function analyze(payload) {
  return request("/analyze", { method: "POST", body: JSON.stringify(payload) });
}

export function getMetrics() {
  return request("/metrics");
}

export function submitFeedback(payload) {
  return request("/feedback", { method: "POST", body: JSON.stringify(payload) });
}

export function fetchCorpusStats() {
  return request("/corpus/stats");
}
