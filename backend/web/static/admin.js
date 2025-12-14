// ===== Shared helpers =====
const TOKEN_KEY = "hr_admin_token";
const USER_KEY  = "hr_admin_user";

function $(id){ return document.getElementById(id); }

function getToken(){ return localStorage.getItem(TOKEN_KEY); }
function setToken(t){ localStorage.setItem(TOKEN_KEY, t); }
function clearToken(){ localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(USER_KEY); }

function setUser(u){ localStorage.setItem(USER_KEY, JSON.stringify(u)); }
function getUser(){ try { return JSON.parse(localStorage.getItem(USER_KEY) || "null"); } catch { return null; } }

function showErr(msg){
  const el = $("err");
  if(!el) return;
  el.textContent = msg;
  el.classList.remove("d-none");
}

async function apiFetch(path, opts={}){
  const token = getToken();
  const headers = opts.headers || {};
  headers["Content-Type"] = headers["Content-Type"] || "application/json";
  if(token) headers["Authorization"] = `Bearer ${token}`;
  opts.headers = headers;

  const resp = await fetch(path, opts);
  if(resp.status === 401 || resp.status === 403){
    clearToken();
    location.href = "/admin-web/login";
    throw new Error("Unauthorized");
  }
  if(!resp.ok){
    let detail = `${resp.status}`;
    try{
      const j = await resp.json();
      detail = j.detail ? String(j.detail) : JSON.stringify(j);
    }catch{
      try{ detail = await resp.text(); }catch{}
    }
    throw new Error(detail);
  }
  return resp;
}

function fmtTime(iso){
  if(!iso) return "-";
  const d = new Date(iso);
  const pad = (n)=> String(n).padStart(2,"0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function pad2(n){ return String(n).padStart(2,"0"); }

// ===== Login page =====
function initLogin(){
  const form = $("loginForm");
  if(!form) return;

  form.addEventListener("submit", async (e)=>{
    e.preventDefault();
    const email = $("email").value.trim();
    const password = $("password").value;

    try{
      const resp = await fetch("/auth/login", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({email, password})
      });
      if(!resp.ok){
        const j = await resp.json().catch(()=>({detail:"登入失敗"}));
        throw new Error(j.detail || "登入失敗");
      }
      const data = await resp.json();
      if(!data.token || !data.user) throw new Error("後端回傳格式不正確");
      if(data.user.role !== "admin") throw new Error("此帳號不是管理員，無法進入後台");

      setToken(data.token);
      setUser(data.user);
      location.href = "/admin-web/dashboard";
    }catch(err){
      showErr(err.message || String(err));
    }
  });
}

// ===== Common navbar =====
function initNavbar(){
  const token = getToken();
  if(!token){
    // Only redirect if it's dashboard/payroll pages
    if($("attendanceTbody") || $("payrollTbody")) location.href = "/admin-web/login";
    return;
  }
  const u = getUser();
  if(u && $("who")){
    $("who").textContent = `${u.name}（${u.email}）`;
  }
  $("logoutBtn")?.addEventListener("click", ()=>{
    clearToken();
    location.href = "/admin-web/login";
  });
}

// ===== Attendance dashboard =====
let attendanceRows = [];

function computeAttendanceCards(rows){
  const total = rows.length;
  const absent = rows.filter(r=> r.status === "未打卡").length;
  const missingOut = rows.filter(r=> (r.status || "").includes("未下班")).length;
  const late = rows.filter(r=> r.late).length;

  const otMinutes = rows.reduce((s,r)=> s + (r.overtime_minutes || 0), 0);
  const workedMinutes = rows.reduce((s,r)=> s + (r.worked_minutes || 0), 0);
  const avgHours = total ? (workedMinutes/60/total) : 0;

  $("c_total").textContent = total;
  $("c_absent").textContent = absent;
  $("c_missingOut").textContent = missingOut;
  $("c_late").textContent = late;
  $("c_ot").textContent = (otMinutes/60).toFixed(2);
  $("c_avg").textContent = avgHours.toFixed(2);
}

function badgeForStatus(status){
  if(status === "未打卡") return `<span class="badge badge-soft-danger text-danger fw-semibold">${status}</span>`;
  if(status.includes("未下班") || status.includes("異常")) return `<span class="badge badge-soft-warn text-warning fw-semibold">${status}</span>`;
  return `<span class="badge badge-soft-ok text-success fw-semibold">${status}</span>`;
}

function renderAttendanceTable(rows){
  const q = ($("searchInput")?.value || "").trim().toLowerCase();
  const filtered = !q ? rows : rows.filter(r =>
    (r.name||"").toLowerCase().includes(q) ||
    (r.email||"").toLowerCase().includes(q) ||
    (r.dept||"").toLowerCase().includes(q)
  );

  const tbody = $("attendanceTbody");
  if(!tbody) return;

  if(filtered.length === 0){
    tbody.innerHTML = `<tr><td colspan="10" class="text-center text-secondary py-4">沒有資料</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(r=>{
    const dept = r.dept || "-";
    const wage = r.hourly_rate ?? 200;

    const rowClass = (r.status === "未打卡") ? "table-danger" :
                     ((r.status || "").includes("未下班") || (r.status || "").includes("異常")) ? "table-warning" : "";

    return `<tr class="${rowClass}">
      <td>${dept}</td>
      <td class="fw-semibold">${r.name}</td>
      <td class="text-secondary">${r.email}</td>
      <td>
        <input class="form-control form-control-sm wage-input" type="number" min="0" max="100000"
               value="${wage}" data-user-id="${r.user_id}">
      </td>
      <td>${fmtTime(r.first_in)}</td>
      <td>${fmtTime(r.last_out)}</td>
      <td class="text-end">${(r.worked_hours ?? 0).toFixed(2)}</td>
      <td class="text-end">${(r.overtime_hours ?? 0).toFixed(2)}</td>
      <td>${badgeForStatus(r.status)}</td>
      <td>${r.late ? `<span class="text-warning fw-semibold">是</span>` : `<span class="text-secondary">否</span>`}</td>
    </tr>`;
  }).join("");

  tbody.querySelectorAll(".wage-input").forEach(inp=>{
    inp.addEventListener("change", async (e)=>{
      const userId = e.target.dataset.userId;
      const hourly_rate = parseInt(e.target.value || "0", 10);

      try{
        await apiFetch(`/admin/users/${userId}/hourly_rate`, {
          method: "PATCH",
          body: JSON.stringify({hourly_rate})
        });
      }catch(err){
        showErr(`更新時薪失敗：${err.message || err}`);
      }
    });
  });
}

async function loadAttendance(){
  try{
    $("err")?.classList.add("d-none");
    const date = $("dateInput")?.value;
    const qs = date ? `?date=${encodeURIComponent(date)}` : "";
    const resp = await apiFetch(`/admin/attendance/daily${qs}`);
    const data = await resp.json();
    attendanceRows = data || [];
    computeAttendanceCards(attendanceRows);
    renderAttendanceTable(attendanceRows);
  }catch(err){
    showErr(err.message || String(err));
  }
}

async function exportAttendanceCSV(){
  try{
    const date = $("dateInput")?.value;
    const qs = date ? `?date=${encodeURIComponent(date)}` : "";
    const resp = await apiFetch(`/admin/attendance/export/csv${qs}`, {
      method: "GET",
      headers: {"Content-Type":"text/plain"}
    });
    const text = await resp.text();
    const blob = new Blob([text], {type: "text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `attendance_${date || "today"}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }catch(err){
    showErr(`匯出失敗：${err.message || err}`);
  }
}

function initAttendance(){
  const tbody = $("attendanceTbody");
  if(!tbody) return;

  const di = $("dateInput");
  if(di && !di.value){
    const now = new Date();
    di.value = `${now.getFullYear()}-${pad2(now.getMonth()+1)}-${pad2(now.getDate())}`;
  }

  $("refreshBtn")?.addEventListener("click", loadAttendance);
  $("exportAttendanceBtn")?.addEventListener("click", exportAttendanceCSV);
  $("searchInput")?.addEventListener("input", ()=> renderAttendanceTable(attendanceRows));
  $("dateInput")?.addEventListener("change", loadAttendance);

  loadAttendance();
}

// ===== Payroll (monthly) =====
let payrollRows = [];

function computePayrollCards(rows){
  const total = rows.length;
  const hours = rows.reduce((s,r)=> s + (r.regular_hours || 0), 0);
  const ot = rows.reduce((s,r)=> s + (r.overtime_hours || 0), 0);
  const pay = rows.reduce((s,r)=> s + (r.total_pay || 0), 0);

  $("p_total").textContent = total;
  $("p_hours").textContent = hours.toFixed(2);
  $("p_ot").textContent = ot.toFixed(2);
  $("p_totalPay").textContent = pay.toLocaleString();
}

function renderPayrollTable(rows){
  const q = ($("payrollSearchInput")?.value || "").trim().toLowerCase();
  const filtered = !q ? rows : rows.filter(r =>
    (r.name||"").toLowerCase().includes(q) ||
    (r.email||"").toLowerCase().includes(q) ||
    (r.dept||"").toLowerCase().includes(q)
  );

  const tbody = $("payrollTbody");
  if(!tbody) return;

  if(filtered.length === 0){
    tbody.innerHTML = `<tr><td colspan="9" class="text-center text-secondary py-4">沒有資料</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(r=>{
    const dept = r.dept || "-";
    const wage = r.hourly_rate ?? 200;
    return `<tr>
      <td>${dept}</td>
      <td class="fw-semibold">${r.name}</td>
      <td class="text-secondary">${r.email}</td>
      <td>
        <input class="form-control form-control-sm wage-input" type="number" min="0" max="100000"
               value="${wage}" data-user-id="${r.user_id}">
      </td>
      <td class="text-end">${(r.regular_hours ?? 0).toFixed(2)}</td>
      <td class="text-end">${(r.overtime_hours ?? 0).toFixed(2)}</td>
      <td class="text-end">${(r.regular_pay ?? 0).toLocaleString()}</td>
      <td class="text-end">${(r.overtime_pay ?? 0).toLocaleString()}</td>
      <td class="text-end fw-semibold">${(r.total_pay ?? 0).toLocaleString()}</td>
    </tr>`;
  }).join("");

  tbody.querySelectorAll(".wage-input").forEach(inp=>{
    inp.addEventListener("change", async (e)=>{
      const userId = e.target.dataset.userId;
      const hourly_rate = parseInt(e.target.value || "0", 10);
      try{
        await apiFetch(`/admin/users/${userId}/hourly_rate`, {
          method: "PATCH",
          body: JSON.stringify({hourly_rate})
        });
        // 更新後重新算一次月薪
        loadPayroll();
      }catch(err){
        showErr(`更新時薪失敗：${err.message || err}`);
      }
    });
  });
}

async function loadPayroll(){
  try{
    $("err")?.classList.add("d-none");
    const m = $("monthInput")?.value; // YYYY-MM
    const qs = m ? `?month=${encodeURIComponent(m)}` : "";
    const resp = await apiFetch(`/admin/payroll/monthly${qs}`);
    const data = await resp.json();
    payrollRows = data || [];
    computePayrollCards(payrollRows);
    renderPayrollTable(payrollRows);
  }catch(err){
    showErr(err.message || String(err));
  }
}

async function exportPayrollCSV(){
  try{
    const m = $("monthInput")?.value;
    const qs = m ? `?month=${encodeURIComponent(m)}` : "";
    const resp = await apiFetch(`/admin/payroll/export/csv${qs}`, {
      method: "GET",
      headers: {"Content-Type":"text/plain"}
    });
    const text = await resp.text();
    const blob = new Blob([text], {type: "text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `payroll_${m || "this_month"}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }catch(err){
    showErr(`匯出失敗：${err.message || err}`);
  }
}

function initPayroll(){
  const tbody = $("payrollTbody");
  if(!tbody) return;

  const mi = $("monthInput");
  if(mi && !mi.value){
    const now = new Date();
    mi.value = `${now.getFullYear()}-${pad2(now.getMonth()+1)}`;
  }

  $("refreshPayrollBtn")?.addEventListener("click", loadPayroll);
  $("exportPayrollBtn")?.addEventListener("click", exportPayrollCSV);
  $("payrollSearchInput")?.addEventListener("input", ()=> renderPayrollTable(payrollRows));
  $("monthInput")?.addEventListener("change", loadPayroll);

  loadPayroll();
}

// boot
initLogin();
initNavbar();
initAttendance();
initPayroll();
