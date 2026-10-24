import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, ArrowUpRight, CheckCircle2, ChevronRight, CircleHelp, Download, FileCheck2, ShieldCheck, Sparkles, TrendingDown, TrendingUp } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ErrorBar, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import "./styles.css";

const initial = {
  annual_income: 82000, credit_score: 718, debt_to_income: 0.28,
  employment_years: 6, loan_amount: 24000, savings_balance: 42000,
  employment_type: "Salaried", home_ownership: "Mortgage", loan_purpose: "Home improvement",
};
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const queryParams = new URLSearchParams(window.location.search);
const researcherMode = queryParams.get("researcher") === "1" || queryParams.has("condition");
const REVIEWER_ID_STORAGE_KEY = "clearcredit_reviewer_id";

function createReviewerId() {
  return `reviewer-${crypto.randomUUID().slice(0, 8)}`;
}

function loadReviewerId() {
  try {
    const storedId = window.localStorage.getItem(REVIEWER_ID_STORAGE_KEY);
    if (storedId) return storedId;
    const newId = createReviewerId();
    window.localStorage.setItem(REVIEWER_ID_STORAGE_KEY, newId);
    return newId;
  } catch {
    return createReviewerId();
  }
}

function startNewReviewerSession() {
  try {
    window.localStorage.removeItem(REVIEWER_ID_STORAGE_KEY);
  } catch {
    // Continue with a fresh in-memory session when storage is unavailable.
  }
  window.location.reload();
}

function formatLocalValue(feature, value) {
  const numericValue = Number(value);
  if (feature === "Credit score") return String(Math.round(numericValue));
  if (feature === "Annual income" || feature === "Requested loan amount" || feature === "Savings balance") {
    return `$${(Math.round(numericValue / 100) * 100).toLocaleString()}`;
  }
  if (feature === "Debt-to-income ratio") return `${(numericValue * 100).toFixed(1)}%`;
  if (feature === "Employment history") return `${numericValue.toFixed(1)} yrs`;
  return value;
}

function App() {
  const [form, setForm] = useState(initial);
  const [reviewerId, setReviewerId] = useState(loadReviewerId);
  const [mode, setMode] = useState("study");
  const [condition, setCondition] = useState(null);
  const [caseId, setCaseId] = useState(null);
  const [caseNumber, setCaseNumber] = useState(1);
  const [totalCases, setTotalCases] = useState(12);
  const [stageStartedAt, setStageStartedAt] = useState(() => Date.now());
  const [initialSeconds, setInitialSeconds] = useState(0);
  const [initialDecision, setInitialDecision] = useState("Approve");
  const [initialConfidence, setInitialConfidence] = useState(3);
  const [finalDecision, setFinalDecision] = useState("Approve");
  const [finalConfidence, setFinalConfidence] = useState(3);
  const [result, setResult] = useState(null);
  const [stage, setStage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [error, setError] = useState("");
  const [page, setPage] = useState("review");
  const update = (key, value) => setForm({ ...form, [key]: value });
  useEffect(() => {
    setError("");
    const conditionQuery = queryParams.get("condition");
    const conditionUrl = conditionQuery
      ? `${API}/api/session/${encodeURIComponent(reviewerId)}/condition?condition=${encodeURIComponent(conditionQuery)}`
      : `${API}/api/session/${encodeURIComponent(reviewerId)}/condition`;
    fetch(conditionUrl)
      .then(response => response.ok ? response.json() : Promise.reject(new Error("Unable to assign study condition.")))
      .then(data => {
        setCondition(data.condition);
        setStageStartedAt(Date.now());
        if (mode !== "study") return;
        return fetch(`${API}/api/study/next-case?session_id=${encodeURIComponent(reviewerId)}`)
          .then(response => response.ok ? response.json() : Promise.reject(new Error("Unable to load the next study case.")))
          .then(studyCase => {
            if (studyCase.done) {
              setCompleted(true);
              return;
            }
            setForm(studyCase.applicant);
            setCaseId(studyCase.case_id);
            setCaseNumber(studyCase.case_number);
            setTotalCases(studyCase.total_cases);
          });
      })
      .catch(requestError => setError(requestError.message));
  }, [reviewerId, mode]);
  const applicant = {
    ...form,
    annual_income: Number(form.annual_income), credit_score: Number(form.credit_score),
    debt_to_income: Number(form.debt_to_income), employment_years: Number(form.employment_years),
    loan_amount: Number(form.loan_amount), savings_balance: Number(form.savings_balance),
  };

  async function revealAI(event) {
    event.preventDefault();
    setLoading(true); setError(""); setSaved(false);
    try {
      const conditionResponse = await fetch(`${API}/api/session/${encodeURIComponent(reviewerId)}/condition`);
      if (!conditionResponse.ok) throw new Error("Unable to assign study condition.");
      const assigned = await conditionResponse.json();
      setCondition(assigned.condition);
      setInitialSeconds(Math.max(0, Math.round((Date.now() - stageStartedAt) / 1000)));
      const decideQuery = mode === "study"
        ? `?session_id=${encodeURIComponent(reviewerId)}&case_id=${encodeURIComponent(caseId)}`
        : "";
      const response = await fetch(`${API}/api/decide${decideQuery}`, {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(applicant),
      });
      if (!response.ok) throw new Error("Unable to reach the decision service.");
      const decision = await response.json();
      setResult(decision);
      setFinalDecision(decision.decision === "Approved" ? "Approve" : "Decline");
      setStage(2);
    } catch (requestError) {
      setError(requestError.message);
    } finally { setLoading(false); }
  }

  async function saveReview() {
    if (mode !== "study") {
      setSaved(true);
      return;
    }
    setLoading(true); setError("");
    try {
      const response = await fetch(`${API}/api/reviews`, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          session_id: reviewerId,
          case_id: mode === "study" ? caseId : null,
          initial_decision: initialDecision,
          initial_confidence: Number(initialConfidence),
          final_decision: finalDecision,
          final_confidence: Number(finalConfidence),
          initial_seconds: initialSeconds,
          final_seconds: Math.max(0, Math.round((Date.now() - stageStartedAt) / 1000)),
        }),
      });
      if (!response.ok) throw new Error("The review could not be saved.");
      setSaved(true);
      if (caseNumber >= totalCases) setCompleted(true);
    } catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
  }

  if (page === "dashboard") return <Dashboard onBack={() => setPage("review")} />;
  if (page === "fairness") return <FairnessPage onBack={() => setPage("review")} />;
  if (completed) return <div className="app"><nav><div className="brand"><span className="brand-mark"><Sparkles size={17}/></span> clear<span>credit</span></div><div className="nav-meta"><span className="live-dot"/> Model service online <button className="nav-link" onClick={() => setPage("dashboard")}>Trust dashboard</button></div></nav><main><section className="card empty"><div className="empty-orb"><CheckCircle2 size={26}/></div><h2>Thank you for completing the study.</h2><p>You reviewed all {totalCases} assigned cases. Your responses have been recorded.</p>{researcherMode && <button className="secondary" onClick={startNewReviewerSession}>New reviewer session</button>}</section></main></div>;
  return <div className="app">
    <nav><div className="brand"><span className="brand-mark"><Sparkles size={17}/></span> clear<span>credit</span></div><div className="nav-meta"><span className="live-dot"/> Model service online <button className="nav-link" onClick={() => setPage("dashboard")}>Trust dashboard</button><button className="nav-link" onClick={() => setPage("fairness")}>Fairness</button>{researcherMode && <button className="nav-link" onClick={startNewReviewerSession}>New reviewer session</button>}<button className="icon-button"><CircleHelp size={17}/></button></div></nav>
    <main>
      <header className="hero"><div><div className="eyebrow">TRUST CALIBRATION / LOAN REVIEW</div><h1>Make decisions<br/><em>people can trust.</em></h1><p>Record a human decision before revealing the model, then compare how local explanations influence reviewer confidence.</p></div><div className="hero-badge"><ShieldCheck size={20}/><div><strong>Two-stage study flow</strong><small>Human judgment before AI</small></div></div></header>
      <div className="workspace">
        <section className="card form-card"><div className="card-heading"><div><span className="step">01</span><div><h2>Applicant profile</h2><p>{stage === 1 ? "Review the profile before seeing AI output." : "Profile locked for this review."}</p></div></div><FileCheck2 size={21}/></div>
          <form onSubmit={revealAI}>
            <div className="form-grid">
              <Field label="Annual income" prefix="$" value={form.annual_income} disabled={mode === "study" || stage === 2} onChange={v=>update("annual_income", v)} />
              <Field label="Credit score" value={form.credit_score} disabled={mode === "study" || stage === 2} onChange={v=>update("credit_score", v)} />
              <Field label="Debt-to-income ratio" suffix="%" value={Math.round(form.debt_to_income*100)} disabled={mode === "study" || stage === 2} onChange={v=>update("debt_to_income", Number(v)/100)} />
              <Field label="Employment history" suffix="yrs" value={form.employment_years} disabled={mode === "study" || stage === 2} onChange={v=>update("employment_years", v)} />
              <Field label="Requested loan" prefix="$" value={form.loan_amount} disabled={mode === "study" || stage === 2} onChange={v=>update("loan_amount", v)} />
              <Field label="Savings balance" prefix="$" value={form.savings_balance} disabled={mode === "study" || stage === 2} onChange={v=>update("savings_balance", v)} />
            </div>
            <div className="select-grid"><Select label="Employment type" value={form.employment_type} disabled={mode === "study" || stage === 2} options={["Salaried","Self-employed","Contract"]} onChange={v=>update("employment_type",v)}/><Select label="Home ownership" value={form.home_ownership} disabled={mode === "study" || stage === 2} options={["Own","Mortgage","Rent"]} onChange={v=>update("home_ownership",v)}/><Select label="Loan purpose" value={form.loan_purpose} disabled={mode === "study" || stage === 2} options={["Home improvement","Education","Vehicle","Debt consolidation"]} onChange={v=>update("loan_purpose",v)}/></div>
            <div className="study-protocol"><div className="section-title"><span>Review mode</span><small>{mode === "study" ? `Case ${caseNumber} of ${totalCases}` : "Manual exploration"}</small></div><div className="condition-toggle">{[["study","Study"],["explore","Explore"]].map(([value,label])=><button type="button" key={value} className={mode === value ? "selected" : ""} disabled={stage === 2} onClick={()=>{setMode(value);setStage(1);setResult(null);setSaved(false)}}>{label}</button>)}</div>{mode === "study" ? <><p>By continuing, you consent to this synthetic research demo. Review the assigned profile, decide before AI output, then record your final decision.</p>{researcherMode && <small className="assigned-condition">Reviewer: {reviewerId}<br />{condition ? `Assigned condition: ${condition === "prediction_lime" ? "Prediction + LIME explanation" : "Prediction only"}` : "Assigning study condition..."}</small>}</> : <p>Explore mode allows manual profiles. Explore reviews are not included in study data.</p>}</div>
            {stage === 1 && <div className="review-inputs"><div className="section-title"><span>Your initial decision</span><small>Before AI output</small></div><div className="decision-toggle">{["Approve","Decline"].map(value=><button type="button" key={value} className={initialDecision === value ? "selected" : ""} onClick={()=>setInitialDecision(value)}>{value === "Decline" ? "Reject" : value}</button>)}</div><Confidence value={initialConfidence} onChange={setInitialConfidence}/><label className="field reviewer-field"><span>Reviewer / session ID</span><input value={reviewerId} onChange={e=>setReviewerId(e.target.value)} /></label><button className="primary" disabled={loading || !condition}>{loading ? "Loading AI review..." : !condition ? "Assigning condition..." : "Reveal AI decision"} <ArrowUpRight size={18}/></button></div>}
            {stage === 2 && <button type="button" className="secondary" onClick={()=>{setStage(1);setResult(null);setSaved(false);setStageStartedAt(Date.now());}}>Start a new review</button>}
          </form>
        </section>
        <section className="card result-card"><div className="card-heading"><div><span className="step">02</span><div><h2>Decision & explanation</h2><p>{stage === 1 ? "AI output stays hidden until your first decision." : "Review, calibrate, and record your final decision."}</p></div></div><Activity size={21}/></div>
          {stage === 1 && <div className="empty"><div className="empty-orb"><Sparkles size={26}/></div><h3>Human judgment first</h3><p>Choose a decision and confidence, then reveal the model output.</p><div className="empty-line"><span/><span/><span/></div></div>}
          {error && <div className="error">{error}<small>Check that the API is running at <code>{API}</code>.</small></div>}
          {result && <div className="result-content"><div className={"decision " + result.decision.toLowerCase()}><div><span className="decision-label">MODEL OUTCOME</span><h3>{result.decision === "Approved" ? "Approve" : "Reject"} recommended · {Math.round(result.approval_probability * 100)}%</h3><p>{result.risk_band}</p></div><div className="score-ring" style={{"--score": `${result.approval_probability*100}%`}}><strong>{Math.round(result.approval_probability*100)}<small>%</small></strong></div></div>          <div className="summary" style={{display: condition === "prediction_lime" ? undefined : "none"}}><CheckCircle2 size={18}/>{result.summary}</div><div className="counterfactuals lime-only" style={{display: condition === "prediction_lime" ? undefined : "none"}}><div className="section-title"><span>What would change the decision?</span><small>Smallest actionable changes</small></div>{result.counterfactuals.length ? result.counterfactuals.map(item=><div className="counterfactual" key={Object.values(item.changes).join("|")}><div className="counterfactual-change">{Object.entries(item.changes).map(([feature,change])=><span key={feature}><strong>{feature}</strong> {change}</span>)}</div><b>{Math.round(item.approval_probability*100)}% approval</b></div>) : <p className="no-counterfactual">No realistic single or paired change flipped this decision within the search constraints.</p>}</div>                    {condition === "prediction_lime" && <div className="explanation-reliability lime-only"><div className="section-title"><span>Explanation reliability</span><small>LIME quality</small></div><div className="reliability-values"><span>Fidelity <b>{Math.round(result.explanation_reliability.fidelity * 100)}%</b></span><span>Stability <b>{Math.round(result.explanation_reliability.stability * 100)}%</b></span>          <span>Rank correlation <b>{Math.round(result.explanation_reliability.rank_correlation * 100)}%</b></span><span>Top-5 overlap <b>{Math.round(result.explanation_reliability.top5_overlap * 100)}%</b></span></div>{result.explanation_reliability.warning && <p className="reliability-warning">{result.explanation_reliability.warning}          </p>}</div>}{condition === "prediction_lime" && <><div className="section-title"><span>Local drivers</span><small>What shifted the score</small></div>{result.explanation.every(item => item.importance < 0.01) && <p className="reliability-warning">All local contributions are below 0.01; treat this explanation as low-signal.</p>}<div className="drivers">{result.explanation.map(item=><div className="driver" key={item.feature}><div className={"driver-icon " + (item.contribution > 0 ? "positive" : "negative")}>{item.contribution > 0 ? <TrendingUp size={16}/> : <TrendingDown size={16}/>}</div><div className="driver-main"><div>          <strong>{item.feature}</strong><span>{formatLocalValue(item.feature, item.value)}</span></div><div className="bar"><i className={item.contribution > 0 ? "positive-bar" : "negative-bar"} style={{width: `${Math.max(3, item.importance / Math.max(...result.explanation.map(driver => driver.importance), 0.0001) * 100)}%`}}/></div></div><b className={item.contribution > 0 ? "positive-text" : "negative-text"}>{item.contribution > 0 ? "+" : ""}{item.contribution.toFixed(3)}</b></div>)}</div></>}<div className="review-final"><div className="section-title"><span>Final reviewer decision</span><small>After AI output</small></div><div className="decision-toggle">{["Approve","Decline"].map(value=><button type="button" key={value} className={finalDecision === value ? "selected" : ""} onClick={()=>setFinalDecision(value)}>{value === "Decline" ? "Reject" : value}</button>)}</div><Confidence value={finalConfidence} onChange={setFinalConfidence}/>          <button type="button" className="primary" disabled={loading || saved} onClick={saveReview}>{saved ? "Review saved" : loading ? "Saving review..." : "Save review"} <CheckCircle2 size={18}/></button><div className="record-actions"><button type="button" className="secondary" onClick={() => downloadRecord(decisionRecord(result, applicant, finalDecision, finalConfidence, initialDecision, initialConfidence, reviewerId))}>Download JSON record</button><button type="button" className="secondary" onClick={() => printRecord(decisionRecord(result, applicant, finalDecision, finalConfidence, initialDecision, initialConfidence, reviewerId))}>Printable view</button></div>{saved &&           <p className="saved-note">{mode === "study" ? "Review stored for trust-calibration analysis." : "Explore result was not stored as study data."} <a href={`${API}/api/reviews/export`} target="_blank" rel="noreferrer"><Download size={13}/> Export CSV</a></p>}</div><div className="compliance"><div className="section-title"><span><ShieldCheck size={15}/> Responsible-use notes</span>          <small>Review before action</small></div><p>          Reason-code status: <strong>{result.reason_codes_status === "not_applicable" ? "Not applicable" : result.reason_codes_status === "suppressed_low_fidelity" ? "Suppressed: low fidelity" : "OK"}</strong></p>{result.compliance_notes.map(note=><p key={note}><ChevronRight size={14}/>{note}</p>)}</div></div>}
        </section>
      </div>
      <footer><span><Sparkles size={14}/> CLEARCREDIT LAB</span><span>TRUST CALIBRATION · LIME SURROGATE</span><span>DEMO ENVIRONMENT · NOT A CREDIT DECISION</span></footer>
    </main>
  </div>
}
function Confidence({value,onChange}) { return <label className="field confidence"><span>Confidence (1 = low, 5 = high)</span><input type="range" min="1" max="5" step="1" value={value} onChange={e=>onChange(e.target.value)}/><b>{value}/5</b></label> }
function Field({label,prefix,suffix,value,onChange,disabled}) { return <label className="field"><span>{label}</span><div className="input-wrap">{prefix&&<b>{prefix}</b>}<input type="number" disabled={disabled} value={value} onChange={e=>onChange(e.target.value)}/>{suffix&&<b>{suffix}</b>}</div></label> }
function Select({label,value,options,onChange,disabled}) { return <label className="field"><span>{label}</span><select disabled={disabled} value={value} onChange={e=>onChange(e.target.value)}>{options.map(o=><option key={o}>{o}</option>)}</select></label> }
function decisionRecord(result, form, finalDecision, finalConfidence, initialDecision, initialConfidence, reviewerId) {
  return {
    ...result,
    applicant: form,
    reviewer: { reviewer_id: reviewerId, initial_decision: initialDecision, initial_confidence: Number(initialConfidence), final_decision: finalDecision, final_confidence: Number(finalConfidence) },
  };
}
function downloadRecord(record) {
  const blob = new Blob([JSON.stringify(record, null, 2)], {type: "application/json"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `clearcredit-decision-${record.decision_id}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}
function printRecord(record) {
  const popup = window.open("", "_blank", "noopener,noreferrer");
  if (!popup) return;
  const printableRecord = {
    ...record,
    decision: record.decision === "Approved" ? "Approve" : "Reject",
    reviewer: {
      ...record.reviewer,
      initial_decision: record.reviewer.initial_decision === "Decline" ? "Reject" : record.reviewer.initial_decision,
      final_decision: record.reviewer.final_decision === "Decline" ? "Reject" : record.reviewer.final_decision,
    },
  };
  popup.document.write(`<html><head><title>ClearCredit Decision Record</title><style>body{font-family:Arial;max-width:900px;margin:40px auto;color:#172521}pre{white-space:pre-wrap;background:#f4f6f2;padding:20px;border-radius:8px}</style></head><body><h1>ClearCredit Decision Record</h1><p>Demo only — not a real credit decisioning system.</p><pre>${JSON.stringify(printableRecord, null, 2)}</pre></body></html>`);
  popup.document.close();
  popup.print();
}
function FairnessPage({onBack}) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    fetch(`${API}/api/fairness`).then(response => { if (!response.ok) throw new Error("Unable to load fairness results."); return response.json(); }).then(setReport).catch(requestError => setError(requestError.message));
  }, []);
  return <div className="app"><nav><div className="brand"><span className="brand-mark"><Sparkles size={17}/></span> clear<span>credit</span></div><div className="nav-meta"><span className="live-dot"/> Fairness audit <button className="nav-link" onClick={onBack}>New review</button></div></nav><main><header className="hero dashboard-hero"><div><div className="eyebrow">REGULATORY SUPPORT / FAIRNESS AUDIT</div><h1>Check for<br/><em>unequal outcomes.</em></h1><p>Audit-only synthetic demographic groups are evaluated separately from the model features. This is a screening tool, not proof of fairness.</p></div></header>{error && <div className="card error">{error}</div>}{report && <><div className="metric-cards"><MetricCard label="Demographic parity difference" value={report.demographic_parity_difference}/><MetricCard label="Equal opportunity difference" value={report.equal_opportunity_difference}/><MetricCard label="Group column used by model" value={report.model_features_exclude_group ? "No" : "Yes"}/></div><section className="card chart-card"><div className="card-heading"><div><span className="step">05</span><div><h2>Approval rate by group</h2><p>Synthetic groups only; demographic data is excluded from prediction.</p></div></div><ShieldCheck size={21}/></div><div className="fairness-table">{Object.entries(report.approval_rate_by_group).map(([group,data])=><div className="fairness-row" key={group}><strong>{group}</strong><span>{data.count} cases</span><b>{Math.round(data.approval_rate * 100)}% approval</b>  <small>{Math.round(data.true_positive_rate * 100)}% TPR · {Math.round(data.false_positive_rate * 100)}% FPR</small></div>)}</div>{report.flags.map(flag=><p className="fairness-flag" key={flag}>{flag}</p>)}</section><section className="card chart-card"><div className="section-title"><span>Possible proxy features</span><small>Correlation threshold 0.20</small></div>{report.possible_proxy_features.length ? report.possible_proxy_features.map(item=><p className="proxy-row" key={item.feature}><strong>{item.feature}</strong><span>{item.correlation}</span></p>) : <p className="no-counterfactual">No numeric proxy feature crossed the screening threshold.</p>}<p className="chart-note">Synthetic demographic columns are not used by the model. Correlation flags are screening signals requiring expert review.</p></section></>}</main></div>;
}
function Dashboard({onBack}) {
  const [metrics, setMetrics] = useState(null);
  const [calibration, setCalibration] = useState(null);
  const [includeSimulated, setIncludeSimulated] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([fetch(`${API}/api/metrics/trust?include_simulated=${includeSimulated}`), fetch(`${API}/api/model/calibration`)]).then(async ([metricsResponse, calibrationResponse]) => {
      if (!metricsResponse.ok || !calibrationResponse.ok) throw new Error("Unable to load dashboard metrics.");
      return Promise.all([metricsResponse.json(), calibrationResponse.json()]);
    }).then(([trust, modelCalibration]) => { setMetrics(trust); setCalibration(modelCalibration); }).catch(requestError => setError(requestError.message));
  }, [includeSimulated]);
  const chartData = metrics ? ["over_reliance","under_reliance","appropriate_reliance_rate","human_accuracy_before","human_accuracy_after","ai_alone_accuracy","human_ai_team_accuracy"].map(key => ({ metric: key.replaceAll("_"," "), prediction_only: percent(metrics.conditions.prediction_only[key]), prediction_lime: percent(metrics.conditions.prediction_lime[key]), prediction_only_ci: errorBar(metrics.conditions.prediction_only.confidence_intervals_95?.[key], metrics.conditions.prediction_only[key]), prediction_lime_ci: errorBar(metrics.conditions.prediction_lime.confidence_intervals_95?.[key], metrics.conditions.prediction_lime[key]) })) : [];
  const reliabilityData = calibration?.bins.filter(bin => bin.count > 0).map(bin => ({ predicted: percent(bin.mean_predicted_probability), observed: percent(bin.observed_frequency), perfect: percent(bin.mean_predicted_probability) })) || [];
  return <div className="app"><nav><div className="brand"><span className="brand-mark"><Sparkles size={17}/></span> clear<span>credit</span></div><div className="nav-meta"><span className="live-dot"/> Trust calibration <button className="nav-link" onClick={onBack}>New review</button></div></nav><main><header className="hero dashboard-hero"><div><div className="eyebrow">TRUST CALIBRATION / STUDY DASHBOARD</div><h1>Measure the<br/><em>human-AI relationship.</em></h1><p>Compare reviewer behavior with and without local LIME explanations. Metrics use only stored synthetic or explicitly submitted review records.</p></div></header>{error && <div className="card error">{error}</div>}{metrics && calibration && <><div className="metric-cards"><MetricCard label="Stored reviews" value={metrics.total_reviews}/><MetricCard label="Simulated reviews" value={metrics.simulated_reviews}/><MetricCard label="Prediction-only n" value={metrics.conditions.prediction_only.n}/><MetricCard label="Prediction + LIME n" value={metrics.conditions.prediction_lime.n}/></div><section className="card chart-card"><label className="study-toggle"><input type="checkbox" checked={includeSimulated} onChange={event => setIncludeSimulated(event.target.checked)}/> Include simulated reviewers in dashboard metrics</label><div className="card-heading"><div><span className="step">03</span><div><h2>Trust calibration metrics</h2><p>Percentages are calculated separately by study condition.</p></div></div><Activity size={21}/></div><div className="chart-wrap"><ResponsiveContainer width="100%" height={360}><BarChart data={chartData} margin={{top:10,right:20,left:0,bottom:55}}><CartesianGrid strokeDasharray="3 3" stroke="#edf0ed"/><XAxis dataKey="metric" angle={-25} textAnchor="end" interval={0} tick={{fontSize:11,fill:"#71817a"}}/><YAxis unit="%" domain={[0,100]} tick={{fontSize:11,fill:"#71817a"}}/><Tooltip formatter={value => `${value}%`}/>  <Bar dataKey="prediction_only" name="Prediction only" fill="#8ab7a2" radius={[4,4,0,0]}><ErrorBar dataKey="prediction_only_ci" width={4} strokeWidth={2}/></Bar><Bar dataKey="prediction_lime" name="Prediction + LIME" fill="#2d8068" radius={[4,4,0,0]}><ErrorBar dataKey="prediction_lime_ci" width={4} strokeWidth={2}/></Bar></BarChart></ResponsiveContainer></div></section><section className="card chart-card reliability-card"><div className="card-heading"><div><span className="step">04</span><div><h2>Model reliability diagram</h2><p>Held-out predicted probability versus observed approval frequency.</p></div></div><Activity size={21}/></div><div className="calibration-summary"><MetricCard label="Brier score" value={calibration.brier_score}/><MetricCard label="Expected calibration error" value={calibration.expected_calibration_error}/><MetricCard label="Held-out cases" value={calibration.held_out_count}/></div><div className="chart-wrap"><ResponsiveContainer width="100%" height={360}><LineChart data={reliabilityData} margin={{top:10,right:20,left:0,bottom:25}}><CartesianGrid strokeDasharray="3 3" stroke="#edf0ed"/><XAxis dataKey="predicted" unit="%" type="number" domain={[0,100]} tick={{fontSize:11,fill:"#71817a"}}/><YAxis unit="%" domain={[0,100]} tick={{fontSize:11,fill:"#71817a"}}/><Tooltip formatter={value => `${value}%`}/><Line type="monotone" dataKey="perfect" name="Perfect calibration" stroke="#b8c7bf" strokeDasharray="5 5" dot={false}/><Line type="monotone" dataKey="observed" name="Observed frequency" stroke="#2d8068" strokeWidth={3} dot={{r:4}}/></LineChart></ResponsiveContainer></div><p className="chart-note">Lower Brier score and ECE indicate better probability calibration. Empty bins are omitted from the diagram.</p></section><div className="metric-detail-grid"><MetricDetail title="Prediction only" data={metrics.conditions.prediction_only}/><MetricDetail title="Prediction + LIME" data={metrics.conditions.prediction_lime}/></div></>}</main></div>
}
function percent(value) { return value == null ? 0 : Math.round(value * 100); }
function errorBar(interval, value) { return interval ? [Math.max(0, percent(value) - percent(interval[0])), Math.max(0, percent(interval[1]) - percent(value))] : [0, 0]; }
function MetricCard({label,value}) { return <div className="card metric-card"><span>{label}</span><strong>{value}</strong></div> }
function MetricDetail({title,data}) { return <section className="card metric-detail"><h3>{title}</h3><p>n = {data.n}. {data.warning || "Estimates have at least 20 study cases."}</p><dl><dt>Over-reliance</dt><dd>{percent(data.over_reliance)}% ({data.over_reliance_numerator}/{data.over_reliance_denominator})</dd><dt>Under-reliance</dt><dd>{percent(data.under_reliance)}% ({data.under_reliance_numerator}/{data.under_reliance_denominator})</dd><dt>Human accuracy before / after</dt><dd>{percent(data.human_accuracy_before)}% / {percent(data.human_accuracy_after)}%</dd><dt>AI-alone / team accuracy</dt><dd>{percent(data.ai_alone_accuracy)}% / {percent(data.human_ai_team_accuracy)}%</dd><dt>Confidence ECE before / after</dt><dd>{percent(data.human_confidence_ece_before)}% / {percent(data.human_confidence_ece_after)}%</dd></dl><p className="chart-note">Appropriate reliance is {percent(data.appropriate_reliance_rate)}%. Higher means reviewers followed correct AI advice and rejected incorrect AI advice more often.</p></section> }
createRoot(document.getElementById("root")).render(<App />);
