import { useState, useEffect, useRef } from 'react';
import { Bar, Scatter } from 'react-chartjs-2';
import {
Chart as ChartJS, CategoryScale, LinearScale, BarElement,
PointElement, Title, Tooltip, Legend
} from 'chart.js';
ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, Title, Tooltip, Legend);

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Dashboard() {
const [violations, setViolations] = useState([]);
const [loading, setLoading] = useState(false);
const [apiStatus, setApiStatus] = useState('unknown');
const [uploadMsg, setUploadMsg] = useState('');
const [uploading, setUploading] = useState(false);
const fileRef = useRef();

// Wake up Render (cold start) and fetch violations on load
useEffect(() => { pingApi(); fetchViolations(); }, []);

async function pingApi() {
try {
const r = await fetch(`${API}/health`);
setApiStatus(r.ok ? 'online' : 'error');
} catch { setApiStatus('offline'); }
}

async function fetchViolations() {
setLoading(true);
try {
const r = await fetch(`${API}/violations?limit=500`);
const data = await r.json();
setViolations(data);
} catch (e) { console.error(e); }
setLoading(false);
}

async function uploadVideo(e) {
const file = e.target.files[0];
if (!file) return;
setUploading(true);
setUploadMsg('Uploading and analyzing... this may take a minute.');
const form = new FormData();
form.append('file', file);
try {
const r = await fetch(`${API}/analyze-video`, { method: 'POST', body: form });
const data = await r.json();
setUploadMsg(`Done. ${data.frames_processed} frames processed, ${data.violations_logged} violations logged.`);
fetchViolations();
} catch (e) {
setUploadMsg('Error: ' + e.message);
}
setUploading(false);
}

// Chart data — violations by type
const typeCounts = violations.reduce((acc, v) => {
acc[v.violation_type] = (acc[v.violation_type] || 0) + 1; return acc;
}, {});

// Configured chart scale/legend labels to black text
const barData = {
labels: Object.keys(typeCounts),
datasets: [{ 
    label: 'Count', 
    data: Object.values(typeCounts),
    backgroundColor: ['#E94560', '#0F7173', '#F5A623'] 
}]
};

const scatterData = {
datasets: [{
    label: 'Violation locations',
    data: violations.map(v => ({ x: v.bbox_x, y: v.bbox_y })),
    backgroundColor: 'rgba(233,69,96,0.5)',
    pointRadius: 6,
}]
};

// Maintained color-coded state indicators, changed unknown text label from grey to black
const statusColor = { online:'#0F7173', offline:'#E94560', error:'#F5A623', unknown:'#000000' };

const chartOptions = {
responsive: true,
plugins: {
    legend: {
        display: false,
        labels: { color: '#000000' }
    }
},
scales: {
    x: { ticks: { color: '#000000' } },
    y: { ticks: { color: '#000000' } }
}
};

const scatterOptions = {
responsive: true,
plugins: {
    legend: { labels: { color: '#000000' } }
},
scales: { 
    x: {
        title: { display: true, text: 'X position', color: '#000000' },
        ticks: { color: '#000000' }
    },
    y: {
        title: { display: true, text: 'Y position', color: '#000000' },
        ticks: { color: '#000000' },
        reverse: true
    } 
}
};

return (
<div style={{ fontFamily: 'system-ui', maxWidth: 1100, margin: '0 auto', padding: '2rem', color: '#000000' }}>
<h1 style={{ color: '#0F3460', borderBottom: '3px solid #0F7173', paddingBottom: '0.5rem' }}>
Workplace Safety Monitor
</h1>

{/* API Status bar */}
<div style={{ display:'flex', alignItems:'center', gap:'1rem', marginBottom:'1.5rem' }}>
<span style={{ color: statusColor[apiStatus], fontWeight: 'bold' }}>
● API: {apiStatus}
</span>
<button onClick={pingApi}
style={{ padding:'0.4rem 1rem', background:'#0F7173', color:'#000000', border:'none',
borderRadius:6, cursor:'pointer', fontWeight: 'bold' }}>
Wake up API
</button>
<small style={{ color:'#000000' }}>
(Render free tier sleeps after 15 min — click to wake, then wait ~30s)
</small>
</div>

{/* Upload section */}
<div style={{ background:'#F7F9FC', border:'1px solid #CBD5E0', borderRadius:8,
padding:'1.5rem', marginBottom:'2rem' }}>
<h2 style={{ marginTop:0, color:'#0F3460' }}>📹 Upload Video for Analysis</h2>
<input type='file' accept='video/*' ref={fileRef} onChange={uploadVideo}
disabled={uploading} style={{ marginBottom:'1rem', display:'block', color: '#000000' }} />
{uploading && <div style={{ color:'#0F7173', fontWeight: 'bold' }}>⏳ {uploadMsg}</div>}
{!uploading && uploadMsg && <div style={{ color:'#0F3460', fontWeight: 'bold' }}>✅ {uploadMsg}</div>}
</div>

{/* Stats row */}
<div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:'1rem', marginBottom:'2rem' }}>
{[
['Total Violations', violations.length],
['Unique Types', new Set(violations.map(v=>v.violation_type)).size],
['Avg Duration (s)', violations.length > 0
? (violations.reduce((a,v)=>a+v.duration_seconds,0)/violations.length).toFixed(1) : '—']
].map(([label, val]) => (
<div key={label} style={{ background:'#fff', border:'1px solid #CBD5E0',
borderRadius:8, padding:'1rem', textAlign:'center' }}>
<div style={{ fontSize:'2rem', fontWeight:'bold', color:'#0F3460' }}>{val}</div>
<div style={{ color:'#000000', fontSize:'0.9rem', fontWeight: '500' }}>{label}</div>
</div>
))}
</div>

{/* Charts */}
{violations.length > 0 && (
<div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'2rem', marginBottom:'2rem' }}>
<div style={{ background:'#fff', border:'1px solid #CBD5E0', borderRadius:8, padding:'1rem' }}>
<h3 style={{ marginTop:0, color: '#000000' }}>Violations by Type</h3>
<Bar data={barData} options={chartOptions} />
</div>
<div style={{ background:'#fff', border:'1px solid #CBD5E0', borderRadius:8, padding:'1rem' }}>
<h3 style={{ marginTop:0, color: '#000000' }}>Violation Location Map</h3>
<Scatter data={scatterData} options={scatterOptions} />
</div>
</div>
)}

{/* Table */}
<div style={{ background:'#fff', border:'1px solid #CBD5E0', borderRadius:8, padding:'1rem' }}>
<div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
<h2 style={{ marginTop:0, color:'#0F3460' }}>📊 Violation Log</h2>
<button onClick={fetchViolations}
style={{ padding:'0.4rem 1rem', background:'#0F7173', color:'#000000',
border:'none', borderRadius:6, cursor:'pointer', fontWeight: 'bold' }}>
Refresh
</button>
</div>
{loading && <p style={{ color: '#000000' }}>Loading...</p>}
{!loading && violations.length === 0 && (
<p style={{ color:'#000000' }}>No violations yet. Upload a video above.</p>
)}
{violations.length > 0 && (
<table style={{ width:'100%', borderCollapse:'collapse', fontSize:'0.9rem', color: '#000000' }}>
<thead>
<tr style={{ background:'#F7F9FC' }}>
{['Timestamp','Type','Duration (s)'].map(h => (
<th key={h} style={{ padding:'0.5rem', textAlign:'left',
borderBottom:'2px solid #CBD5E0', color: '#000000' }}>{h}</th>
))}
</tr>
</thead>
<tbody>
{violations.map(v => (
<tr key={v.id} style={{ borderBottom:'1px solid #E2E8F0' }}>
<td style={{ padding:'0.5rem', color: '#000000' }}>{new Date(v.timestamp).toLocaleString()}</td>
<td style={{ padding:'0.5rem', color:'#E94560', fontWeight:'bold' }}>
{v.violation_type}
</td>
<td style={{ padding:'0.5rem', color: '#000000' }}>{v.duration_seconds.toFixed(1)}</td>
</tr>
))}
</tbody>
</table>
)}
</div>
</div>
);
}