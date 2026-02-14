
import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import html2canvas from 'html2canvas';
import { Tree, TreeNode } from 'react-organizational-chart';
import './sourcing_verify.css'; // switched to Sourcing_Verify theme
// Admin feature removed (AdminUploadButton not imported)

/* ========================= CONSTANTS ========================= */
// SSE Configuration
const SSE_RECONNECT_BASE_DELAY_MS = 1000;
const SSE_RECONNECT_MAX_DELAY_MS = 30000;
const SSE_MAX_RECONNECT_ATTEMPTS = 5;
const API_PORT = 4000;

/* ========================= HELPERS ========================= */
function isHumanName(name) {
  if (!name || typeof name !== 'string') return false;
  const nonHumanPatterns = /(http|www\.|Font|License|Version|Copyright|Authors|Open Font|Project|games|game)/i;
  if (name.length < 2 || name.length > 60) return false;
  const nonAlpha = name.replace(/[a-zA-Z\s\-']/g, '');
  if (nonAlpha.length > 8) return false;
  return !nonHumanPatterns.test(name);
}
function normalizeTier(s) {
  if (!s) return '';
  const v = String(s).trim().toLowerCase().replace(/\./g, '').replace(/\s+/g, ' ');
  if (v === 'jr' || v === 'junior') return 'Junior';
  if (v === 'mid' || v === 'middle' || v === 'intermediate') return 'Mid';
  if (v === 'sr' || v === 'senior') return 'Senior';
  if (v.includes('lead')) return 'Lead';
  if (v === 'mgr' || v === 'manager' || v.includes(' manager')) return 'Manager';
  if (v === 'expert' || v === 'principal' || v === 'staff' || v.includes('principal') || v.includes('staff')) return 'Expert';
  if (v === 'sr manager' || v === 'senior manager' || v === 'senior mgr' || v === 'sr mgr' || v.includes('sr manager')) return 'Sr Manager';
  if (v === 'sr director' || v === 'senior director' || v === 'sr dir' || v === 'svp' || v.includes('sr director')) return 'Sr Director';
  if (v === 'director' || v === 'dir' || v.includes(' director')) return 'Director';
  if (v === 'executive' || v === 'exec' || v === 'cxo' || v === 'vp' || v === 'chief' || v.includes('executive') || v.includes('vice president') || v.includes('chief')) return 'Executive';
  if (/\bexecutive|chief|vp|vice president|cxo\b/.test(v)) return 'Executive';
  if (/\bsenior director\b/.test(v)) return 'Sr Director';
  if (/\bdirector\b/.test(v)) return 'Director';
  if (/\bsenior manager\b|\bsr manager\b|\bsr mgr\b/.test(v)) return 'Sr Manager';
  if (/\bmanager\b|\bmgr\b/.test(v)) return 'Manager';
  if (/\blead\b/.test(v)) return 'Lead';
  if (/\bsenior\b|\bsr\b/.test(v)) return 'Senior';
  if (/\bmid(dle)?\b|\bintermediate\b/.test(v)) return 'Mid';
  if (/\bjunior\b|\bjr\b/.test(v)) return 'Junior';
  const cap = v.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  return cap;
}
function inferSeniority(candidate) {
  return normalizeTier(candidate?.seniority) ||
    normalizeTier(candidate?.role_tag) ||
    '';
}
async function fetchSkillsetMapping() {
  try {
    const res = await fetch('http://localhost:4000/skillset-mapping');
    if (!res.ok) return {};
    return await res.json();
  } catch {
    return {};
  }
}

/* ========================= LOGIN COMPONENT ========================= */
function LoginScreen({ onLoginSuccess }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      const res = await fetch('http://localhost:4000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
        credentials: 'include' // Important for cookies
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Login failed');
      
      onLoginSuccess(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh',
      background: '#f1f5f9'
    }}>
      <div className="app-card" style={{ padding: 32, width: 360 }}>
        <h2 style={{ marginTop: 0, marginBottom: 24, textAlign: 'center', color: 'var(--azure-dragon)' }}>Login</h2>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: 'block', marginBottom: 6, fontWeight: 700, fontSize: 13 }}>Username</label>
            <input
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              style={{ width: '100%', padding: '8px 12px', boxSizing: 'border-box' }}
              required
            />
          </div>
          <div style={{ marginBottom: 24 }}>
            <label style={{ display: 'block', marginBottom: 6, fontWeight: 700, fontSize: 13 }}>Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              style={{ width: '100%', padding: '8px 12px', boxSizing: 'border-box' }}
              required
            />
          </div>
          {error && <div style={{ color: 'var(--danger)', marginBottom: 16, fontSize: 13, textAlign: 'center', fontWeight: 'bold' }}>{error}</div>}
          <button
            type="submit"
            disabled={loading}
            className="btn-primary"
            style={{ width: '100%', padding: '10px' }}
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  );
}

/* ========================= EMAIL VERIFICATION MODAL ========================= */
function EmailVerificationModal({ data, onClose, email }) {
  if (!data) return null;
  const { 
    status, sub_status, free_email, account, smtp_provider, 
    first_name, last_name, domain, mx_found, mx_record, domain_age_days, did_you_mean 
  } = data;

  // Render a single grid item with label and value
  const Field = ({ label, value }) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ fontSize: 10, color: 'var(--argent)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{label}</div>
      <div style={{ 
        padding: '8px 12px', 
        borderRadius: 6, 
        border: '1px solid var(--neutral-border)',
        background: '#fff',
        fontSize: 14,
        color: 'var(--muted)',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        fontFamily: 'Orbitron, sans-serif'
      }} title={String(value ?? '')}>
        {String(value ?? 'Unknown')}
      </div>
    </div>
  );

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 9999
    }} onClick={onClose}>
      <div className="app-card" style={{ padding: 24, width: 600, maxWidth: '90vw', position: 'relative' }} onClick={e => e.stopPropagation()}>
        <button onClick={onClose} style={{
          position: 'absolute', top: 12, right: 12, border: 'none', background: 'transparent', fontSize: 20, cursor: 'pointer', color: 'var(--argent)'
        }}>×</button>

        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16 }}>
          <div style={{
            width: 48, height: 48, borderRadius: '50%', background: 'var(--accent)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: 24,
            boxShadow: '0 0 15px rgba(7,54,121,0.4)'
          }}>✉</div>
        </div>

        <div style={{ 
          background: '#f3e8ff', padding: '12px', borderRadius: 8, textAlign: 'center', 
          color: '#581c87', fontWeight: 600, fontSize: 16, marginBottom: 24, border: '1px solid #d8b4fe'
        }}>
          {email}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
          {/* Row 1 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ fontSize: 10, color: 'var(--argent)', textTransform: 'uppercase' }}>STATUS</div>
            <div style={{ 
              padding: '8px 12px', borderRadius: 6, border: '1px solid var(--neutral-border)', 
              background: status === 'Capture All' ? '#dcfce7' : '#fff0f0', color: status === 'Capture All' ? '#166534' : '#b91c1c', fontSize: 14, fontWeight: 'bold' 
            }}>{status}</div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ fontSize: 10, color: 'var(--argent)', textTransform: 'uppercase' }}>SUB-STATUS</div>
            <div style={{ 
              padding: '8px 12px', borderRadius: 6, border: '1px solid var(--neutral-border)', 
              background: '#f1f5f9', color: 'var(--muted)', fontSize: 14 
            }}>{sub_status}</div>
          </div>
          <Field label="FREE EMAIL" value={free_email} />

          {/* Row 2 */}
          <Field label="DID YOU MEAN" value={did_you_mean} />
          <Field label="ACCOUNT" value={account} />
          <Field label="DOMAIN" value={domain} />

          {/* Row 3 */}
          <Field label="DOMAIN AGE DAYS" value={domain_age_days} />
          <Field label="SMTP PROVIDER" value={smtp_provider} />
          <Field label="MX FOUND" value={mx_found} />

           {/* Row 4 */}
           <div style={{ gridColumn: '1 / span 3' }}>
             <Field label="MX RECORD" value={mx_record} /> 
           </div>
           
           {/* Row 5 */}
           <Field label="FIRST NAME" value={first_name} />
           <Field label="LAST NAME" value={last_name} />
        </div>
      </div>
    </div>
  );
}

/* ========================= EMAIL COMPOSE MODAL ========================= */
function EmailComposeModal({ isOpen, onClose, toAddresses, candidateName, smtpConfig }) {
  const [from, setFrom] = useState('');
  const [cc, setCc] = useState('');
  const [bcc, setBcc] = useState('');
  const [subject, setSubject] = useState('');
  const [body, setBody] = useState('');
  const [files, setFiles] = useState([]);
  const [sending, setSending] = useState(false);
  const [directSending, setDirectSending] = useState(false); // State for Direct Send

  // Calendar / Google Meet state
  const [addMeet, setAddMeet] = useState(false);
  const [calendarSlots, setCalendarSlots] = useState([]);
  const [slotsLoading, setSlotsLoading] = useState(false);
  const [selectedSlotIndex, setSelectedSlotIndex] = useState(null);
  const [creatingEvent, setCreatingEvent] = useState(false);
  const [meetLink, setMeetLink] = useState('');
  const [icsString, setIcsString] = useState('');
  const [calendarError, setCalendarError] = useState('');

  // Template & AI State
  const [templates, setTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [showAiInput, setShowAiInput] = useState(false);
  const [aiPrompt, setAiPrompt] = useState('');
  const [aiLoading, setAiLoading] = useState(false);

  const [to, setTo] = useState(toAddresses);
  
  // Sync prop to state when prop changes (e.g. opening modal with new selection)
  useEffect(() => {
    setTo(toAddresses);
  }, [toAddresses]);

  // Load Templates on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('emailTemplates');
      if (saved) setTemplates(JSON.parse(saved));
    } catch (e) {
      console.error('Failed to load templates', e);
    }
  }, []);

  // Reset calendar-related temporary state when modal opens/closes
  useEffect(() => {
    if (!isOpen) {
      setAddMeet(false);
      setCalendarSlots([]);
      setSelectedSlotIndex(null);
      setMeetLink('');
      setIcsString('');
      setCalendarError('');
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleFileChange = (e) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  // Template Management
  const handleSaveTemplate = () => {
    if (!subject && !body) {
      alert("Cannot save an empty template.");
      return;
    }
    const name = prompt("Enter a name for this template:");
    if (!name) return;
    
    // Check overwrite
    const existingIndex = templates.findIndex(t => t.name === name);
    let newTemplates;
    
    if (existingIndex >= 0) {
      if (!window.confirm(`Template "${name}" already exists. Do you want to overwrite it?`)) return;
      newTemplates = [...templates];
      newTemplates[existingIndex] = { ...newTemplates[existingIndex], subject, body };
    } else {
      newTemplates = [...templates, { id: Date.now(), name, subject, body }];
    }
    
    setTemplates(newTemplates);
    localStorage.setItem('emailTemplates', JSON.stringify(newTemplates));
    setSelectedTemplate(name);
  };

  // NEW: Delete Template Function
  const handleDeleteTemplate = () => {
    if (!selectedTemplate) return;
    if (!window.confirm(`Are you sure you want to delete template "${selectedTemplate}"?`)) return;

    const newTemplates = templates.filter(t => t.name !== selectedTemplate);
    setTemplates(newTemplates);
    localStorage.setItem('emailTemplates', JSON.stringify(newTemplates));
    setSelectedTemplate('');
    setSubject('');
    setBody('');
  };

  const handleLoadTemplate = (e) => {
    const tmplName = e.target.value;
    setSelectedTemplate(tmplName);
    if (!tmplName) return;
    
    const t = templates.find(x => x.name === tmplName);
    if (t) {
      setSubject(t.subject);
      setBody(t.body);
    }
  };

  // AI Drafting
  const handleAiDraft = async () => {
    if (!aiPrompt.trim()) return;
    setAiLoading(true);
    try {
      // Pass 'from' context as well
      const res = await fetch('http://localhost:4000/draft-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          prompt: aiPrompt, 
          context: { candidateName: candidateName || 'Candidate', myEmail: from } 
        }),
        credentials: 'include'
      });
      if (!res.ok) throw new Error('Failed to draft email');
      const data = await res.json();
      
      if (data.subject) setSubject(data.subject);
      if (data.body) setBody(data.body);
      
      setShowAiInput(false);
      setAiPrompt('');
    } catch (e) {
      alert("Error drafting with AI: " + e.message);
    } finally {
      setAiLoading(false);
    }
  };

  // Calendar helpers
  const handleConnectCalendar = () => {
    // Open OAuth connect in popup
    const url = 'http://localhost:4000/auth/google/calendar/connect';
    const w = 600, h = 700;
    const left = (window.screen.width / 2) - (w / 2);
    const top = (window.screen.height / 2) - (h / 2);
    window.open(url, 'connect_google_calendar', `width=${w},height=${h},top=${top},left=${left}`);
  };

  const handleFindSlots = async () => {
    setSlotsLoading(true);
    setCalendarSlots([]);
    setSelectedSlotIndex(null);
    setCalendarError('');
    try {
      // default: next 3 days from now
      const now = new Date();
      const startISO = new Date(now.getTime()).toISOString();
      const endISO = new Date(now.getTime() + 3 * 24 * 60 * 60 * 1000).toISOString();
      const res = await fetch('http://localhost:4000/calendar/freebusy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ startISO, endISO, durationMinutes: 30 }),
        credentials: 'include'
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Failed to query freebusy. Connect your calendar first.');
      }
      const data = await res.json();
      if (!data.slots || !Array.isArray(data.slots)) {
        throw new Error('No free slots returned.');
      }
      setCalendarSlots(data.slots);
      setSelectedSlotIndex(data.slots.length ? 0 : null);
    } catch (e) {
      console.error('find slots error', e);
      setCalendarError(e.message || 'Failed to find slots');
    } finally {
      setSlotsLoading(false);
    }
  };

  const handleCreateEvent = async () => {
    if (selectedSlotIndex == null || !calendarSlots[selectedSlotIndex]) {
      alert('Please select a time slot first.');
      return;
    }
    setCreatingEvent(true);
    setCalendarError('');
    try {
      const slot = calendarSlots[selectedSlotIndex];
      const attendees = (to || '').split(/[;,]+/).map(s => s.trim()).filter(Boolean);
      const payload = {
        summary: subject || `Meeting with ${candidateName || 'Candidate'}`,
        description: body || '',
        startISO: slot.start,
        endISO: slot.end,
        attendees,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
        sendUpdates: 'none'
      };
      const res = await fetch('http://localhost:4000/calendar/create-event', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        credentials: 'include'
      });
      if (!res.ok) {
        const err = await res.json().catch(()=>({}));
        throw new Error(err.error || 'Failed to create event. Check calendar connection and permissions.');
      }
      const data = await res.json();
      if (data.meetLink) {
        setMeetLink(data.meetLink);
        // Append meet link to body if not already present
        if (!body.includes(data.meetLink)) {
          setBody(prev => prev + '\n\nJoin meeting: ' + data.meetLink);
        }
      }
      if (data.ics) setIcsString(data.ics);
      alert('Event created. Meet link added to the message (and ICS attached on send).');
    } catch (e) {
      console.error('create event error', e);
      setCalendarError(e.message || 'Failed to create event');
      alert('Failed to create event: ' + (e.message || 'unknown'));
    } finally {
      setCreatingEvent(false);
    }
  };

  // Option 1: Open in Client (mailto)
  const handleOpenClient = async (e) => {
    e.preventDefault();
    setSending(true);
    
    // Build mailto link
    const params = new URLSearchParams();
    if (cc) params.append('cc', cc);
    if (bcc) params.append('bcc', bcc);
    if (subject) params.append('subject', subject);
    
    let finalBody = body;
    // Replace placeholders if any
    if (candidateName && finalBody.includes('[name]')) {
      finalBody = finalBody.replace(/\[name\]/g, candidateName);
    }
    if (finalBody) params.append('body', finalBody);

    const queryString = params.toString().replace(/\+/g, '%20');
    const mailtoLink = `mailto:${to}?${queryString}`;

    // Small delay for UI feedback
    await new Promise(r => setTimeout(r, 500));
    window.location.href = mailtoLink;
    
    setSending(false);
    onClose();
  };

  // Option 2: Direct Send via Backend (updated to include ICS if present)
  const handleDirectSend = async () => {
    if (!to || !subject || !body) {
        alert("Please fill in To, Subject, and Message.");
        return;
    }
    setDirectSending(true);
    try {
        let finalBody = body;
        if (candidateName && finalBody.includes('[name]')) {
            finalBody = finalBody.replace(/\[name\]/g, candidateName);
        }

        // If a meet link exists but not in body, append (defensive)
        if (meetLink && !finalBody.includes(meetLink)) {
          finalBody += '\n\nJoin meeting: ' + meetLink;
        }

        const payload = {
            to, cc, bcc, subject, 
            body: finalBody,
            from, // Pass the user-entered FROM address
            smtpConfig, // Pass the SMTP config to the backend
        };
        if (icsString) payload.ics = icsString;

        const res = await fetch('http://localhost:4000/send-email', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            credentials: 'include'
        });

        const data = await res.json();
        if(!res.ok) throw new Error(data.error || 'Failed to send');

        alert('Email sent successfully!');
        onClose();
    } catch (e) {
        alert('Error sending email: ' + e.message);
    } finally {
        setDirectSending(false);
    }
  };

  const labelStyle = { display: 'block', marginBottom: 6, fontWeight: 700, fontSize: 13, color: 'var(--muted)' };
  const inputStyle = { width: '100%', padding: '8px', boxSizing: 'border-box' };

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 10000
    }}>
      <div className="app-card" style={{
        width: 700, maxWidth: '95vw',
        display: 'flex', flexDirection: 'column', maxHeight: '90vh'
      }} onClick={e => e.stopPropagation()}>
        
        {/* Header */}
        <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--neutral-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0, fontSize: 18, color: 'var(--azure-dragon)', fontWeight: 700 }}>New Message</h3>
          <button onClick={onClose} style={{ background: 'none', border: 'none', fontSize: 24, color: 'var(--argent)', cursor: 'pointer' }} title="Close">×</button>
        </div>

        {/* Body */}
        <div style={{ padding: 24, overflowY: 'auto' }}>
          <form id="email-form">
            
            {/* FROM Field - User can edit this */}
            <div style={{ marginBottom: 16 }}>
              <label style={labelStyle}>From</label>
              <input 
                type="email" 
                value={from} 
                onChange={e => setFrom(e.target.value)}
                style={inputStyle}
                placeholder="your.email@example.com (Optional)"
              />
              <div style={{fontSize:11, color:'var(--argent)', marginTop:4}}>Note: This address is sent to the server. If backend uses SMTP auth, it might overwrite this.</div>
            </div>

            <div style={{ marginBottom: 16 }}>
              <label style={labelStyle}>To</label>
              <textarea 
                value={to} 
                onChange={e => setTo(e.target.value)}
                style={{ ...inputStyle, minHeight: 60, fontFamily: 'inherit' }}
                placeholder="recipient@example.com, ..."
                required
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
              <div>
                <label style={labelStyle}>CC</label>
                <input 
                  type="text" 
                  value={cc} 
                  onChange={e => setCc(e.target.value)}
                  style={inputStyle}
                />
              </div>
              <div>
                <label style={labelStyle}>BCC</label>
                <input 
                  type="text" 
                  value={bcc} 
                  onChange={e => setBcc(e.target.value)}
                  style={inputStyle}
                />
              </div>
            </div>

            {/* Template & AI Tools Section */}
            <div style={{ marginBottom: 16, padding: '12px', background: '#f8fafc', borderRadius: 8, border: '1px solid var(--neutral-border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <label style={{...labelStyle, marginBottom: 0}}>Email Template & AI Tools</label>
                <span style={{ fontSize: 11, color: 'var(--argent)' }}>Use <b>[name]</b> for candidate name.</span>
              </div>
              <div style={{ display: 'flex', gap: 10 }}>
                <select 
                  value={selectedTemplate} 
                  onChange={handleLoadTemplate}
                  style={{ ...inputStyle, flex: 1 }}
                >
                  <option value="">-- Load a Template --</option>
                  {templates.map(t => <option key={t.name} value={t.name}>{t.name}</option>)}
                </select>
                <button 
                  type="button" 
                  onClick={handleSaveTemplate}
                  className="btn-secondary"
                  style={{ padding: '8px 12px' }}
                >
                  Save
                </button>
                {/* Delete button added */}
                <button 
                  type="button" 
                  onClick={handleDeleteTemplate}
                  disabled={!selectedTemplate}
                  className="btn-danger"
                  style={{ 
                    padding: '8px 12px',
                    opacity: selectedTemplate ? 1 : 0.5,
                    cursor: selectedTemplate ? 'pointer' : 'not-allowed'
                  }}
                >
                  Delete
                </button>
                <button 
                  type="button" 
                  onClick={() => setShowAiInput(!showAiInput)}
                  style={{ 
                    padding: '8px 12px', borderRadius: 6, border: 'none', 
                    background: 'linear-gradient(135deg, var(--cool-blue), var(--azure-dragon))', color: '#fff', fontWeight: 600, cursor: 'pointer',
                    display: 'flex', alignItems: 'center', gap: 6
                  }}
                >
                  ✨ Draft with AI
                </button>
              </div>
              
              {showAiInput && (
                <div style={{ marginTop: 12, padding: 12, background: '#fff', borderRadius: 6, border: '1px solid var(--neutral-border)', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                  <label style={{ display: 'block', marginBottom: 6, fontSize: 12, fontWeight: 500, color: 'var(--muted)' }}>
                    What kind of email do you want to write?
                  </label>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <input 
                      type="text" 
                      value={aiPrompt} 
                      onChange={e => setAiPrompt(e.target.value)}
                      placeholder="e.g. Reject candidate nicely, Follow up on interview..." 
                      style={{ ...inputStyle, flex: 1 }}
                      onKeyDown={e => e.key === 'Enter' && (e.preventDefault(), handleAiDraft())}
                    />
                    <button 
                      type="button" 
                      onClick={handleAiDraft}
                      disabled={aiLoading}
                      className="btn-primary"
                      style={{ 
                        padding: '0 16px', cursor: aiLoading ? 'wait' : 'pointer'
                      }}
                    >
                      {aiLoading ? 'Drafting...' : 'Go'}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Calendar / Google Meet Section */}
            <div style={{ marginBottom: 16, padding: '12px', background: '#fffef6', borderRadius: 8, border: '1px solid #fde68a' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <div style={{ fontWeight: 700, color: '#92400e' }}>Calendar & Google Meet</div>
                <div style={{ fontSize: 12, color: '#92400e' }}>
                  <button
                    type="button"
                    onClick={handleConnectCalendar}
                    className="btn-secondary"
                    style={{ padding: '6px 10px' }}
                  >
                    Connect Calendar
                  </button>
                </div>
              </div>

              <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input type="checkbox" checked={addMeet} onChange={e => setAddMeet(e.target.checked)} />
                  <span style={{ fontSize: 13, fontWeight: 700 }}>Add Google Meet</span>
                </label>

                <button
                  type="button"
                  onClick={handleFindSlots}
                  disabled={!addMeet || slotsLoading}
                  className="btn-secondary"
                  style={{ padding: '6px 10px' }}
                >
                  {slotsLoading ? 'Finding slots...' : 'Find Available Slots'}
                </button>

                {calendarError && <div style={{ color: '#b91c1c', fontSize: 13 }}>{calendarError}</div>}
              </div>

              {calendarSlots && calendarSlots.length > 0 && addMeet && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 13, marginBottom: 8 }}>Select a slot to create a Meet:</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {calendarSlots.map((s, i) => (
                      <label key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, background: '#fff', padding: 8, borderRadius: 6, border: selectedSlotIndex === i ? '1px solid #2563eb' : '1px solid #e5e7eb' }}>
                        <input type="radio" name="slot" checked={selectedSlotIndex === i} onChange={() => setSelectedSlotIndex(i)} />
                        <div style={{ fontSize: 13 }}>
                          <div style={{ fontWeight: 700 }}>{new Date(s.start).toLocaleString()} — {new Date(s.end).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                          <div style={{ fontSize: 12, color: '#64748b' }}>{new Date(s.start).toLocaleDateString()}</div>
                        </div>
                      </label>
                    ))}
                  </div>

                  <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
                    <button type="button" onClick={handleCreateEvent} disabled={creatingEvent || selectedSlotIndex == null} className="btn-primary" style={{ padding: '6px 12px' }}>
                      {creatingEvent ? 'Creating...' : 'Create Event & Attach'}
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        // Insert meet link preview into body if available
                        if (meetLink) {
                          if (!body.includes(meetLink)) setBody(prev => prev + '\n\nJoin meeting: ' + meetLink);
                        } else {
                          alert('No meet link present. Create event first.');
                        }
                      }}
                      disabled={!meetLink}
                      className="btn-secondary"
                      style={{ padding: '6px 12px' }}
                    >
                      Insert Meet Link into Message
                    </button>

                    {meetLink && (
                      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <a href={meetLink} target="_blank" rel="noopener noreferrer" style={{ fontSize: 13, color: '#065f46', textDecoration: 'underline' }}>Open Meet</a>
                        <span style={{ fontSize: 12, color: '#047857' }}>Meet created</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div style={{ marginBottom: 16 }}>
              <label style={labelStyle}>Subject</label>
              <input 
                type="text" 
                value={subject} 
                onChange={e => setSubject(e.target.value)}
                style={inputStyle}
                placeholder="Enter subject here..."
                required
              />
            </div>

            <div style={{ marginBottom: 16 }}>
              <label style={labelStyle}>Message</label>
              <textarea 
                value={body} 
                onChange={e => setBody(e.target.value)}
                style={{ ...inputStyle, minHeight: 200, fontFamily: 'inherit', resize: 'vertical' }}
                placeholder="Type your message..."
              />
            </div>

            <div style={{ marginBottom: 8 }}>
              <label style={labelStyle}>Attachments</label>
              <div style={{ 
                border: '2px dashed var(--desired-dawn)', borderRadius: 8, padding: 20, 
                textAlign: 'center', background: '#f8fafc', cursor: 'pointer', position: 'relative'
              }}>
                <input 
                  type="file" 
                  multiple 
                  onChange={handleFileChange}
                  style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', opacity: 0, cursor: 'pointer' }}
                />
                <div style={{ color: 'var(--argent)', fontSize: 14 }}>
                  {files.length > 0 ? (
                    <div style={{ color: 'var(--muted)', fontWeight: 500 }}>
                      {files.length} file(s) selected: {files.map(f => f.name).join(', ')}
                    </div>
                  ) : (
                    <span>Click or drag files here to attach</span>
                  )}
                </div>
              </div>
            </div>
          </form>
        </div>

        {/* Footer with TWO Send Options */}
        <div style={{ padding: '16px 24px', borderTop: '1px solid var(--neutral-border)', display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
          <button 
            type="button"
            onClick={onClose}
            className="btn-secondary"
            style={{ padding: '8px 16px' }}
          >
            Cancel
          </button>
          
          <button 
            type="button"
            onClick={handleOpenClient}
            disabled={sending}
            className="btn-secondary"
            style={{ padding: '8px 16px', color: 'var(--accent)', borderColor: 'var(--accent)' }}
          >
            Open in Email Client
          </button>

          <button 
            type="button"
            onClick={handleDirectSend}
            disabled={directSending}
            className="btn-primary"
            style={{ 
              padding: '8px 24px', display: 'flex', alignItems: 'center', gap: 8
            }}
          >
             {directSending ? 'Sending...' : 'Send Email'}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ========================= SMTP CONFIG MODAL ========================= */
function SmtpConfigModal({ isOpen, onClose, onSave, currentConfig }) {
  const [host, setHost] = useState('');
  const [port, setPort] = useState('587');
  const [user, setUser] = useState('');
  const [pass, setPass] = useState('');
  const [secure, setSecure] = useState(false);

  useEffect(() => {
    if (isOpen && currentConfig) {
      setHost(currentConfig.host || 'smtp.gmail.com');
      setPort(currentConfig.port || '587');
      setUser(currentConfig.user || '');
      setPass(currentConfig.pass || '');
      setSecure(currentConfig.secure || false);
    } else if (isOpen && !currentConfig) {
      setHost('smtp.gmail.com');
      setPort('587');
      setUser('');
      setPass('');
      setSecure(false);
    }
  }, [isOpen, currentConfig]);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave({ host, port, user, pass, secure });
  };

  const labelStyle = { display: 'block', marginBottom: 6, fontWeight: 700, fontSize: 13, color: 'var(--muted)' };
  const inputStyle = { width: '100%', padding: '8px', marginBottom: 12, boxSizing: 'border-box' };

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 10001
    }} onClick={onClose}>
      <div className="app-card" style={{ width: 400, padding: 24 }} onClick={e => e.stopPropagation()}>
        <h3 style={{ marginTop: 0, marginBottom: 16, color: 'var(--azure-dragon)' }}>SMTP Configuration</h3>
        
        <label style={labelStyle}>Host</label>
        <input type="text" value={host} onChange={e => setHost(e.target.value)} style={inputStyle} placeholder="smtp.example.com" />
        
        <label style={labelStyle}>Port</label>
        <input type="text" value={port} onChange={e => setPort(e.target.value)} style={inputStyle} placeholder="587" />
        
        <label style={labelStyle}>User</label>
        <input type="text" value={user} onChange={e => setUser(e.target.value)} style={inputStyle} placeholder="user@example.com" />
        
        <label style={labelStyle}>Password</label>
        <input type="password" value={pass} onChange={e => setPass(e.target.value)} style={inputStyle} />
        
        <div style={{ marginBottom: 20 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, cursor: 'pointer', fontFamily: 'Orbitron, sans-serif' }}>
            <input type="checkbox" checked={secure} onChange={e => setSecure(e.target.checked)} />
            Use Secure Connection (TLS/SSL)
          </label>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
          <button onClick={onClose} className="btn-secondary" style={{ padding: '8px 16px' }}>Cancel</button>
          <button onClick={handleSave} className="btn-primary" style={{ padding: '8px 16px' }}>Save Config</button>
        </div>
      </div>
    </div>
  );
}

/* ========================= STATUS MANAGER MODAL ========================= */
function StatusManagerModal({ isOpen, onClose, statuses, onAddStatus, onRemoveStatus }) {
  const [newStatus, setNewStatus] = useState('');

  if (!isOpen) return null;

  const handleAdd = () => {
    if (newStatus.trim() && !statuses.includes(newStatus.trim())) {
      onAddStatus(newStatus.trim());
      setNewStatus('');
    }
  };

  const inputStyle = { width: '100%', padding: '8px', marginBottom: 12, boxSizing: 'border-box' };

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 10001
    }} onClick={onClose}>
      <div className="app-card" style={{ width: 400, padding: 24 }} onClick={e => e.stopPropagation()}>
        <h3 style={{ marginTop: 0, marginBottom: 16, color: 'var(--azure-dragon)' }}>Manage Status Labels</h3>
        
        <div style={{ marginBottom: 16 }}>
          <h4 style={{ fontSize: 13, marginBottom: 8, color: 'var(--muted)' }}>Existing Statuses:</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {statuses.map(s => (
              <span key={s} style={{ 
                background: '#f1f5f9', border: '1px solid #cbd5e1', borderRadius: 4, 
                padding: '4px 8px', fontSize: 12, color: '#334155',
                display: 'flex', alignItems: 'center', gap: 6
              }}>
                {s}
                <button 
                  onClick={() => onRemoveStatus && onRemoveStatus(s)}
                  style={{
                    border: 'none', background: 'transparent', color: '#ef4444', 
                    fontSize: 14, fontWeight: 'bold', cursor: 'pointer', padding: 0, lineHeight: 1
                  }}
                  title="Remove"
                >×</button>
              </span>
            ))}
          </div>
        </div>

        <label style={{ display: 'block', marginBottom: 6, fontWeight: 700, fontSize: 13, color: 'var(--muted)' }}>Add New Status</label>
        <div style={{ display: 'flex', gap: 8 }}>
          <input 
            type="text" 
            value={newStatus} 
            onChange={e => setNewStatus(e.target.value)} 
            style={{ ...inputStyle, marginBottom: 0 }} 
            placeholder="e.g. In Progress" 
            onKeyDown={e => e.key === 'Enter' && handleAdd()}
          />
          <button onClick={handleAdd} className="btn-primary" style={{ padding: '8px 16px' }}>Add</button>
        </div>
        
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12, marginTop: 24 }}>
          <button onClick={onClose} className="btn-secondary" style={{ padding: '8px 16px' }}>Close</button>
        </div>
      </div>
    </div>
  );
}

/* ========================= CANDIDATES TABLE ========================= */
function CandidatesTable({
  candidates = [],
  onDelete, deleteError, onSave, onAutoSave, type, page, setPage, totalPages, editRows, setEditRows,
  skillsetMapping,
  filters, onChangeFilter, onClearAllFilters,
  onViewProfile, // NEW PROP to handle viewing profile
  statusOptions, // Prop for status options
  onOpenStatusModal, // Prop to open status modal
  allCandidates // Passed for bulk verification/sync
}) {
  const COLUMN_WIDTHS_KEY = 'candidateTableColumnWidths';
  const DEFAULT_WIDTH = 140;
  const MIN_WIDTH = 90;
  const GLOBAL_MAX_WIDTH = 500;
  const FIELD_MAX_WIDTHS = { skillset: 900 };

  const [selectedIds, setSelectedIds] = useState([]);
  const [deleting, setDeleting] = useState(false);
  const [colWidths, setColWidths] = useState({});
  const [savingAll, setSavingAll] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');
  const [saveError, setSaveError] = useState('');
  
  // Sync Entries State
  const [syncLoading, setSyncLoading] = useState(false);
  const [syncMessage, setSyncMessage] = useState('');
  
  // Checkbox Rename Workflow State
  const [renameCheckboxId, setRenameCheckboxId] = useState(null);
  const [renameCategory, setRenameCategory] = useState('');
  const [renameValue, setRenameValue] = useState('');
  const [renameMessage, setRenameMessage] = useState('');
  const [renameError, setRenameError] = useState('');
  
  // Email modal & SMTP state
  const [emailModalOpen, setEmailModalOpen] = useState(false);
  const [composedToAddresses, setComposedToAddresses] = useState('');
  const [singleCandidateName, setSingleCandidateName] = useState('');
  const [smtpConfig, setSmtpConfig] = useState(null);
  const [smtpModalOpen, setSmtpModalOpen] = useState(false);

  const tableRef = useRef(null);

  const fields = [
    { key: 'name', label: 'Name', type: 'text', editable: true },
    { key: 'role', label: 'Job Title', type: 'text', editable: true },
    { key: 'organisation', label: 'Company', type: 'text', editable: true },
    { key: 'type', label: 'Product', type: 'text', editable: false },
    { key: 'sector', label: 'Sector', type: 'text', editable: true },
    { key: 'personal', label: 'Personal', type: 'text', editable: true },
    { key: 'seniority', label: 'Seniority', type: 'text', editable: true },
    { key: 'job_family', label: 'Job Family', type: 'text', editable: true },
    { key: 'skillset', label: 'Skillset', type: 'text', editable: false },
    { key: 'geographic', label: 'Geographic', type: 'text', editable: true },
    { key: 'country', label: 'Country', type: 'text', editable: true },
    { key: 'email', label: 'Email', type: 'email', editable: true },
    { key: 'mobile', label: 'Mobile', type: 'text', editable: true },
    { key: 'office', label: 'Office', type: 'text', editable: true },
    { key: 'sourcing_status', label: 'Sourcing Status', type: 'text', editable: true },
  ];

  const visibleFields = useMemo(() => fields, [fields]);

  useEffect(() => {
    const stored = (() => {
      try { return JSON.parse(localStorage.getItem(COLUMN_WIDTHS_KEY) || '{}'); } catch { return {}; }
    })();
    if (stored && typeof stored === 'object' && Object.keys(stored).length) {
      setColWidths(stored);
    } else {
      const init = {};
      fields.forEach(f => { init[f.key] = DEFAULT_WIDTH; });
      if (init.skillset < 260) init.skillset = 260;
      setColWidths(init);
    }
  }, []);

  useEffect(() => {
    if (colWidths && Object.keys(colWidths).length) {
      localStorage.setItem(COLUMN_WIDTHS_KEY, JSON.stringify(colWidths));
    }
  }, [colWidths]);

  const prevKeysRef = useRef({ ids: '', type: '' });
  useEffect(() => {
    const idsKey = candidates.map(c => c.id).join(',');
    if (
      idsKey === prevKeysRef.current.ids &&
      type === prevKeysRef.current.type
    ) return;
    prevKeysRef.current = { ids: idsKey, type };
    const initialEdit = {};
    candidates.forEach(c => {
      initialEdit[c.id] = {
        ...c,
        type: c.type ?? c.product ?? ''
      };
    });
    setEditRows(prev => ({ ...prev, ...initialEdit }));
  }, [candidates, type, setEditRows]);

  useEffect(() => { setSelectedIds([]); }, [page]);

  // Helper to reset rename workflow state
  const resetRenameState = () => {
    setRenameCheckboxId(null);
    setRenameCategory('');
    setRenameValue('');
    setRenameMessage('');
    setRenameError('');
  };

  const handleCheckboxChange = id => {
    const wasChecked = selectedIds.includes(id);
    // Update checkbox state first
    setSelectedIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
    
    if (!wasChecked) {
      // Show rename UI when checking
      setRenameCheckboxId(id);
      setRenameCategory('');
      setRenameValue('');
    } else {
      // Hide rename UI when unchecking
      if (renameCheckboxId === id) {
        resetRenameState();
      }
    }
  };
  const handleSelectAll = e => {
    if (e.target.checked) setSelectedIds(candidates.map(c => c.id));
    else setSelectedIds([]);
    // Clear rename UI on select all
    resetRenameState();
  };

  const handleSaveAll = async () => {
    if (typeof onSave !== 'function') return;
    setSavingAll(true);
    setSaveMessage('');
    setSaveError('');
    try {
      for (const c of candidates) {
        const id = c.id;
        const payload = { ...(c || {}), ...(editRows && editRows[id] ? editRows[id] : {}) };
        try {
          await onSave(id, payload);
        } catch (e) {
          console.warn('saveAll row save error', e && e.message);
        }
      }
      setSaveMessage('All visible candidates saved.');
    } catch (e) {
      setSaveError('Failed to save all candidates.');
    } finally {
      setSavingAll(false);
    }
  };

  const handleSync = async () => {
    setSyncLoading(true);
    setSyncMessage('');
    try {
      const rows = (allCandidates || []).map(r => ({
        id: r.id,
        organisation: r.organisation ?? r.company ?? '',
        jobtitle: r.role ?? r.jobtitle ?? '',
        seniority: r.seniority ?? '',
        country: r.country ?? ''
      }));

      if (!rows.length) {
          setSyncMessage('No data to sync.');
          setSyncLoading(false);
          return;
      }

      const res = await fetch('http://localhost:4000/verify-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rows }),
        credentials: 'include'
      });

      const payload = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(payload?.error || 'Sync request failed.');
      }

      const corrected = Array.isArray(payload?.corrected) ? payload.corrected : [];
      if (!corrected.length) {
        setSyncMessage('No corrections returned.');
        setSyncLoading(false);
        return;
      }

      setEditRows(prev => {
        const next = { ...prev };
        corrected.forEach(row => {
          if (row?.id == null) return;
          const id = row.id;
          const entry = { ...(next[id] ?? {}) };

          const newOrg = (row.organisation ?? row.company ?? null);
          if (newOrg != null && String(newOrg).trim() !== '') {
            entry.organisation = String(newOrg).trim();
          } else if (newOrg === null) {
            entry.organisation = '';
          }

          const jtCandidates = ['jobtitle', 'job_title', 'role', 'title', 'standardized_job_title', 'personal'];
          let foundJT = false;
          for (const key of jtCandidates) {
            if (Object.prototype.hasOwnProperty.call(row, key)) {
              const jtVal = row[key];
              if (jtVal == null || String(jtVal).trim() === '') {
                entry.personal = '';
              } else {
                entry.personal = String(jtVal).trim();
              }
              foundJT = true;
              break;
            }
          }

          if (!foundJT || entry.personal == null || String(entry.personal).trim() === '') {
            const src = (allCandidates || []).find(d => String(d?.id) === String(id));
            const fallbackJT = (src && (src.jobtitle || src.role || src.job_title || src.title)) ? (src.jobtitle || src.role || src.job_title || src.title) : '';
            if (fallbackJT && String(fallbackJT).trim() !== '') {
              entry.personal = String(fallbackJT).trim();
            } else {
              if (!Object.prototype.hasOwnProperty.call(entry, 'personal')) entry.personal = '';
            }
          }

          // Sync seniority field
          if (row.seniority !== null && row.seniority !== undefined) {
            if (String(row.seniority).trim() !== '') {
              entry.seniority = String(row.seniority).trim();
            } else {
              entry.seniority = '';
            }
          }

          // Sync country field
          if (row.country !== null && row.country !== undefined) {
            if (String(row.country).trim() !== '') {
              entry.country = String(row.country).trim();
            } else {
              entry.country = '';
            }
          }

          next[id] = entry;
        });
        return next;
      });

      setSyncMessage(`Synced ${corrected.length} row(s).`);
    } catch (err) {
      setSyncMessage(err.message || 'Sync failed.');
    } finally {
      setSyncLoading(false);
    }
  };

  const handleRenameSubmit = async () => {
    setRenameMessage('');
    setRenameError('');
    
    if (!renameCheckboxId || !renameCategory || !renameValue.trim()) {
      setRenameError('Please select a category and enter a new value.');
      return;
    }

    try {
      // Map frontend category names to database field names
      const fieldMap = {
        'Job Title': 'role',
        'Company': 'organisation',
        'Sector': 'sector',
        'Personal': 'personal',
        'Job Family': 'job_family',
        'Geographic': 'geographic',
        'Country': 'country'
      };
      
      const dbField = fieldMap[renameCategory];
      if (!dbField) {
        console.error('Invalid category selected:', renameCategory);
        setRenameError('Invalid category selected.');
        return;
      }

      // Update the edit rows state
      setEditRows(prev => ({
        ...prev,
        [renameCheckboxId]: {
          ...(prev[renameCheckboxId] || {}),
          [dbField]: renameValue.trim()
        }
      }));

      // Save to database via onSave callback
      if (typeof onSave === 'function') {
        const candidate = candidates.find(c => c.id === renameCheckboxId);
        const payload = {
          ...(candidate || {}),
          ...(editRows[renameCheckboxId] || {}),
          [dbField]: renameValue.trim()
        };
        await onSave(renameCheckboxId, payload);
      }

      // Show success message and clear rename UI after successful update
      setRenameMessage(`Successfully updated ${renameCategory} to "${renameValue.trim()}"`);
      setTimeout(() => {
        resetRenameState();
      }, 2000);
    } catch (err) {
      console.error('Rename failed:', err);
      setRenameError(`Failed to update: ${err.message || 'Unknown error'}`);
    }
  };

  const handleOpenEmailModal = () => {
    const selected = candidates.filter(c => selectedIds.includes(c.id));
    const allEmails = [];
    
    selected.forEach(c => {
      const raw = editRows[c.id]?.email ?? c.email;
      if (!raw) return;
      
      const parts = String(raw).split(/[;,]+/).map(s => s.trim()).filter(Boolean);
      allEmails.push(...parts);
    });

    const unique = [...new Set(allEmails)];
    
    setComposedToAddresses(unique.join(', '));
    
    if (selected.length === 1) {
        setSingleCandidateName(selected[0].name || '');
    } else {
        setSingleCandidateName('');
    }

    setEmailModalOpen(true);
  };

  const handleEditChange = (id, field, value) => {
    if (['skillset', 'type'].includes(field)) return;

    setEditRows(prev => {
      const prior = prev[id] || {};
      const original = (candidates && candidates.find(cc => String(cc.id) === String(id))) || {};
      const base = { ...original, ...prior };
      const nextRow = { ...base, [field]: value };

      if (field === 'role_tag' && skillsetMapping) {
        const rt = (value || '').trim();
        nextRow.skillset = rt && skillsetMapping[rt] ? skillsetMapping[rt] : '';
      }

      try {
        if (typeof onAutoSave === 'function') {
          onAutoSave(id, { ...nextRow });
        }
      } catch (e) {
        console.warn('onAutoSave call failed', e && e.message);
      }

      return { ...prev, [id]: nextRow };
    });
  };

  const multiWordSet = new Set([
    'Project Management', 'Version Control', 'Milestone Planning', 'Team Coordination',
    'Visual Style Guides', 'Team Leadership', 'Creative Direction', 'Game Design',
    'Level Design', 'Production Management'
  ]);
  function prettifySkillset(raw) {
    if (raw == null) return '';
    if (Array.isArray(raw)) {
      raw = raw.filter(v => v != null && v !== '').map(v => String(v).trim()).join(', ');
    } else if (typeof raw === 'object') {
      try {
        const vals = Object.values(raw)
          .filter(v => v != null && (typeof v === 'string' || typeof v === 'number'))
          .map(v => String(v).trim())
          .filter(Boolean);
        if (vals.length) raw = vals.join(', ');
        else raw = String(raw);
      } catch {
        raw = String(raw);
      }
    } else {
      raw = String(raw);
    }
    raw = raw.trim();
    if (!raw) return '';
    if (/[;,]/.test(raw)) {
      return raw.split(/[;,]/).map(s => s.trim()).filter(Boolean).join(', ');
    }
    const withDelims = raw.replace(/([a-z])([A-Z])/g, '$1|$2');
    let tokens = withDelims.split(/[\s|]+/).filter(Boolean);
    const merged = [];
    for (let i = 0; i < tokens.length; i++) {
      const cur = tokens[i];
      const next = tokens[i + 1];
      if (next) {
        const pair = cur + ' ' + next;
        if (multiWordSet.has(pair)) {
          merged.push(pair);
          i++;
          continue;
        }
      }
      merged.push(cur);
    }
    const deduped = merged.filter((t, i) => i === 0 || t !== merged[i - 1]);
    return deduped.join(', ');
  }

  const [colResizing, setColResizing] = useState({ active: false, field: '', startX: 0, startW: 0 });
  const onMouseDown = (field, e) => {
    e.preventDefault();
    setColResizing({ active: true, field, startX: e.clientX, startW: colWidths[field] });
  };
  useEffect(() => {
    const move = e => {
      if (!colResizing.active) return;
      setColWidths(prev => {
        const maxForField = FIELD_MAX_WIDTHS[colResizing.field] || GLOBAL_MAX_WIDTH;
        const nw = Math.max(MIN_WIDTH, Math.min(maxForField, colResizing.startW + (e.clientX - colResizing.startX)));
        return { ...prev, [colResizing.field]: nw };
      });
    };
    const up = () => setColResizing({ active: false, field: '', startX: 0, startW: 0 });
    if (colResizing.active) {
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
    }
    return () => {
      document.removeEventListener('mousemove', move);
      document.removeEventListener('mouseup', up);
    };
  }, [colResizing]);

  const autoSizeColumn = useCallback((fieldKey) => {
    if (!tableRef.current) return;
    const headerCell = tableRef.current.querySelector(`th[data-field="${fieldKey}"]`);
    let max = 0;
    if (headerCell) {
      const headerLabel = headerCell.querySelector('.header-label');
      if (headerLabel) max = headerLabel.scrollWidth;
    }
    const cells = tableRef.current.querySelectorAll(`td[data-field="${fieldKey}"]`);
    cells.forEach(cell => {
      const node = cell.firstChild;
      const w = node ? node.scrollWidth : cell.scrollWidth;
      if (w > max) max = w;
    });
    const padded = Math.ceil(max + 24);
    const maxForField = FIELD_MAX_WIDTHS[fieldKey] || GLOBAL_MAX_WIDTH;
    setColWidths(prev => ({
      ...prev,
      [fieldKey]: Math.max(MIN_WIDTH, Math.min(maxForField, padded))
    }));
  }, []);

  const resetAllColumns = () => {
    setColWidths(prev => {
      const next = {};
      Object.keys(prev).forEach(k => {
        next[k] = k === 'skillset' ? 260 : 140;
      });
      return next;
    });
  };

  const handleHeaderDoubleClick = (e, fieldKey) => {
    if (e.altKey) {
      if (fieldKey === '__ALL__') resetAllColumns();
      else setColWidths(prev => ({
        ...prev,
        [fieldKey]: fieldKey === 'skillset' ? 260 : 140
      }));
    } else {
      autoSizeColumn(fieldKey);
    }
  };

  // Converted to declared function so it's always in scope where referenced in JSX
  function handleResizerKey(e, fieldKey) {
    const step = e.shiftKey ? 30 : 10;
    const maxForField = FIELD_MAX_WIDTHS[fieldKey] || GLOBAL_MAX_WIDTH;
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      setColWidths(prev => ({ ...prev, [fieldKey]: Math.min(maxForField, (prev[fieldKey] || 140) + step) }));
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setColWidths(prev => ({ ...prev, [fieldKey] : Math.max(MIN_WIDTH, (prev[fieldKey] || 140) - step) }));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      autoSizeColumn(fieldKey);
    }
  }

  // Sticky header & name column constants
  const HEADER_ROW_HEIGHT = 38;
  const stickyHeaderCell = {
    width: 44,
    minWidth: 44,
    textAlign: 'center',
    background: '#f1f5f9',
    position: 'sticky',
    left: 0,
    top: 0,
    zIndex: 40,
    cursor: 'default',
    userSelect: 'none',
    borderRight: '1px solid var(--neutral-border)'
  };
  const stickyFilterCell = {
    ...stickyHeaderCell,
    top: HEADER_ROW_HEIGHT,
    zIndex: 39,
    background: '#ffffff'
  };
  const stickyBodyCell = {
    textAlign: 'center', background: '#fff', position: 'sticky', left: 0, zIndex: 9,
    minWidth: 44, width: 44, height: 38, overflow: 'hidden', boxShadow: '2px 0 0 var(--neutral-border)'
  };
  const nonSticky = { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' };

  return (
    <>
      <div className="app-card" style={{
        overflowX: 'auto', width: '100%', maxWidth: '100%', position: 'relative', padding: 16
      }}>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', marginBottom: 12 }}>
          {selectedIds.length > 0 && (
            <button
              disabled={!selectedIds.length || deleting}
              onClick={async () => {
                if (!selectedIds.length) return;
                setDeleting(true);
                await onDelete([...selectedIds]);
                setDeleting(false);
                setSelectedIds([]);
              }}
              className="btn-danger"
              style={{ padding: '8px 16px' }}
            >{deleting ? 'Deleting…' : 'Delete'}</button>
          )}

          <button
            onClick={onClearAllFilters}
            disabled={!Object.values(filters||{}).some(v => v)}
            className="btn-secondary"
            style={{ padding: '8px 16px' }}
          >Clear Filters</button>

          {selectedIds.length > 0 && (
            <button
              onClick={handleOpenEmailModal}
              title="Send email to selected candidates"
              className="btn-primary"
              style={{ padding: '8px 16px', display: 'flex', alignItems: 'center', gap: 6 }}
            >
              <span>✉</span> Send Email
            </button>
          )}

          <button
            onClick={onOpenStatusModal}
            className="btn-secondary"
            style={{ padding: '8px 16px' }}
          >
            Manage Statuses
          </button>
          <button
            onClick={handleSync}
            disabled={syncLoading}
            className="btn-primary"
            style={{ padding: '8px 16px' }}
          >
            {syncLoading ? 'Syncing...' : 'Sync Entries'}
          </button>

          <button
            onClick={handleSaveAll}
            disabled={savingAll}
            className="btn-primary"
            style={{ padding: '8px 16px', marginLeft: 'auto' }}
          >{savingAll ? 'Saving  ' : 'Save'}</button>

          <button
            onClick={() => setSmtpModalOpen(true)}
            className="btn-secondary"
            style={{ padding: '8px 16px', marginLeft: 0 }}
          >
            Configure SMTP
          </button>
          
          {deleteError && <div style={{ color: 'var(--danger)', fontSize: 14 }}>{deleteError}</div>}
          {saveError && <div style={{ color: 'var(--danger)', fontSize: 14 }}>{saveError}</div>}
          {saveMessage && <div style={{ color: 'var(--success)', fontSize: 14 }}>{saveMessage}</div>}
          {syncMessage && <div style={{ color: 'var(--success)', fontSize: 14 }}>{syncMessage}</div>}
        </div>

        {/* Checkbox Rename Workflow UI */}
        {renameCheckboxId && (
          <div style={{
            padding: '12px 16px',
            background: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderRadius: 8,
            marginBottom: 12,
            display: 'flex',
            gap: 12,
            alignItems: 'center',
            flexWrap: 'wrap'
          }}>
            <span style={{ fontSize: 14, fontWeight: 600, color: '#334155' }}>
              Rename field for selected record:
            </span>
            
            <select
              value={renameCategory}
              onChange={(e) => setRenameCategory(e.target.value)}
              style={{
                padding: '6px 12px',
                fontSize: 14,
                border: '1px solid #cbd5e1',
                borderRadius: 6,
                background: '#ffffff',
                cursor: 'pointer'
              }}
            >
              <option value="">Select Category...</option>
              <option value="Job Title">Job Title</option>
              <option value="Company">Company</option>
              <option value="Sector">Sector</option>
              <option value="Personal">Personal</option>
              <option value="Job Family">Job Family</option>
              <option value="Geographic">Geographic</option>
              <option value="Country">Country</option>
            </select>

            {renameCategory && (
              <>
                <input
                  type="text"
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  placeholder={`Enter new ${renameCategory}...`}
                  style={{
                    padding: '6px 12px',
                    fontSize: 14,
                    border: '1px solid #cbd5e1',
                    borderRadius: 6,
                    minWidth: 250,
                    background: '#ffffff'
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleRenameSubmit();
                    }
                  }}
                />
                
                <button
                  onClick={handleRenameSubmit}
                  className="btn-primary"
                  style={{ padding: '6px 16px', fontSize: 14 }}
                >
                  Update
                </button>
                
                <button
                  onClick={resetRenameState}
                  className="btn-secondary"
                  style={{ padding: '6px 16px', fontSize: 14 }}
                >
                  Cancel
                </button>
                
                {renameError && <div style={{ color: 'var(--danger)', fontSize: 14, width: '100%' }}>{renameError}</div>}
                {renameMessage && <div style={{ color: 'var(--success)', fontSize: 14, width: '100%' }}>{renameMessage}</div>}
              </>
            )}
          </div>
        )}

        <div style={{ overflowX: 'auto', width: '100%', maxWidth: '100%' }}>
          <table
            ref={tableRef}
            style={{
              minWidth: visibleFields.length * 110 + 110,
              width: '100%',
              borderCollapse: 'separate',
              borderSpacing: 0,
              marginBottom: 12,
              tableLayout: 'fixed'
            }}
          >
            <thead>
              <tr>
                <th
                  style={{ ...stickyHeaderCell, background: '#f1f5f9', fontSize: 12, fontWeight: 700, fontFamily: "Orbitron" }}
                  onDoubleClick={(e) => handleHeaderDoubleClick(e, '__ALL__')}
                >
                  <div style={{ position: 'relative', height: HEADER_ROW_HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <input
                      type="checkbox"
                      checked={candidates.length > 0 && selectedIds.length === candidates.length}
                      onChange={handleSelectAll}
                      style={{ cursor: 'pointer' }}
                    />
                  </div>
                </th>
                {visibleFields.map(f => {
                  const maxForField = FIELD_MAX_WIDTHS[f.key] || GLOBAL_MAX_WIDTH;
                  const isName = f.key === 'name';
                  const stickyStyle = isName ? {
                      position: 'sticky',
                      left: 44,
                      top: 0,
                      zIndex: 39,
                      borderRight: '1px solid var(--neutral-border)'
                  } : {};
                  return (
                    <th
                      key={f.key}
                      data-field={f.key}
                      onDoubleClick={(e) => handleHeaderDoubleClick(e, f.key)}
                      style={{
                        position: 'sticky',
                        top: 0,
                        width: colWidths[f.key],
                        minWidth: MIN_WIDTH,
                        maxWidth: maxForField,
                        background: '#f1f5f9',
                        userSelect: 'none',
                        padding: '6px 8px 4px',
                        verticalAlign: 'bottom',
                        fontSize: 12,
                        fontWeight: 700,
                        color: 'var(--muted)',
                        borderBottom: '1px solid var(--neutral-border)',
                        borderRight: '1px solid var(--neutral-border)',
                        fontFamily: "Orbitron",
                        ...stickyStyle
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 4 }}>
                        <span className="header-label" style={{ flex: '1 1 auto' }}>{f.label}</span>
                        <span
                          role="separator"
                          tabIndex={0}
                          style={{
                            cursor: 'col-resize',
                            padding: '0 4px',
                            userSelect: 'none',
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            fontSize: 14,
                            lineHeight: 1,
                            color: 'var(--argent)'
                          }}
                          onMouseDown={e => onMouseDown(f.key, e)}
                          onKeyDown={e => handleResizerKey(e, f.key)}
                        >▕</span>
                      </div>
                    </th>
                  );
                })}
                <th style={{
                  width: 110, 
                  position: 'sticky',
                  top: 0,
                  background: '#f1f5f9',
                  fontSize: 12,
                  fontWeight: 700,
                  color: 'var(--muted)',
                  borderBottom: '1px solid var(--neutral-border)',
                  fontFamily: "Orbitron",
                  zIndex: 38
                }}>Actions</th>
              </tr>
              <tr>
                <th style={{ ...stickyFilterCell, borderBottom: '1px solid var(--neutral-border)' }}>
                  <span style={{ fontSize: 10, color: 'var(--argent)', fontWeight: 500 }}>Filters</span>
                </th>
                {visibleFields.map(f => {
                  const maxForField = FIELD_MAX_WIDTHS[f.key] || GLOBAL_MAX_WIDTH;
                  const isName = f.key === 'name';
                  const stickyStyle = isName ? {
                      position: 'sticky',
                      left: 44,
                      top: HEADER_ROW_HEIGHT,
                      zIndex: 29,
                      borderRight: '1px solid var(--neutral-border)'
                  } : {};
                  return (
                    <th
                      key={'filter-' + f.key}
                      style={{
                        position: 'sticky',
                        top: HEADER_ROW_HEIGHT,
                        width: colWidths[f.key],
                        minWidth: MIN_WIDTH,
                        maxWidth: maxForField,
                        background: '#ffffff',
                        padding: 4,
                        borderBottom: '1px solid var(--neutral-border)',
                        borderRight: '1px solid #f1f5f9',
                        ...stickyStyle
                      }}
                    >
                      <input
                        type="text"
                        value={filters[f.key] || ''}
                        onChange={e => onChangeFilter(f.key, e.target.value)}
                        placeholder="Filter..."
                        style={{
                          width: '100%',
                          boxSizing: 'border-box',
                          padding: '4px 6px',
                          fontSize: 12,
                          background: '#f8fafc'
                        }}
                      />
                    </th>
                  );
                })}
                <th style={{ position: 'sticky', top: HEADER_ROW_HEIGHT, background: '#ffffff', borderBottom: '1px solid var(--neutral-border)' }} />
              </tr>
            </thead>
            <tbody>
              {candidates.map((c, idx) => (
                <tr key={c.id} style={{ background: idx % 2 ? '#ffffff' : '#f9fafb' }}>
                  <td style={stickyBodyCell}>
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(c.id)}
                      onChange={() => handleCheckboxChange(c.id)}
                      style={{ cursor: 'pointer' }}
                    />
                  </td>
                  {visibleFields.map(f => {
                    const readOnly = ['skillset', 'type'].includes(f.key);
                    const maxForField = FIELD_MAX_WIDTHS[f.key] || GLOBAL_MAX_WIDTH;
                    const isName = f.key === 'name';
                    const stickyStyle = isName ? {
                        position: 'sticky',
                        left: 44,
                        zIndex: 9,
                        background: idx % 2 ? '#ffffff' : '#f9faffb',
                        borderRight: '1px solid #eef2f5',
                        boxShadow: '2px 0 0 rgba(0,0,0,0.02)'
                    } : {};

                    let displayValue = editRows[c.id]?.[f.key] ?? '';
                    if (displayValue === '' || displayValue == null) {
                      if (f.key === 'type') displayValue = c.type ?? c.product ?? '';
                      else displayValue = c[f.key] ?? '';
                    }
                    if (f.key === 'skillset') {
                      displayValue = prettifySkillset(displayValue);
                    }
                    return (
                      <td
                        key={f.key}
                        data-field={f.key}
                        style={{
                          ...nonSticky,
                          width: colWidths[f.key],
                          maxWidth: maxForField,
                          minWidth: MIN_WIDTH,
                          padding: '4px 6px',
                          verticalAlign: 'middle',
                          fontSize: 13,
                          color: 'var(--muted)',
                          borderBottom: '1px solid #eef2f5',
                          ...stickyStyle
                        }}
                      >
                        {readOnly
                          ? <span style={{
                            display: 'block',
                            width: '100%',
                            background: f.key === 'skillset' ? '#fff' : '#f1f5f9',
                            padding: '3px 8px',
                            border: '1px solid var(--neutral-border)',
                            borderRadius: 4,
                            boxSizing: 'border-box',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            fontSize: 12
                          }}
                            title={displayValue}
                          >
                            {displayValue}
                          </span>
                          : f.key === 'sourcing_status' ? (
                            <select
                              value={displayValue || 'Reviewing'}
                              onChange={e => handleEditChange(c.id, f.key, e.target.value)}
                              style={{
                                width: '100%',
                                boxSizing: 'border-box',
                                padding: '4px 8px',
                                font: 'inherit',
                                fontSize: 12,
                                background: '#ffffff',
                                border: '1px solid var(--desired-dawn)',
                                borderRadius: 6
                              }}
                            >
                              {statusOptions.map(opt => (
                                <option key={opt} value={opt}>{opt}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type={f.type}
                              value={displayValue}
                              onChange={e => handleEditChange(c.id, f.key, e.target.value)}
                              style={{
                                width: '100%',
                                boxSizing: 'border-box',
                                padding: '4px 8px',
                                font: 'inherit',
                                fontSize: 12,
                                background: '#ffffff'
                              }}
                            />
                          )
                        }
                      </td>
                    );
                  })}
                  <td style={{ background: '#ffffff', textAlign: 'center', borderBottom: '1px solid #eef2f5' }}>
                   <div style={{display:'flex', alignItems:'center', justifyContent:'center', gap:4}}>
                    <button
                        onClick={() => onViewProfile && onViewProfile(c)}
                        title="View Resume & Profile"
                        style={{
                            background: 'var(--azure-dragon)',
                            color: '#fff',
                            border: 'none',
                            padding: '6px 10px',
                            borderRadius: 6,
                            cursor: 'pointer',
                            fontSize: 12,
                            fontWeight: 700
                        }}
                    >
                        Profile
                    </button>
                   </div>
                  </td>
                </tr>
              ))}
              {!candidates.length && (
                <tr>
                  <td colSpan={visibleFields.length + 2} style={{ padding: 16, textAlign: 'center', color: 'var(--argent)', fontSize: 14 }}>
                    No candidates match current filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 14, marginBottom: 4, alignItems: 'center' }}>
            <button disabled={page <= 1} onClick={() => setPage(page - 1)} className="btn-secondary" style={{ padding: '6px 14px' }}>Prev</button>
            <span style={{ fontSize: 13, color: 'var(--muted)', fontFamily: 'Orbitron' }}>Page {page} of {totalPages}</span>
            <button disabled={page >= totalPages} onClick={() => setPage(page + 1)} className="btn-secondary" style={{ padding: '6px 14px' }}>Next</button>
          </div>
        </div>
      </div>
      <EmailComposeModal 
        isOpen={emailModalOpen}
        onClose={() => setEmailModalOpen(false)}
        toAddresses={composedToAddresses}
        candidateName={singleCandidateName}
        smtpConfig={smtpConfig}
      />
      <SmtpConfigModal
        isOpen={smtpModalOpen}
        onClose={() => setSmtpModalOpen(false)}
        onSave={(cfg) => { setSmtpConfig(cfg); setSmtpModalOpen(false); }}
        currentConfig={smtpConfig}
      />
    </>
  );
}

/* ========================= ORG CHART CORE ========================= */
// ... (OrgChart logic unchanged, just applying styles via classes implicitly) ...
function buildOrgChartTrees(candidates, manualParentOverrides, editingLayout, draggingId, onManualDrop) {
  // Logic mostly identical, ensuring consistent styling
  const LAYERS = ['Executive','Sr Director','Director','Sr Manager','Manager','Lead','Expert','Senior','Mid','Junior'];
  const rankOf = tier => {
    const t = normalizeTier(tier);
    const i = LAYERS.indexOf(t);
    return i === -1 ? LAYERS.length : i;
  };
  const ALLOWED_PARENTS = {
    'Junior': ['Lead','Manager','Sr Manager','Director','Sr Director'],
    'Mid': ['Lead','Manager','Sr Manager','Director','Sr Director'],
    'Senior': ['Lead','Manager','Sr Manager','Director','Sr Director'],
    'Expert': ['Lead','Manager','Sr Manager','Director','Sr Director'],
    'Lead': ['Manager','Sr Manager','Director','Sr Director','Lead'],
    'Manager': ['Director','Sr Director'],
    'Sr Manager': ['Director','Sr Director'],
    'Director': ['Sr Director','Executive'],
    'Sr Director': ['Executive'],
    'Executive': []
  };
  const allowedParentsFor = tier => ALLOWED_PARENTS[normalizeTier(tier)] || [];

  const grouped = new Map();
  (candidates||[]).forEach(c=>{
    if(!isHumanName(c.name)) return;
    const org=(c.organisation||'N/A').trim()||'N/A';
    const fam=(c.job_family||'N/A').trim()||'N/A';
    if(!grouped.has(org)) grouped.set(org,new Map());
    const famMap=grouped.get(org);
    if(!famMap.has(fam)) famMap.set(fam,[]);
    famMap.get(fam).push(c);
  });

  const charts=[];
  for(const org of [...grouped.keys()].sort()){
    const famMap=grouped.get(org);
    for(const family of [...famMap.keys()].sort()){
      const people=famMap.get(family)||[];
      const nodes = people.map(p=>{
        const tier=inferSeniority(p);
        return {
          id:p.id, name:p.name, seniority:tier,
          roleTag:(p.role_tag||'').trim(),
          personal:(p.personal||'').trim(), 
          jobFamily:p.job_family||'',
          country:(p.country||'').trim(),
          geographic:(p.geographic||'').trim(),
          rank:rankOf(tier),
          raw:p
        };
      }).filter(n=> n.id!=null && n.name && n.seniority);

      if(!nodes.length){
        charts.push(
          <div key={`${org}:::${family}`} className="org-group" data-group-key={`${org}:::${family}`} style={{padding:12,marginBottom:16}}>
            <div className="org-header" style={{textAlign:'center',fontWeight:600,marginBottom:8}}>
              Organisation: <b>{org}</b> | Job Family: <b>{family}</b>
            </div>
            <div style={{color:'#64748b'}}>Not Applicable</div>
          </div>
        );
        continue;
      }

      const stableKey=n=>`${String(n.name||'').toLowerCase()}|${n.id}`;
      nodes.sort((a,b)=> a.rank - b.rank || stableKey(a).localeCompare(stableKey(b)));

      const byId=new Map(nodes.map(n=>[n.id,n]));
      const parent=new Map();
      const children=new Map(nodes.map(n=>[n.id,[]]));
      const load=new Map(nodes.map(n=>[n.id,0]));

      const sameCountry=(a,b)=> a.country && b.country && a.country.toLowerCase()===b.country.toLowerCase();
      const sameGeo=(a,b)=> a.geographic && b.geographic && a.geographic.toLowerCase()===b.geographic.toLowerCase();
      const canEqualTier=tier=> normalizeTier(tier)==='Lead';
      const isEqual=(a,b)=> normalizeTier(a)===normalizeTier(b);

      function chooseParent(child,buckets){
        const allowed=allowedParentsFor(child.seniority);
        for(const bucket of buckets){
          for(const pref of allowed){
            let subset=bucket.filter(c=> normalizeTier(c.seniority)===pref);
            if(isEqual(child.seniority,pref)){
              if(!canEqualTier(child.seniority)) subset=[];
              else subset=subset.filter(c=> stableKey(c) < stableKey(child));
            }
            if(!subset.length) continue;
            subset.sort((a,b)=>{
              const la=load.get(a.id)||0, lb=load.get(b.id)||0;
              if(la!==lb) return la-lb;
              if(a.rank!==b.rank) return a.rank - b.rank;
              return stableKey(a).localeCompare(stableKey(b));
            });
            return subset[0];
          }
        }
        return null;
      }

      for(const child of nodes){
        const eligible=nodes.filter(c=>{
          if(c.id===child.id) return false;
          return allowedParentsFor(child.seniority).includes(normalizeTier(c.seniority));
        });
        const sameRole=eligible.filter(e=> (e.roleTag||'')===(child.roleTag||''));
        const sr_country=sameRole.filter(e=> sameCountry(child,e));
        const sr_geo=sameRole.filter(e=> !sameCountry(child,e)&&sameGeo(child,e));
        const sr_any=sameRole;
        const otherRole=eligible.filter(e=> (e.roleTag||'')!==(child.roleTag||''));
        const or_country=otherRole.filter(e=> sameCountry(child,e));
        const or_geo=otherRole.filter(e=> !sameCountry(child,e)&&sameGeo(child,e));
        const or_any=otherRole;
        const buckets=[sr_country,sr_geo,sr_any,or_country,or_geo,or_any];
        const chosen=chooseParent(child,buckets);
        if(chosen){
          parent.set(child.id,chosen.id);
          children.get(chosen.id).push(child.id);
          load.set(chosen.id,(load.get(chosen.id)||0)+1);
        }
      }

      function buildDescendants(rootId, acc=new Set()){
        const ch=children.get(rootId)||[];
        for(const cid of ch){
          if(!acc.has(cid)){
            acc.add(cid);
            buildDescendants(cid,acc);
          }
        }
        return acc;
      }
      for(const [childStr,newParentId] of Object.entries(manualParentOverrides||{})){
        const childId=Number(childStr);
        if(!byId.has(childId)) continue;
        if(newParentId!=null && !byId.has(newParentId)) continue;
        const oldP=parent.get(childId);
        if(oldP!=null){
            const arr=children.get(oldP)||[];
            const idx=arr.indexOf(childId);
            if(idx>=0) arr.splice(idx,1);
            children.set(oldP,arr);
            parent.delete(childId);
        }
        if(newParentId==null) continue;
        if(childId===newParentId) continue;
        const desc=buildDescendants(childId);
        if(desc.has(newParentId)) continue;
        parent.set(childId,newParentId);
        const parr=children.get(newParentId)||[];
        if(!parr.includes(childId)) parr.push(childId);
        children.set(newParentId,parr);
      }

      const roots=nodes.filter(n=> !parent.has(n.id));

      const handleDragStart=(e,node)=>{
        if(!editingLayout) return;
        e.stopPropagation();
        e.dataTransfer.setData('text/plain', String(node.id));
        e.dataTransfer.effectAllowed='move';
      };
      const handleDragOver=e=>{
        if(!editingLayout) return;
        e.preventDefault();
        e.dataTransfer.dropEffect='move';
      };
      const handleDropOnNode=(e,target)=>{
        if(!editingLayout) return;
        e.preventDefault();
        const draggedId=Number(e.dataTransfer.getData('text/plain'));
        if(!draggedId || draggedId===target.id) return;
        onManualDrop(draggedId,target.id);
      };

      /* NodeCard: improved rendering to avoid duplicated seniority words and use short badge labels (Sr / Jr) */
      const NodeCard=({node})=>{
        const normalizedSen = normalizeTier(node.seniority);
        const isMgr=['Lead','Manager','Sr Manager','Director','Sr Director','Executive'].includes(normalizedSen);

        // Build base role: prefer personal (standardized), then roleTag, then raw.role
        let baseRole = (node.personal||'').trim() || (node.roleTag||'').trim() || (node.raw?.role ? String(node.raw.role).trim() : '') || '';

        // Remove leading seniority tokens from baseRole to avoid duplication when we prefix seniority
        function stripLeadingSeniority(s){
          if(!s) return s;
          return s.replace(/^(?:(sr|senior|lead|principal|expert|manager|mgr|director|dir|executive|exec|jr|junior|mid)\b[\s\.\-:]*)+/i, '').trim();
        }
        function collapseDuplicates(s){
          if(!s) return s;
          const toks = s.split(/\s+/);
          const dedup = toks.filter((t,i)=> i===0 || t.toLowerCase() !== toks[i-1].toLowerCase());
          return dedup.join(' ');
        }

        baseRole = baseRole.replace(/\s{2,}/g,' ').trim();
        if (isMgr && baseRole) baseRole = stripLeadingSeniority(baseRole);
        baseRole = collapseDuplicates(baseRole);

        // If baseRole emptied, fallback to raw.role trimmed
        if(!baseRole && node.raw?.role) baseRole = String(node.raw.role).trim();

        // Compose title
        let title;
        if (isMgr && normalizedSen) {
          const baseLower = (baseRole||'').toLowerCase();
          const senLower = normalizedSen.toLowerCase();
          if (baseLower.startsWith(senLower)) {
            title = baseRole;
          } else {
            title = `${normalizedSen}${baseRole ? ' ' + baseRole : ''}`;
          }
        } else {
          title = baseRole || normalizedSen || '';
        }

        // Badge text mapping: use short tokens (Sr, Jr, Mid, Lead, Mgr, Dir, Exec, Expert)
        const badge = (() => {
          const t = normalizeTier(node.seniority);
          if (!t) return '';
          const map = {
            'Senior': 'Sr',
            'Sr Manager': 'Sr Mgr',
            'Sr Director': 'Sr Dir',
            'Junior': 'Jr',
            'Mid': 'Mid',
            'Lead': 'Lead',
            'Manager': 'Mgr',
            'Director': 'Dir',
            'Executive': 'Exec',
            'Expert': 'Expert'
          };
          if (map[t]) return map[t];
          // fallback: split and map tokens
          const tokenMap = {
            'senior': 'Sr',
            'sr': 'Sr',
            'junior': 'Jr',
            'jr': 'Jr',
            'mid': 'Mid',
            'lead': 'Lead',
            'manager': 'Mgr',
            'mgr': 'Mgr',
            'director': 'Dir',
            'dir': 'Dir',
            'executive': 'Exec',
            'exec': 'Exec',
            'expert': 'Expert'
          };
          const parts = t.split(/\s+/).map(p => tokenMap[p.toLowerCase()] || (p.charAt(0).toUpperCase() + p.slice(1)));
          return parts.join(' ');
        })();

        // Derive badgeClass (lowercased word for CSS)
        let badgeClass = '';
        if (badge) {
          const cls = badge.toLowerCase().replace(/\s+/g,'-').replace(/[^a-z0-9\-]/g,'');
          badgeClass = `label-${cls}`;
        }

        const isDragging=draggingId===node.id;
        return(
          <div
            className="org-box"
            data-node-id={node.id}
            draggable={editingLayout}
            onDragStart={e=>handleDragStart(e,node)}
            onDragOver={handleDragOver}
            onDrop={e=>handleDropOnNode(e,node)}
            style={{
              opacity:isDragging?0.4:1,
              cursor:editingLayout?'grab':'default',
              border:editingLayout?'1px solid #6366f1':'1px solid #cbd5e1',
              position:'relative',
              transition:'border-color .2s'
            }}
          >
            <span className="org-box-accent" />
            <div className="org-name">{node.name}</div>

            <div className="org-title" style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:4 }}>
              <div style={{ lineHeight: 1.15, whiteSpace: 'normal', textAlign: 'center', fontWeight:600 }}>
                {title}
              </div>
              {/* badge rendered and CSS positions it; badge text uses short tokens */}
              {badge && (
                <span className={`org-inline-label ${badgeClass}`} title={badge}>
                  {badge}
                </span>
              )}
            </div>

            <div className="org-meta" style={{ fontSize:11, color:'#64748b', marginTop:6 }}>
              {(node.country||node.geographic) && (<>{node.country||'—'}{node.geographic?` • ${node.geographic}`:''}</>)}
            </div>
            {editingLayout && <div style={{ position:'absolute', top:2, right:4, fontSize:10, color:'#64748b' }}>id:{node.id}</div>}
            
            {editingLayout && (
             <button
               onClick={(e) => {
                 e.stopPropagation();
                 onManualDrop(node.id, null);
               }}
               title="Detach / Make Root"
               style={{
                 position: 'absolute',
                 top: -8,
                 right: -8,
                 background: '#ef4444',
                 color: 'white',
                 border: '1px solid #fff',
                 borderRadius: '50%',
                 width: 20,
                 height: 20,
                 display: 'flex',
                 alignItems: 'center',
                 justifyContent: 'center',
                 fontSize: 12,
                 cursor: 'pointer',
                 zIndex: 50,
                 boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
               }}
             >
               ×
             </button>
           )}
          </div>
        );
      };

      function renderSubtree(node){
        const kidIds=[...(children.get(node.id)||[])];
        kidIds.sort((a,b)=>{
          const na=byId.get(a), nb=byId.get(b);
          if(na.rank!==nb.rank) return na.rank - nb.rank;
          return (String(na.name||'').toLowerCase()+na.id).localeCompare(String(nb.name||'').toLowerCase()+nb.id);
        });
        if(!kidIds.length) return <TreeNode key={node.id} label={<NodeCard node={node}/>}/>;
        return (
          <TreeNode key={node.id} label={<NodeCard node={node}/>}>
            {kidIds.map(id=> renderSubtree(byId.get(id)))}
          </TreeNode>
        );
      }

      charts.push(
        <div
          key={`${org}:::${family}`}
          className="org-group"
          data-group-key={`${org}:::${family}`}
          style={{
            breakAfter:'page',
            pageBreakAfter:'always',
            padding:12,
            marginBottom:16,
            border:'1px solid #f1f5f9',
            borderRadius:6,
            position:'relative'
          }}
        >
          <div style={{ textAlign:'center', fontWeight:600, marginBottom:8, position:'relative' }}>
            Organisation: <b>{org}</b> | Job Family: <b>{family}</b>
          </div>
          <div className="org-chart-scroll" style={{
            overflowX:'auto', overflowY:'auto', width:'100%', maxWidth:'98vw',
            background:'#fff', padding:12, position:'relative', maxHeight:'80vh',
            border: editingLayout ? '1px solid #6366f1' : '1px solid #e2e8f0'
          }}>
            <div className="org-center-wrapper" style={{ display:'flex', gap:48, justifyContent:'center', width:'100%' }}>
              <div style={{ display:'flex', gap:48, justifyContent:'center', alignItems:'flex-start' }}>
                {roots.map(root=>(
                  <Tree
                    key={`root-${root.id}`}
                    lineWidth={'2px'}
                    lineColor={'#6b7280'}
                    lineBorderRadius={'0px'}
                    label={<NodeCard node={root}/>}
                  >
                    {(children.get(root.id)||[]).map(cid=> renderSubtree(byId.get(cid)))}
                  </Tree>
                ))}
              </div>
            </div>
          </div>
        </div>
      );
    }
  }
  return charts;
}

/* ========================= ORG CHART DISPLAY ========================= */
function OrgChartDisplay({
  candidates,
  jobFamilyOptions,
  selectedJobFamily,
  onChangeJobFamily,
  manualParentOverrides,
  setManualParentOverrides,
  editingLayout,
  setEditingLayout,
  lastSavedOverrides,
  setLastSavedOverrides,
  showOrgChart,
  setShowOrgChart,
  organisationOptions,
  selectedOrganisation,
  onChangeOrganisation,
  countryOptions,
  selectedCountry,
  onChangeCountry
}) {
  const [orgChart, setOrgChart] = useState([]);
  const [loading, setLoading] = useState(false);
  const [draggingId, setDraggingId] = useState(null);
  const chartRef = useRef();

  const pruneOverrides = useCallback((overrides) => {
    const valid = new Set(candidates.map(c=>c.id));
    const cleaned={};
    Object.entries(overrides||{}).forEach(([child,parent])=>{
      const cNum=Number(child);
      if(!valid.has(cNum)) return;
      if(parent!=null && !valid.has(parent)) return;
      cleaned[child]=parent;
    });
    return cleaned;
  },[candidates]);

  const rebuild = useCallback(()=>{
    if(!showOrgChart){
      setOrgChart([]);
      return;
    }
    const cleaned = pruneOverrides(manualParentOverrides);
    if (JSON.stringify(cleaned) !== JSON.stringify(manualParentOverrides)) {
      setManualParentOverrides(cleaned);
    }
    setOrgChart(
      buildOrgChartTrees(
        candidates,
        cleaned,
        editingLayout,
        draggingId,
        (childId,newParentId)=>{
          setManualParentOverrides(prev=>({...prev,[childId]:newParentId}));
        }
      )
    );
  }, [candidates, draggingId, editingLayout, manualParentOverrides, pruneOverrides, setManualParentOverrides, showOrgChart]);

  useEffect(()=>{ rebuild(); },[rebuild]);

  const adjustCentering = useCallback(()=>{
    if(!chartRef.current) return;
    const groups = chartRef.current.querySelectorAll('.org-group .org-chart-scroll');
    groups.forEach(scroll=>{
      const wrapper = scroll.querySelector('.org-center-wrapper');
      if(!wrapper) return;
      const overflow = scroll.scrollWidth > scroll.clientWidth + 2;
      wrapper.style.justifyContent = overflow ? 'flex-start' : 'center';
    });
  },[]);

  useEffect(()=>{
    if(showOrgChart){
      const id = requestAnimationFrame(adjustCentering);
      return ()=> cancelAnimationFrame(id);
    }
  },[orgChart, showOrgChart, adjustCentering, editingLayout, draggingId]);

  useEffect(()=>{
    function onResize(){ adjustCentering(); }
    window.addEventListener('resize', onResize);
    return ()=> window.removeEventListener('resize', onResize);
  },[adjustCentering]);

  useEffect(()=>{
    if(!editingLayout) return;
    const handleDragStart = e=>{
      const id=e.target?.getAttribute?.('data-node-id');
      if(id) setDraggingId(Number(id));
    };
    const handleDragEnd = ()=> setDraggingId(null);
    const root=chartRef.current;
    if(root){
      root.addEventListener('dragstart',handleDragStart);
      root.addEventListener('dragend',handleDragEnd);
    }
    return ()=>{
      if(root){
        root.removeEventListener('dragstart',handleDragStart);
        root.removeEventListener('dragend',handleDragEnd);
      }
    };
  },[editingLayout]);

  const handleGenerateChart=()=>{
    setLoading(true);
    setTimeout(()=>{ rebuild(); setLoading(false); },30);
  };

  const unsavedChanges = useMemo(
    ()=> JSON.stringify(manualParentOverrides)!==JSON.stringify(lastSavedOverrides),
    [manualParentOverrides,lastSavedOverrides]
  );

  const handleSaveLayout=()=>{
    const cleaned = pruneOverrides(manualParentOverrides);
    localStorage.setItem('orgChartManualOverrides', JSON.stringify(cleaned));
    setLastSavedOverrides(cleaned);
  };
  const handleCancelLayout=()=>{ setManualParentOverrides(lastSavedOverrides||{}); };
  const handleResetManual=()=>{ setManualParentOverrides({}); };

  const handleDownload=async()=>{
    if(!chartRef.current) return;
    const root=chartRef.current;
    const originalId=root.id;
    if(!root.id) root.id='org-chart-root';
    const scrollElems=Array.from(root.querySelectorAll('.org-chart-scroll'));
    const originals=scrollElems.map(el=>({
      el,
      overflow:el.style.overflow,
      width:el.style.width,
      height:el.style.height,
      maxWidth:el.style.maxWidth,
      maxHeight:el.style.maxHeight
    }));
    try{
      scrollElems.forEach(el=>{
        el.style.overflow='visible';
        const sw=el.scrollWidth;
        const sh=el.scrollHeight;
        if(sw>el.clientWidth) el.style.width=sw+'px';
        if(sh>el.clientHeight) el.style.height=sh+'px';
        el.style.maxWidth='unset';
        el.style.maxHeight='unset';
      });
      await new Promise(r=>requestAnimationFrame(r));
      const fullWidth=root.scrollWidth;
      const fullHeight=root.scrollHeight;
      const canvas=await html2canvas(root,{
        backgroundColor:'#ffffff',
        useCORS:true,
        allowTaint:true,
        foreignObjectRendering:true,
        logging:false,
        imageTimeout:0,
        width:fullWidth,
        height:fullHeight,
        scrollX:0,
        scrollY:0
      });
      const url=canvas.toDataURL('image/png');
      const a=document.createElement('a');
      a.download='org_chart.png';
      a.href=url;
      a.click();
    }catch{
      alert('Export failed. Try Print -> PDF.');
    }finally{
      originals.forEach(o=>{
        o.el.style.overflow=o.overflow;
        o.el.style.width=o.width;
        o.el.style.height=o.height;
        o.el.style.maxWidth=o.maxWidth;
        o.el.style.maxHeight=o.maxHeight;
      });
      if(!originalId) root.id='';
    }
  };

  const handlePrint=()=>{
    const afterPrint=()=>{ window.removeEventListener('afterprint',afterPrint); };
    window.addEventListener('afterprint',afterPrint);
    window.print();
  };

  return (
    <div
      id="org-chart-root"
      ref={chartRef}
      style={{
        overflowX:'auto',
        width:'100%',
        maxWidth:'98vw',
        border:'1px solid #e2e8f0',
        borderRadius:8,
        background:'#fff',
        position:'relative',
        padding:12,
        marginBottom:24
      }}
    >
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', flexWrap:'wrap', gap:8 }}>
        <h2 style={{ margin:0, color: 'var(--azure-dragon)' }}>Org Chart</h2>
        <div style={{ display:'flex', gap:8, flexWrap:'wrap', alignItems:'center' }}>
          <label style={{ display:'flex', alignItems:'center', gap:6, fontSize:14, color: 'var(--muted)' }}>
            <input
              type="checkbox"
              checked={showOrgChart}
              onChange={e=>{
                setShowOrgChart(e.target.checked);
                if(!e.target.checked){
                  setEditingLayout(false);
                  setDraggingId(null);
                }
              }}
            />
            Show Org Chart
          </label>
          <button
            onClick={()=>{
              if(!showOrgChart) return;
              if(editingLayout){
                setEditingLayout(false); setDraggingId(null);
              } else { setEditingLayout(true); }
            }}
            disabled={!showOrgChart}
            style={{
              background: !showOrgChart ? '#cbd5e1' : (editingLayout? '#334155':'#2563eb'),
              color:'#fff', border:'none', padding:'6px 14px',
              borderRadius:4, cursor: !showOrgChart ? 'not-allowed':'pointer', fontWeight:700
            }}
          >
            {editingLayout ? 'Finish Editing' : 'Edit Layout'}
          </button>
          <button
            onClick={handleSaveLayout}
            disabled={!showOrgChart || !unsavedChanges}
            style={{
              background: (!showOrgChart || !unsavedChanges)? '#94a3b8':'#059669',
              color:'#fff', border:'none', padding:'6px 14px',
              borderRadius:4, cursor: (!showOrgChart || !unsavedChanges)? 'not-allowed':'pointer', fontWeight:700
            }}
          >Save Layout</button>
          <button
            onClick={handleCancelLayout}
            disabled={!showOrgChart || !unsavedChanges}
            style={{
              background: (!showOrgChart || !unsavedChanges)? '#fde0c2':'#f97316',
              color: (!showOrgChart || !unsavedChanges)? '#9a9a9a':'#fff',
              border:'none', padding:'6px 14px', borderRadius:4,
              cursor: (!showOrgChart || !unsavedChanges)? 'not-allowed':'pointer', fontWeight:700
            }}
          >Cancel</button>
          <button
            onClick={handleResetManual}
            disabled={!showOrgChart || !Object.keys(manualParentOverrides||{}).length}
            style={{
              background: (!showOrgChart || !Object.keys(manualParentOverrides||{}).length)? '#f8d7da':'#b91c1c',
              color: (!showOrgChart || !Object.keys(manualParentOverrides||{}).length)? '#9a9a9a':'#fff',
              border:'none', padding:'6px 14px', borderRadius:4,
              cursor: (!showOrgChart || !Object.keys(manualParentOverrides||{}).length)? 'not-allowed':'pointer', fontWeight:700
            }}
          >Reset Manual</button>
          <button
            onClick={handleGenerateChart}
            disabled={!showOrgChart || loading}
            style={{
              background: (!showOrgChart)? '#cbd5e1' : '#4f46e5',
              color:'#fff', border:'none',
              padding:'6px 14px', borderRadius:4, cursor:(!showOrgChart||loading)?'not-allowed':'pointer', fontWeight:700
            }}
          >{loading ? 'Regenerating...' : 'Regenerate'}</button>
          {showOrgChart && orgChart.length>0 && (
            <>
              <button
                onClick={handleDownload}
                style={{
                  background:'#0d9488', color:'#fff', border:'none',
                  padding:'6px 14px', borderRadius:4, cursor:'pointer', fontWeight:700
                }}
              >Export PNG</button>
              <button
                onClick={handlePrint}
                style={{
                  background:'#eab308', color:'#1e293b', border:'none',
                  padding:'6px 14px', borderRadius:4, cursor:'pointer', fontWeight:700
                }}
              >Print / PDF</button>
            </>
          )}
        </div>
      </div>

      <div style={{ marginTop:10, marginBottom:10, display:'flex', alignItems:'center', gap:10, flexWrap:'wrap' }}>
        <label htmlFor="job-family-dropdown" style={{ fontWeight:700, color: 'var(--muted)' }}>Filter By Job Family</label>
        <select
            id="job-family-dropdown"
            value={selectedJobFamily}
            onChange={e=>onChangeJobFamily(e.target.value)}
            disabled={!showOrgChart}
            style={{
              padding:'6px 10px',
              borderRadius:4,
              border:'1px solid var(--neutral-border)',
              background: showOrgChart ? '#fff' : '#e5e7eb',
              cursor: showOrgChart ? 'pointer':'not-allowed'
            }}
        >
          {jobFamilyOptions.map(jf=> <option key={jf} value={jf}>{jf}</option>)}
        </select>

        <label htmlFor="organisation-dropdown" style={{ fontWeight:700, color: 'var(--muted)' }}>Organisation</label>
        <select
          id="organisation-dropdown"
          value={selectedOrganisation}
          onChange={e=>onChangeOrganisation(e.target.value)}
          disabled={!showOrgChart}
          style={{
            padding:'6px 10px',
            borderRadius:4,
            border:'1px solid var(--neutral-border)',
            background: showOrgChart ? '#fff' : '#e5e7eb',
            cursor: showOrgChart ? 'pointer':'not-allowed'
          }}
        >
          {organisationOptions.map(opt=> <option key={opt} value={opt}>{opt}</option>)}
        </select>

        <label htmlFor="country-dropdown" style={{ fontWeight:700, color: 'var(--muted)' }}>Country</label>
        <select
          id="country-dropdown"
          value={selectedCountry}
          onChange={e=>onChangeCountry(e.target.value)}
          disabled={!showOrgChart}
          style={{
            padding:'6px 10px',
            borderRadius:4,
            border:'1px solid var(--neutral-border)',
            background: showOrgChart ? '#fff' : '#e5e7eb',
            cursor: showOrgChart ? 'pointer':'not-allowed'
          }}
        >
          {countryOptions.map(opt=> <option key={opt} value={opt}>{opt}</option>)}
        </select>

        {showOrgChart && (
          <span style={{ fontSize:12, color:'#64748b' }}>
            {editingLayout ? 'Drag to re-parent (drop on Make Root to promote)' : 'Click Edit Layout to enable dragging.'}
          </span>
        )}
        {showOrgChart && unsavedChanges && <span style={{ fontSize:12, color:'#dc2626', fontWeight:600 }}>Unsaved changes</span>}
        {!showOrgChart && <span style={{ fontSize:12, color:'#64748b' }}>Org chart hidden (enable checkbox to view)</span>}
      </div>

      {showOrgChart && (
        <div style={{ marginTop:12 }}>
          {orgChart.length ? orgChart : <span style={{ color:'#64748b' }}>No org chart generated yet.</span>}
        </div>
      )}
      {!showOrgChart && (
        <div style={{ marginTop:12, fontSize:14, color:'#475569' }}>
          Org chart display is turned off.
        </div>
      )}
    </div>
  );
}

/* ========================= UPLOAD ========================= */
function CandidateUpload({ onUpload }) {
  const [file,setFile] = useState(null);
  const [uploading,setUploading] = useState(false);
  const [error,setError] = useState('');

  const first = (row, ...keys) => {
    for (const k of keys) {
      if (Object.prototype.hasOwnProperty.call(row, k) && row[k] != null && String(row[k]).trim() !== '') {
        return row[k];
      }
    }
    return undefined;
  };

  const mapRow = (row) => {
    return {
      type: first(row, 'type', 'Type', 'product', 'Product') || '',
      name: first(row, 'name', 'Name') || '',
      role: first(row, 'role', 'Role') || '',
      organisation: first(row, 'organisation', 'Organisation') || '',
      sector: first(row, 'sector', 'Sector') || '',
      job_family: first(row, 'job_family', 'Job Family') || '',
      role_tag: first(row, 'role_tag', 'Role Tag') || '',
      skillset: first(row, 'skillset', 'Skillset') || '',
      geographic: first(row, 'geographic', 'Geographic') || '',
      country: first(row, 'country', 'Country') || '',
      email: first(row, 'email', 'Email') || '',
      mobile: first(row, 'mobile', 'Mobile') || '',
      office: first(row, 'office', 'Office') || '',
      personal: first(row, 'personal', 'Personal') || '',
      seniority: first(row, 'seniority', 'Seniority') || '',
      sourcing_status: first(row, 'sourcing_status', 'Sourcing Status') || '',
      lskillset: first(row, 'lskillset', 'Unmatched Skillset') || '',
      tenure: first(row, 'tenure', 'Tenure', 'avg_tenure', 'AVG Tenure', 'Average Tenure') || '',
      pic: first(row, 'pic', 'Pic', 'picture', 'Picture', 'image', 'Image') || null,
      education: first(row, 'education', 'Education') || '',
      comment: first(row, 'comment', 'Comment', 'comments', 'Comments', 'note', 'Note', 'notes', 'Notes') || '',
      cv: first(row, 'cv', 'CV', 'resume', 'Resume') || ''
    };
  };

  const handleFileChange = e => { setFile(e.target.files[0]); setError(''); };

  const handleUpload=()=>{
    if(!file){ setError('Please select a CSV or Excel file.'); return; }
    setUploading(true);
    const ext=file.name.split('.').pop().toLowerCase();
    if(ext==='csv'){
      Papa.parse(file,{
        header:true,
        complete: async results=>{
          const candidates=results.data.filter(r=> r && (r.name||r.Name)).map(mapRow);
          try{
            const res=await fetch('http://localhost:4000/candidates/bulk',{
              method:'POST',
              headers:{'Content-Type':'application/json'},
              body: JSON.stringify({ candidates }),
              credentials:'include' // Important for cookie
            });
            if(!res.ok) throw new Error();
            setFile(null); setError(''); onUpload && onUpload();
          }catch{
            setError('Failed to upload candidates.');
          }finally{ setUploading(false); }
        },
        error:()=>{ setError('Failed to parse CSV file.'); setUploading(false); }
      });
    } else if (ext==='xlsx' || ext==='xls'){
      file.arrayBuffer().then(data=>{
        const wb=XLSX.read(data);
        const ws=wb.Sheets[wb.SheetNames[0]];
        const json=XLSX.utils.sheet_to_json(ws);
        const candidates=json.filter(r=> r && (r.name||r.Name)).map(mapRow);
        fetch('http://localhost:4000/candidates/bulk',{
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ candidates }),
          credentials:'include'
        })
        .then(res=>{
          if(!res.ok) throw new Error();
            setFile(null); setError(''); onUpload && onUpload();
        })
        .catch(()=> setError('Failed to upload candidates.'))
        .finally(()=> setUploading(false));
      }).catch(()=>{
        setError('Failed to parse Excel file.');
        setUploading(false);
      });
    } else {
      setError('Unsupported file type. Please select CSV, XLSX, or XLS.');
      setUploading(false);
    }
  };
  return (
    <div style={{ marginBottom:24, padding:12, border:'1px solid var(--neutral-border)', borderRadius:8, background:'#fff', boxShadow: 'var(--shadow)' }}>
      <h2 style={{color: 'var(--azure-dragon)'}}>Bulk Upload Candidates (CSV/XLSX/XLS)</h2>
      <input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileChange}/>
      <button
        onClick={handleUpload}
        disabled={uploading}
        style={{
          marginLeft:8,
          background:'var(--cool-blue)',
          color:'#fff',
          border: 'none',
          padding:'6px 14px',
          borderRadius:4,
          cursor:uploading?'not-allowed':'pointer'
        }}
      >{uploading?'Uploading...':'Upload'}</button>
      {error && <div style={{ color:'var(--danger)', marginTop:8 }}>{error}</div>}
      <div style={{ fontSize:12, marginTop:8, color: 'var(--argent)' }}>
        Columns supported: common candidate columns. Project_Title/Project Date variants are accepted but not required.
      </div>
    </div>
  );
}

/* ========================= MAIN APP ========================= */
export default function App() {
  const [user, setUser] = useState(null);
  const [checkingAuth, setCheckingAuth] = useState(true);

  // Candidates & main state
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [deleteError, setDeleteError] = useState('');
  const [type, setType] = useState('Console');
  const [page, setPage] = useState(1);
  const [editRows, setEditRows] = useState({});
  const [skillsetMapping, setSkillsetMapping] = useState(null);

  // org chart state
  const [manualParentOverrides, setManualParentOverrides] = useState({});
  const [lastSavedOverrides, setLastSavedOverrides] = useState({});
  const [editingLayout, setEditingLayout] = useState(false);
  const [showOrgChart, setShowOrgChart] = useState(false);

  // Tabs state
  const [activeTab, setActiveTab] = useState('list'); // 'list' or 'chart' or 'resume'

  // NEW: Resume tab state
  const [resumeCandidate, setResumeCandidate] = useState(null);
  
  // State for resume email updating
  const [resumeEmailList, setResumeEmailList] = useState([]);

  // State for email generation/verification in Resume Tab
  const [generatingEmails, setGeneratingEmails] = useState(false);
  const [verifyingEmail, setVerifyingEmail] = useState(false);
  const [verifyModalData, setVerifyModalData] = useState(null);
  const [verifyModalEmail, setVerifyModalEmail] = useState('');

  // State for calculating unmatched skills
  const [calculatingUnmatched, setCalculatingUnmatched] = useState(false);
  const [unmatchedCalculated, setUnmatchedCalculated] = useState({});  // Store by candidate ID

  // State for skillset management
  const [newSkillInput, setNewSkillInput] = useState('');

  // Token state - only Account Token and Tokens Left
  const [accountTokens, setAccountTokens] = useState(0);
  const [tokensLeft, setTokensLeft] = useState(0);

  // Status Management State
  const DEFAULT_STATUSES = ['Reviewing', 'Contacted', 'Unresponsive', 'Declined', 'Unavailable', 'Screened', 'Not Proceeding', 'Prospected'];
  const [statusOptions, setStatusOptions] = useState(DEFAULT_STATUSES);
  const [statusModalOpen, setStatusModalOpen] = useState(false);

  useEffect(() => {
    if (user && user.username) {
      const key = `sourcingStatuses_${user.username}`;
      const saved = localStorage.getItem(key);
      if (saved) {
        try {
          setStatusOptions(JSON.parse(saved));
        } catch (e) {
          console.error(e);
        }
      } else {
        setStatusOptions(DEFAULT_STATUSES);
      }
    }
  }, [user]);

  // Load persisted unmatched calculation state from localStorage
  useEffect(() => {
    try {
      const storedState = localStorage.getItem('unmatchedCalculated');
      if (storedState) {
        setUnmatchedCalculated(JSON.parse(storedState));
      }
    } catch (e) {
      console.error('Failed to load unmatched state:', e);
    }
  }, []);

  // Fetch account tokens from login table when user logs in
  useEffect(() => {
    if (user && user.username) {
      fetch('http://localhost:4000/user-tokens', { credentials: 'include' })
        .then(res => res.json())
        .then(data => {
          if (data.accountTokens !== undefined) {
            setAccountTokens(data.accountTokens);
          }
          if (data.tokensLeft !== undefined) {
            setTokensLeft(data.tokensLeft);
          }
        })
        .catch(err => console.error('Failed to fetch tokens:', err));
    }
  }, [user]);

  const handleAddStatus = (newStat) => {
    if (!user || !user.username) return;
    const updated = [...statusOptions, newStat];
    setStatusOptions(updated);
    localStorage.setItem(`sourcingStatuses_${user.username}`, JSON.stringify(updated));
  };

  const handleRemoveStatus = (stat) => {
    if (!user || !user.username) return;
    if(!window.confirm(`Remove status "${stat}"?`)) return;
    const updated = statusOptions.filter(s => s !== stat);
    setStatusOptions(updated);
    localStorage.setItem(`sourcingStatuses_${user.username}`, JSON.stringify(updated));
  };

  const [filters, setFilters] = useState({});
  const handleChangeFilter = (field, value) => {
    setFilters(prev => ({ ...prev, [field]: value }));
    setPage(1);
  };
  const clearAllFilters = () => {
    setFilters({});
    setPage(1);
  };

  // Check auth on mount
  useEffect(() => {
    fetch('http://localhost:4000/user/resolve', { credentials: 'include' })
      .then(res => res.json())
      .then(data => {
        if (data.ok) setUser(data);
      })
      .catch(() => {}) // ignore error, stay logged out
      .finally(() => setCheckingAuth(false));
  }, []);

  const handleLogout = () => {
    fetch('http://localhost:4000/logout', { method: 'POST', credentials: 'include' })
      .then(() => setUser(null));
  };

  // SSE & Autosave setup (only if logged in)
  const eventSourceRef = useRef(null);
  const pendingSavesRef = useRef(new Map()); // id -> timeout
  const reconnectTimeoutRef = useRef(null);

  useEffect(() => {
    if (!user) return;
    let mounted = true;
    let reconnectAttempts = 0;

    const connectSSE = () => {
      if (!mounted) return;

      try {
        // Use relative URL or environment-based URL
        // For production, use the same protocol/host without explicit port
        const sseUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
          ? `http://localhost:${API_PORT}/api/events`
          : `${window.location.protocol}//${window.location.host}/api/events`;

        const eventSource = new EventSource(sseUrl);
        eventSourceRef.current = eventSource;

        eventSource.addEventListener('connected', (e) => {
          console.log('[SSE] connected', e.data);
          reconnectAttempts = 0; // Reset on successful connection
        });

        eventSource.addEventListener('candidate_updated', (e) => {
          try {
            const updated = JSON.parse(e.data);
            if (!updated || updated.id == null) return;
            setCandidates(prev => {
              const exists = prev.some(c => String(c.id) === String(updated.id));
              if (!exists) return prev;
              return prev.map(c => (String(c.id) === String(updated.id) ? { ...c, ...updated } : c));
            });
            setEditRows(prev => ({ ...(prev || {}), [updated.id]: { ...(prev[updated.id] || {}), ...updated } }));
            // Update resume candidate in view if selected
            setResumeCandidate(prev => (prev && String(prev.id) === String(updated.id) ? { ...prev, ...updated } : prev));
          } catch (err) {
            console.warn('[SSE] Error parsing candidate_updated:', err);
          }
        });

        eventSource.addEventListener('candidates_changed', (e) => {
          try {
            const payload = JSON.parse(e.data);
            console.log('[SSE] candidates_changed:', payload);
            // simple reaction: refetch list
            fetchCandidates();
          } catch (err) {
            console.warn('[SSE] Error parsing candidates_changed:', err);
          }
        });

        eventSource.onerror = (err) => {
          console.warn('[SSE] connection error', err);
          eventSource.close();

          // Implement exponential backoff reconnection
          if (mounted && reconnectAttempts < SSE_MAX_RECONNECT_ATTEMPTS) {
            const delay = Math.min(SSE_RECONNECT_BASE_DELAY_MS * Math.pow(2, reconnectAttempts), SSE_RECONNECT_MAX_DELAY_MS);
            reconnectAttempts++;
            console.log(`[SSE] Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${SSE_MAX_RECONNECT_ATTEMPTS})`);
            reconnectTimeoutRef.current = setTimeout(connectSSE, delay);
          } else if (reconnectAttempts >= SSE_MAX_RECONNECT_ATTEMPTS) {
            console.error('[SSE] Max reconnection attempts reached');
          }
        };
      } catch (e) {
        console.warn('[SSE] connection failed', e && e.message);
      }
    };

    connectSSE();

    return () => {
      mounted = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      try {
        eventSourceRef.current?.close();
      } catch {}
    };
    // eslint-disable-next-line
  }, [user]);

  // debounced autosave function
  const saveCandidateDebounced = useCallback((id, partialData) => {
    const key = String(id);
    // Use original id string as key to preserve temp keys
    if (pendingSavesRef.current.has(key)) {
      clearTimeout(pendingSavesRef.current.get(key));
    }
    const timeout = setTimeout(async () => {
      pendingSavesRef.current.delete(key);
      try {
        const numId = Number(id);
        const isExisting = Number.isInteger(numId) && numId > 0;
        if (isExisting) {
          // existing row -> update
          const res = await fetch(`http://localhost:4000/candidates/${numId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(partialData),
            credentials: 'include'
          });
          if (!res.ok) {
            console.warn('autosave PUT failed', await res.text());
            return;
          }
          const updated = await res.json();
          setCandidates(prev => prev.map(c => String(c.id) === String(updated.id) ? { ...c, ...updated } : c));
          setEditRows(prev => ({ ...(prev||{}), [updated.id]: { ...(prev?.[updated.id]||{}), ...updated } }));
        } else {
          // no numeric id -> create new process row
          const res = await fetch(`http://localhost:4000/candidates`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(partialData),
            credentials: 'include'
          });
          if (!res.ok) {
            console.warn('autosave POST failed', await res.text());
            return;
          }
          const created = await res.json();
          // move any editRows from temp key to new numeric id
          setEditRows(prev => {
            const next = { ...(prev || {}) };
            if (next[created.id]) {
              next[created.id] = { ...(next[key] || {}), ...created };
              delete next[key];
            } else {
              next[created.id] = { ...(created || {}) };
            }
            return next;
          });
          // Refresh list to include newly created row (keeps ordering consistent)
          await fetchCandidates();
        }
      } catch (e) {
        console.warn('autosave failed', e && e.message);
      }
    }, 700);
    pendingSavesRef.current.set(key, timeout);
  }, []);

  useEffect(() => {
    let mounted = true;
    fetchSkillsetMapping().then(m => { if (mounted) setSkillsetMapping(m); });
    return () => { mounted = false; };
  },[]);

  const PER_PAGE = 10;
  const fetchCandidates = async ()=>{
    if (!user) return;
    setLoading(true);
    try{
      const res=await fetch('http://localhost:4000/candidates', { credentials: 'include' });
      const raw=await res.json();
      const candidatesList = Array.isArray(raw)?raw:[];
      setCandidates(candidatesList);
      setPage(1);
    }catch{
      setCandidates([]);
    }
    setLoading(false);
  };
  useEffect(()=>{ if(user) fetchCandidates(); },[user]);

  // Robust merging (from earlier)
  const mergedCandidates = useMemo(()=>{
    const map = new Map();
    (candidates || []).forEach((c, i) => {
      const hasId = c != null && (c.id !== undefined && c.id !== null && String(c.id).trim() !== '');
      const key = hasId ? `id:${String(c.id)}` : `tmp:${i}`;
      const existing = map.get(key) || {};
      const mergedRow = { ...existing, ...c };
      if (hasId) {
        const edits = editRows[c.id] || {};
        Object.assign(mergedRow, edits);
      }
      mergedRow.id = hasId ? c.id : (existing.id || `tmp_${i}`);
      map.set(key, mergedRow);
    });
    return Array.from(map.values());
  },[candidates, editRows]);

  const [selectedJobFamily, setSelectedJobFamily] = useState('All');
  const [selectedOrganisation, setSelectedOrganisation] = useState('All');
  const [selectedCountry, setSelectedCountry] = useState('All');

  // Org Chart Auto-Filter Effect
  useEffect(() => {
    if (resumeCandidate) {
        const org = resumeCandidate.organisation || resumeCandidate.company;
        if (org) {
            setSelectedOrganisation(org);
        } else {
            setSelectedOrganisation('All');
        }
    } else {
        setSelectedOrganisation('All');
    }
  }, [resumeCandidate]);

  const baseFilter = useCallback((c, { jf, org, country }) => {
    if (jf && jf !== 'All' && (c.job_family||'').toString().trim() !== jf) return false;
    if (org && org !== 'All' && (c.organisation||'').toString().trim() !== org) return false;
    if (country && country !== 'All' && (c.country||'').toString().trim() !== country) return false;
    return true;
  }, []);

  const jobFamilyOptions = useMemo(()=>{
    const s=new Set();
    mergedCandidates.forEach(c=>{
      if (!baseFilter(c, { jf: null, org: selectedOrganisation, country: selectedCountry })) return;
      const jf=(c.job_family||'').toString().trim();
      if(jf) s.add(jf);
    });
    const opts = ['All', ...Array.from(s).sort((a,b)=> a.localeCompare(b))];
    return opts;
  },[mergedCandidates, selectedOrganisation, selectedCountry, baseFilter]);

  const organisationOptions = useMemo(()=>{
    const s=new Set();
    mergedCandidates.forEach(c=>{
      if (!baseFilter(c, { jf: selectedJobFamily, org: null, country: selectedCountry })) return;
      const org=(c.organisation||'').toString().trim();
      if(org) s.add(org);
    });
    const opts = ['All', ...Array.from(s).sort((a,b)=> a.localeCompare(b))];
    return opts;
  },[mergedCandidates, selectedJobFamily, selectedCountry, baseFilter]);

  const countryOptions = useMemo(()=>{
    const s=new Set();
    mergedCandidates.forEach(c=>{
      if (!baseFilter(c, { jf: selectedJobFamily, org: selectedOrganisation, country: null })) return;
      const cc=(c.country||'').toString().trim();
      if(cc) s.add(cc);
    });
    const opts = ['All', ...Array.from(s).sort((a,b)=> a.localeCompare(b))];
    return opts;
  },[mergedCandidates, selectedJobFamily, selectedOrganisation, baseFilter]);

  useEffect(()=>{ if (!jobFamilyOptions.includes(selectedJobFamily)) setSelectedJobFamily('All'); }, [jobFamilyOptions, selectedJobFamily]);
  useEffect(()=>{ if (!organisationOptions.includes(selectedOrganisation)) setSelectedOrganisation('All'); }, [organisationOptions, selectedOrganisation]);
  useEffect(()=>{ if (!countryOptions.includes(selectedCountry)) setSelectedCountry('All'); }, [countryOptions, selectedCountry]);

  const intersectionFiltered = useMemo(()=>{
    return mergedCandidates.filter(c =>
      baseFilter(c, { jf: selectedJobFamily, org: selectedOrganisation, country: selectedCountry })
    );
  }, [mergedCandidates, selectedJobFamily, selectedOrganisation, selectedCountry, baseFilter]);

  const filteredCandidates = useMemo(()=>{
    const activeKeys = Object.entries(filters).filter(([,v])=> v && String(v).trim()!=='');
    if(!activeKeys.length) return intersectionFiltered;
    return intersectionFiltered.filter(c=>{
      return activeKeys.every(([k,v])=>{
        const cell = c[k];
        if(cell==null) return false;
        return String(cell).toLowerCase().includes(String(v).toLowerCase());
      });
    });
  },[intersectionFiltered, filters]);

  const totalPages = Math.max(1, Math.ceil((filteredCandidates||[]).length / PER_PAGE));
  const pagedCandidates = useMemo(()=> (filteredCandidates||[]).slice((page-1)*PER_PAGE, page*PER_PAGE), [filteredCandidates,page]);

  const orgChartCandidates = intersectionFiltered;

  const deleteCandidatesBulk = async (ids)=>{
    setDeleteError('');
    const numericIds=(ids||[]).map(x=>Number(x)).filter(Number.isInteger);
    if(!numericIds.length){
      setDeleteError('No valid numeric IDs selected to delete.');
      return;
    }
    try{
      const res=await fetch('http://localhost:4000/candidates/bulk-delete',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ ids:numericIds }),
        credentials:'include'
      });
      const payload=await res.json().catch(()=>({}));
      if(!res.ok){
        setDeleteError(payload?.error || 'Delete failed.');
        return;
      }
      if((payload?.deletedCount ?? 0)===0){
        setDeleteError('Nothing was deleted (IDs may not exist).');
        return;
      }
      await fetchCandidates();
    }catch{
      setDeleteError('Delete failed (network).');
    }
  };

  const saveCandidate = async (id,data)=>{
    // Accept either numeric existing id (update) or create new row when id can't be parsed to integer
    const numId = Number(id);
    const isExisting = Number.isInteger(numId) && numId > 0;
    try{
      if (isExisting) {
        const res=await fetch(`http://localhost:4000/candidates/${numId}`,{
          method:'PUT',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify(data),
          credentials:'include'
        });
        if(!res.ok) throw new Error();
        const updated = await res.json();
        setCandidates(prev=> prev.map(c=> String(c.id) === String(updated.id) ? { ...c, ...updated } : c));
        setEditRows(prev => ({ ...(prev || {}), [updated.id]: { ...(prev[updated.id] || {}), ...updated } }));
      } else {
        // Create new
        const res=await fetch('http://localhost:4000/candidates',{
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify(data),
          credentials:'include'
        });
        if(!res.ok) throw new Error();
        const created = await res.json();
        // move any editRows from temp key to new numeric id
        setEditRows(prev => {
          const next = { ...(prev||{}) };
          if (next[id]) {
            next[created.id] = { ...(next[id] || {}), ...created };
            delete next[id];
          } else {
            next[created.id] = { ...(created || {}) };
          }
          return next;
        });
        await fetchCandidates();
      }
    }catch{
      alert('Failed to save candidate.');
    }
  };

  // Handler for viewing profile
  const handleViewProfile = (candidate) => {
    const rawEmails = (candidate.email || '').split(/[;,]+/).map(s => s.trim()).filter(Boolean);
    // Initialize emails list with check state, and default confidence for existing emails (N/A)
    setResumeEmailList(rawEmails.map(e => ({ value: e, checked: false, confidence: 'Stored (N/A)' })));
    setResumeCandidate(candidate);
    setActiveTab('resume');
  };

  // Handler for generating emails for resume candidate
  const handleGenerateResumeEmails = async () => {
    if (!resumeCandidate) return;
    const { name, organisation, company, country, id } = resumeCandidate;
    const org = organisation || company;
    
    if (!name || !org) {
      alert('Name and Company are required to generate emails.');
      return;
    }

    setGeneratingEmails(true);

    try {
      const res = await fetch('http://localhost:4000/generate-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, company: org, country }),
        credentials: 'include'
      });
      if (!res.ok) throw new Error('Request failed');
      const data = await res.json();
      
      if (data.emails && Array.isArray(data.emails) && data.emails.length > 0) {
         // Merge unique generated emails into current list
         const currentEmails = resumeEmailList.map(item => item.value);
         const newEmails = data.emails.filter(email => !currentEmails.includes(email));
         
         if (newEmails.length > 0) {
            // Since backend returns ranked list (1, 2, 3...), we infer confidence
            // Index 0: High, 1: Medium, 2+: Low
            const newEntries = newEmails.map((e, idx) => {
               let conf = 'Low (~50%)';
               if (idx === 0) conf = 'High (~95%)';
               else if (idx === 1) conf = 'Medium (~75%)';
               return { value: e, checked: false, confidence: conf };
            });

            setResumeEmailList(prev => [...prev, ...newEntries]);
         } else {
            alert('No new emails were generated (duplicates found).');
         }
      } else if (data.error) {
         alert(data.error);
      } else {
         alert('No valid generated emails found.');
      }
    } catch (e) {
      console.error(e);
      alert('Failed to generate emails.');
    } finally {
      setGeneratingEmails(false);
    }
  };

  // Handler for verifying selected email in resume tab
  const handleVerifySelectedEmail = async () => {
    const selected = resumeEmailList.filter(item => item.checked);
    if (selected.length === 0) {
        alert('Please select an email to verify.');
        return;
    }
    
    // Updated Logic: Only allow 1 check at a time
    if (selected.length > 1) {
        alert('Please verify one email at a time.');
        return;
    }
    
    const emailToVerify = selected[0].value;
    setVerifyingEmail(true);
    setVerifyModalEmail(emailToVerify);
    setVerifyModalData(null);

    try {
      const res = await fetch('http://localhost:4000/verify-email-details', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: emailToVerify }),
        credentials: 'include'
      });
      if (!res.ok) throw new Error('Verification failed');
      const data = await res.json();
      setVerifyModalData(data);
    } catch (e) {
      alert('Email verification failed.');
    } finally {
      setVerifyingEmail(false);
    }
  };

  // Handler for calculating unmatched skills
  const handleCalculateUnmatched = async () => {
      if (!resumeCandidate || !resumeCandidate.id) return;
      setCalculatingUnmatched(true);
      try {
          const res = await fetch(`http://localhost:4000/candidates/${resumeCandidate.id}/calculate-unmatched`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              credentials: 'include'
          });
          if (!res.ok) {
             const data = await res.json();
             throw new Error(data.error || 'Failed to calculate');
          }
          const data = await res.json();
          // The backend returns { lskillset, fullUpdate }
          // We only want to trigger lskillset update as requested
          
          if (data.lskillset !== undefined) {
              const updates = { lskillset: data.lskillset };
              // Update local lists with only lskillset
               setCandidates(prev => prev.map(c => String(c.id) === String(resumeCandidate.id) ? { ...c, ...updates } : c));
               setResumeCandidate(prev => ({ ...prev, ...updates }));
               
               // Mark as calculated and store the result (green if empty/matched, was red before)
               setUnmatchedCalculated(prev => ({
                   ...prev,
                   [resumeCandidate.id]: true
               }));
               
               // Persist the calculated state in localStorage
               try {
                   const storedState = JSON.parse(localStorage.getItem('unmatchedCalculated') || '{}');
                   storedState[resumeCandidate.id] = true;
                   localStorage.setItem('unmatchedCalculated', JSON.stringify(storedState));
               } catch (e) {
                   console.error('Failed to persist unmatched state:', e);
               }
          }

      } catch (e) {
          alert("Error: " + e.message);
      } finally {
          setCalculatingUnmatched(false);
      }
  };

  // Handler for updating email from resume tab
  const handleUpdateResumeEmail = () => {
    if (!resumeCandidate) return;
    
    // Join all checked emails
    const newEmail = resumeEmailList
        .filter(item => item.checked)
        .map(item => item.value)
        .join(', ');

    if (!newEmail) {
        if(!window.confirm("No emails selected. This will clear the email field. Continue?")) return;
    }

    const id = resumeCandidate.id;

    // Update editRows state so table shows it immediately
    setEditRows(prev => {
      const prior = prev[id] || {};
      const original = (candidates && candidates.find(cc => String(cc.id) === String(id))) || {};
      const base = { ...original, ...prior };
      return { ...prev, [id]: { ...base, email: newEmail } };
    });

    // Also update candidate in main list if it exists there
    setCandidates(prev => prev.map(c => String(c.id) === String(id) ? { ...c, email: newEmail } : c));
    // Also update the resumeCandidate object itself so the UI doesn't stale
    setResumeCandidate(prev => ({ ...prev, email: newEmail }));

    // Trigger save to backend
    saveCandidateDebounced(id, { email: newEmail });

    alert('Email updated in candidate list.');
  };

  // Skillset management handlers
  const handleRemoveSkill = (skillToRemove) => {
    if (!resumeCandidate) return;
    
    const currentSkills = resumeCandidate.skillset ? String(resumeCandidate.skillset).split(/[;,|]+/).map(s => s.trim()).filter(Boolean) : [];
    const updatedSkills = currentSkills.filter(s => s !== skillToRemove);
    const newSkillset = updatedSkills.join(', ');
    
    const id = resumeCandidate.id;
    
    // Update state
    setCandidates(prev => prev.map(c => String(c.id) === String(id) ? { ...c, skillset: newSkillset } : c));
    setResumeCandidate(prev => ({ ...prev, skillset: newSkillset }));
    
    // Save to backend
    saveCandidateDebounced(id, { skillset: newSkillset });
  };

  const handleAddSkill = (newSkill) => {
    if (!resumeCandidate || !newSkill || !newSkill.trim()) return;
    
    const currentSkills = resumeCandidate.skillset ? String(resumeCandidate.skillset).split(/[;,|]+/).map(s => s.trim()).filter(Boolean) : [];
    
    // Check if skill already exists
    if (currentSkills.some(s => s.toLowerCase() === newSkill.trim().toLowerCase())) {
      alert(`The skill "${newSkill.trim()}" already exists.`);
      return;
    }
    
    const updatedSkills = [...currentSkills, newSkill.trim()];
    const newSkillset = updatedSkills.join(', ');
    
    const id = resumeCandidate.id;
    
    // Update state
    setCandidates(prev => prev.map(c => String(c.id) === String(id) ? { ...c, skillset: newSkillset } : c));
    setResumeCandidate(prev => ({ ...prev, skillset: newSkillset }));
    
    // Save to backend
    saveCandidateDebounced(id, { skillset: newSkillset });
  };

  const handleResumeEmailCheck = (idx) => {
    setResumeEmailList(prev => prev.map((item, i) => i === idx ? { ...item, checked: !item.checked } : item));
  };

  if (checkingAuth) return <div style={{padding:20}}>Loading...</div>;
  if (!user) return <LoginScreen onLoginSuccess={setUser} />;

  // --- Header Helpers for Banner ---
  const getDisplayName = () => {
    if (user.full_name && user.full_name.trim()) return user.full_name;
    return user.username;
  };
  const getInitials = () => {
    const name = getDisplayName();
    return (name || 'U').split(/\s+/).slice(0, 2).map(p => p[0].toUpperCase()).join('');
  };

  return (
    <div style={{
      width: '100%',
      minHeight: '100vh',
      margin: 0,
      padding: 24,
      boxSizing: 'border-box',
      background:'var(--bg)',
      color: 'var(--muted)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'stretch',
      justifyContent: 'flex-start',
      overflowX: 'hidden' // Ensure no horizontal body scroll
    }}>
      {/* Updated Session Banner UI */}
      <div style={{
        width: '100%',
        margin: '0 0 24px 0',
        display:'flex',
        alignItems:'center',
        justifyContent:'space-between',
        padding:'12px 18px',
        borderBottom:'1px solid var(--neutral-border)',
        background:'var(--card)',
        boxShadow:'var(--shadow)',
        borderRadius: 0
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:14 }}>
          <div style={{
            width:52, height:52, borderRadius:14, background:'var(--azure-dragon)',
            display:'flex', alignItems:'center', justifyContent:'center',
            color:'#fff', fontSize:17, fontWeight:700,
            boxShadow:'0 4px 10px -4px rgba(7,54,121,.35)', letterSpacing:'.5px'
          }}>
            {getInitials()}
          </div>
          <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
            <div style={{ fontWeight:700, fontSize:16, color:'var(--black-beauty)' }}>
              Welcome, {getDisplayName()}
            </div>
            <div style={{ fontSize:12.5, color:'var(--cool-blue)' }}>
              @{user.username}
            </div>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <h1 style={{ fontSize: 24, margin: 0, fontWeight: 700, color: 'var(--muted)', marginRight: 12, display: 'none' /* hidden on small, maybe show */ }}>
             Candidate Management System
          </h1>
          <button
            onClick={handleLogout}
            style={{
              background:'var(--bg)', color:'var(--azure-dragon)',
              border:'1px solid var(--cool-blue)', borderRadius:10,
              padding:'8px 18px', fontWeight:700, fontSize:13,
              cursor:'pointer', boxShadow:'0 2px 6px -4px rgba(11,98,192,.25)'
            }}
          >Logout</button>
        </div>
      </div>
      
      {/* Token Metrics UI - Account Token and Tokens Left only */}
      <div style={{
        width: '100%',
        margin: '0 0 24px 0',
        padding: '12px 18px',
        background: '#f8f9fa',
        border: '1px solid #e5e7eb',
        borderRadius: 8,
        display: 'flex',
        gap: 16,
        alignItems: 'center',
        flexWrap: 'wrap'
      }}>
        <div style={{ 
          display: 'inline-flex', 
          alignItems: 'center', 
          gap: 6,
          padding: '6px 12px',
          background: '#dbeafe',
          border: '1px solid #3b82f6',
          borderRadius: 6,
          fontSize: 13
        }}>
          <strong style={{ color: '#1e40af' }}>Account Tokens:</strong>
          <span style={{ fontWeight: 600, color: '#1e40af' }}>{accountTokens}</span>
        </div>
        
        <div style={{ 
          display: 'inline-flex', 
          alignItems: 'center', 
          gap: 6,
          padding: '6px 12px',
          background: '#fff',
          border: '1px solid #d0d7de',
          borderRadius: 6,
          fontSize: 13
        }}>
          <strong style={{ color: '#0969da' }}>Tokens Left:</strong>
          <span style={{ fontWeight: 600 }}>{tokensLeft}</span>
        </div>
      </div>
      
      {/* Title only visible below banner now */}
      <h1 style={{ fontSize:32, margin:'0 0 24px', fontWeight:900, color:'var(--azure-dragon)', letterSpacing: '1px' }}>Candidate Management System</h1>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20, borderBottom: '1px solid var(--neutral-border)' }}>
        <button 
          onClick={() => setActiveTab('list')}
          className={activeTab === 'list' ? 'tab-active' : 'tab-inactive'}
          style={{
            padding: '10px 20px',
            borderRadius: '8px 8px 0 0',
            fontWeight: 700,
            cursor: 'pointer',
            marginBottom: -1,
            fontFamily: 'Orbitron, sans-serif'
          }}
        >
          Candidate List
        </button>
        <button 
          onClick={() => setActiveTab('resume')}
          className={activeTab === 'resume' ? 'tab-active' : 'tab-inactive'}
          style={{
            padding: '10px 20px',
            borderRadius: '8px 8px 0 0',
            fontWeight: 700,
            cursor: 'pointer',
            marginBottom: -1,
            fontFamily: 'Orbitron, sans-serif'
          }}
        >
          Resume
        </button>
        <button 
          onClick={() => {
            setActiveTab('chart');
            if (!showOrgChart) setShowOrgChart(true);
          }}
          className={activeTab === 'chart' ? 'tab-active' : 'tab-inactive'}
          style={{
            padding: '10px 20px',
            borderRadius: '8px 8px 0 0',
            fontWeight: 700,
            cursor: 'pointer',
            marginBottom: -1,
            fontFamily: 'Orbitron, sans-serif'
          }}
        >
          Org Chart
        </button>
      </div>

      <div style={{ display: activeTab === 'list' ? 'block' : 'none' }}>
        <CandidateUpload onUpload={fetchCandidates} />

        <div style={{ marginTop:32 }}>
          <h2 style={{ fontSize:24, fontWeight:700, margin:'0 0 16px', color:'var(--azure-dragon)' }}>Candidate List</h2>
          {loading
            ? <p>Loading candidates...</p>
            : <CandidatesTable
                candidates={pagedCandidates}
                onDelete={deleteCandidatesBulk}
                deleteError={deleteError}
                onSave={saveCandidate}
                onAutoSave={saveCandidateDebounced}
                type={type}
                page={page}
                setPage={setPage}
                totalPages={totalPages}
                editRows={editRows}
                setEditRows={setEditRows}
                skillsetMapping={skillsetMapping}
                filters={filters}
                onChangeFilter={handleChangeFilter}
                onClearAllFilters={clearAllFilters}
                onViewProfile={handleViewProfile}
                statusOptions={statusOptions}
                onOpenStatusModal={() => setStatusModalOpen(true)}
                allCandidates={filteredCandidates}
              />
          }
        </div>
      </div>

      <div style={{ display: activeTab === 'resume' ? 'block' : 'none' }}>
        <div className="app-card" style={{ padding: 24, minHeight: '60vh' }}>
            {!resumeCandidate ? (
                <div style={{ textAlign: 'center', color: 'var(--argent)', padding: 40 }}>
                    <h3>No Candidate Selected</h3>
                    <p>Please go to the <b>Candidate List</b> tab and click the <b>Profile</b> button on a candidate to view their resume.</p>
                </div>
            ) : (
                <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, borderBottom: '1px solid var(--neutral-border)', paddingBottom: 16 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                            {/* Candidate Image */}
                            {resumeCandidate.pic && typeof resumeCandidate.pic === 'string' ? (
                                <>
                                    <img 
                                        src={resumeCandidate.pic.startsWith('data:') || resumeCandidate.pic.startsWith('http') 
                                            ? resumeCandidate.pic 
                                            : `data:image/jpeg;base64,${resumeCandidate.pic}`}
                                        alt={resumeCandidate.name || 'Candidate'}
                                        style={{
                                            width: 60,
                                            height: 60,
                                            borderRadius: '50%',
                                            objectFit: 'cover',
                                            border: '2px solid #e5e7eb',
                                            display: 'block'
                                        }}
                                        onError={(e) => {
                                            // Hide image and show placeholder
                                            const imgElement = e.target;
                                            const placeholder = imgElement.nextElementSibling;
                                            if (imgElement && placeholder) {
                                                imgElement.style.display = 'none';
                                                placeholder.style.display = 'flex';
                                            }
                                        }}
                                    />
                                    {/* Placeholder for failed image load (hidden by default) */}
                                    <div style={{
                                        width: 60,
                                        height: 60,
                                        borderRadius: '50%',
                                        background: '#f3f4f6',
                                        display: 'none',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        color: '#9ca3af',
                                        fontWeight: 600,
                                        fontSize: 24,
                                        border: '2px solid #e5e7eb'
                                    }}>
                                        {(resumeCandidate.name || '?').charAt(0).toUpperCase()}
                                    </div>
                                </>
                            ) : (
                                // Placeholder for missing image
                                <div style={{
                                    width: 60,
                                    height: 60,
                                    borderRadius: '50%',
                                    background: '#f3f4f6',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    color: '#9ca3af',
                                    fontWeight: 600,
                                    fontSize: 24,
                                    border: '2px solid #e5e7eb'
                                }}>
                                    {(resumeCandidate.name || '?').charAt(0).toUpperCase()}
                                </div>
                            )}
                            <div>
                                <h2 style={{ margin: 0, fontSize: 24, fontWeight: 900, color: 'var(--azure-dragon)' }}>{resumeCandidate.name}</h2>
                                <div style={{ color: 'var(--muted)', fontSize: 14, marginTop: 4, fontWeight: 700 }}>
                                    {resumeCandidate.role || resumeCandidate.jobtitle} at {resumeCandidate.organisation || resumeCandidate.company}
                                </div>
                            </div>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                           {/* LinkedIn Button */}
                           {resumeCandidate.linkedinurl && (
                             <a 
                               href={resumeCandidate.linkedinurl.startsWith('http') ? resumeCandidate.linkedinurl : `https://${resumeCandidate.linkedinurl}`}
                               target="_blank"
                               rel="noopener noreferrer"
                               style={{
                                 background: '#0a66c2',
                                 color: '#fff',
                                 textDecoration: 'none',
                                 padding: '8px 16px',
                                 borderRadius: 6,
                                 fontSize: 13,
                                 fontWeight: 700,
                                 display: 'flex', 
                                 alignItems: 'center', 
                                 gap: 6,
                                 fontFamily: 'Orbitron, sans-serif'
                               }}
                             >
                               <span>LinkedIn Profile</span>
                             </a>
                           )}

                           {/* Resume Button */}
                           <button 
                                onClick={() => {
                                    if(resumeCandidate.linkedinurl) {
                                        window.open('http://localhost:4000/process/download_cv?linkedin=' + encodeURIComponent(resumeCandidate.linkedinurl), '_blank');
                                    } else if(resumeCandidate.cv) {
                                        // Fallback if no linkedinurl but CV blob/path exists somehow
                                        // (e.g. from /candidates/:id/cv)
                                        if (typeof resumeCandidate.cv === 'string' && resumeCandidate.cv.startsWith('http')) {
                                            window.open(resumeCandidate.cv, '_blank');
                                        } else {
                                            window.open(`http://localhost:4000/candidates/${resumeCandidate.id}/cv`, '_blank');
                                        }
                                    } else {
                                        alert('No CV available for this candidate.');
                                    }
                                }} 
                                className="btn-secondary" 
                                style={{ padding: '8px 16px', display: 'flex', alignItems: 'center', gap: 6 }}
                           >
                              <span style={{ fontSize: 14 }}>📄</span> Resume
                           </button>
                        </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
                        {/* LEFT COLUMN: Professional Details + AVG Tenure + Location + Mobile + Office */}
                        <div style={{ padding: 16, background: '#f8fafc', borderRadius: 8, border: '1px solid var(--neutral-border)' }}>
                            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--argent)', marginBottom: 12, textTransform: 'uppercase' }}>Professional Details</div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                                <div>
                                    <label style={{ display: 'block', fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Seniority</label>
                                    <div style={{ fontSize: 14 }}>{resumeCandidate.seniority || '—'}</div>
                                </div>
                                <div>
                                    <label style={{ display: 'block', fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Total Experience</label>
                                    {/* Switched to use .exp */}
                                    <div style={{ fontSize: 14 }}>{resumeCandidate.exp ? `${resumeCandidate.exp} Years` : '—'}</div>
                                </div>
                                <div>
                                    <label style={{ display: 'block', fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Job Family</label>
                                    <div style={{ fontSize: 14 }}>{resumeCandidate.job_family || '—'}</div>
                                </div>
                                <div>
                                    <label style={{ display: 'block', fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Sector</label>
                                    <div style={{ fontSize: 14 }}>{resumeCandidate.sector || '—'}</div>
                                </div>
                                
                                {/* AVG Tenure field added above Location */}
                                <div>
                                    <label style={{ display: 'block', fontSize: 13, fontWeight: 700, marginBottom: 4 }}>AVG Tenure</label>
                                    <div style={{ fontSize: 14, color: 'var(--muted)' }}>
                                        {resumeCandidate.tenure ? `${resumeCandidate.tenure} Years` : '—'}
                                    </div>
                                </div>
                                
                                {/* Location field */}
                                <div>
                                    <label style={{ display: 'block', fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Location</label>
                                    <div style={{ fontSize: 14, color: 'var(--muted)' }}>
                                        {[resumeCandidate.city, resumeCandidate.country].filter(Boolean).join(', ') || '—'}
                                    </div>
                                </div>
                                
                                {/* Mobile field moved below Location */}
                                <div>
                                    <label style={{ display: 'block', fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Mobile</label>
                                    <div style={{ fontSize: 14, color: 'var(--muted)' }}>{resumeCandidate.mobile || '—'}</div>
                                </div>
                                
                                {/* Office field added below Mobile */}
                                <div>
                                    <label style={{ display: 'block', fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Office</label>
                                    <div style={{ fontSize: 14, color: 'var(--muted)' }}>{resumeCandidate.office || '—'}</div>
                                </div>
                            </div>
                        </div>

                        {/* RIGHT SIDE: Contact Information and Comment split vertically */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                            {/* Contact Information */}
                            <div style={{ padding: 16, background: '#f8fafc', borderRadius: 8, border: '1px solid var(--neutral-border)', flex: 1 }}>
                                <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--argent)', marginBottom: 12, textTransform: 'uppercase' }}>Contact Information</div>
                                
                                <div style={{ marginBottom: 16 }}>
                                    <label style={{ display: 'block', fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Email</label>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                        {resumeEmailList.map((item, idx) => {
                                            // Determine Color Style based on Confidence text
                                            let badgeStyle = { bg: '#f1f5f9', color: '#64748b', border: '#e2e8f0' };
                                            if (item.confidence && item.confidence.includes('High')) badgeStyle = { bg: '#dcfce7', color: '#15803d', border: '#bbf7d0' };
                                            else if (item.confidence && item.confidence.includes('Medium')) badgeStyle = { bg: '#fef9c3', color: '#a16207', border: '#fde047' };
                                            else if (item.confidence && item.confidence.includes('Low')) badgeStyle = { bg: '#fee2e2', color: '#b91c1c', border: '#fecaca' };

                                            return (
                                                <div key={idx} style={{ 
                                                    display: 'flex', alignItems: 'center', gap: 8, 
                                                    background: '#fff', border: '1px solid var(--neutral-border)', padding: '6px 10px', borderRadius: 6, fontSize: 13
                                                }}>
                                                    <input 
                                                        type="checkbox" 
                                                        checked={item.checked} 
                                                        onChange={() => handleResumeEmailCheck(idx)}
                                                    />
                                                    <span style={{ flex: 1, fontFamily: 'monospace' }}>{item.value}</span>
                                                    <span style={{ 
                                                        fontSize: 10, fontWeight: 700, 
                                                        backgroundColor: badgeStyle.bg, color: badgeStyle.color, border: `1px solid ${badgeStyle.border}`,
                                                        padding: '2px 6px', borderRadius: 4, textTransform: 'uppercase'
                                                    }}>
                                                        {item.confidence}
                                                    </span>
                                                </div>
                                            );
                                        })}
                                        {resumeEmailList.length === 0 && <span style={{ color: 'var(--argent)', fontSize: 13 }}>No emails found.</span>}
                                    </div>
                                    <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                                        <button 
                                            onClick={handleGenerateResumeEmails} 
                                            disabled={generatingEmails}
                                            className="btn-primary"
                                            style={{ fontSize: 12, padding: '6px 12px' }}
                                        >
                                            {generatingEmails ? 'Generating...' : 'Generate Emails'}
                                        </button>
                                        <button 
                                            onClick={handleVerifySelectedEmail} 
                                            disabled={verifyingEmail || resumeEmailList.filter(i=>i.checked).length !== 1}
                                            className="btn-secondary"
                                            style={{ fontSize: 12, padding: '6px 12px' }}
                                        >
                                            {verifyingEmail ? 'Verifying...' : 'Verify Selected'}
                                        </button>
                                        <button 
                                            onClick={handleUpdateResumeEmail}
                                            className="btn-secondary"
                                            style={{ fontSize: 12, padding: '6px 12px', marginLeft: 'auto' }}
                                        >
                                            Update & Save
                                        </button>
                                    </div>
                                </div>
                            </div>
                            
                            {/* Comment Textbox */}
                            <div style={{ padding: 16, background: '#f8fafc', borderRadius: 8, border: '1px solid var(--neutral-border)', flex: 1 }}>
                                <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--argent)', marginBottom: 12, textTransform: 'uppercase' }}>Comment</div>
                                <textarea
                                    value={resumeCandidate.comment || ''}
                                    onChange={(e) => {
                                        const newComment = e.target.value;
                                        setResumeCandidate(prev => ({ ...prev, comment: newComment }));
                                        // Auto-save to database
                                        saveCandidateDebounced(resumeCandidate.id, { comment: newComment });
                                    }}
                                    placeholder="Add notes or comments about this candidate..."
                                    style={{
                                        width: '100%',
                                        minHeight: 120,
                                        padding: 12,
                                        border: '1px solid var(--neutral-border)',
                                        borderRadius: 6,
                                        fontSize: 13,
                                        fontFamily: 'inherit',
                                        resize: 'vertical',
                                        boxSizing: 'border-box'
                                    }}
                                />
                            </div>
                        </div>
                    </div>

                    <div style={{ marginBottom: 24 }}>
                        <h3 style={{ fontSize: 16, fontWeight: 700, borderBottom: '2px solid var(--neutral-border)', paddingBottom: 8, marginBottom: 12, color: 'var(--black-beauty)' }}>Skillset</h3>
                        <div style={{ padding: 12, background: '#f8fafc', borderRadius: 8, border: '1px solid var(--neutral-border)', minHeight: 60, lineHeight: '1.6' }}>
                            {resumeCandidate.skillset ? (
                                String(resumeCandidate.skillset).split(/[;,|]+/).map((skill, i) => {
                                    const s = skill.trim();
                                    if(!s) return null;
                                    return (
                                        <span key={i} className="skill-bubble" style={{ position: 'relative', paddingRight: 28 }}>
                                            {s}
                                            <button
                                                onClick={() => handleRemoveSkill(s)}
                                                style={{
                                                    position: 'absolute',
                                                    right: 6,
                                                    top: '50%',
                                                    transform: 'translateY(-50%)',
                                                    background: 'transparent',
                                                    border: 'none',
                                                    color: '#b91c1c',
                                                    cursor: 'pointer',
                                                    fontSize: 16,
                                                    fontWeight: 'bold',
                                                    padding: 0,
                                                    width: 16,
                                                    height: 16,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    lineHeight: 1
                                                }}
                                                title="Remove skill"
                                            >
                                                ×
                                            </button>
                                        </span>
                                    );
                                })
                            ) : <span style={{ color: 'var(--argent)' }}>No skills listed.</span>}
                            
                            {/* Add new skill input */}
                            <div style={{ marginTop: 12, display: 'flex', gap: 8, alignItems: 'center' }}>
                                <input
                                    type="text"
                                    value={newSkillInput}
                                    onChange={(e) => setNewSkillInput(e.target.value)}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter') {
                                            e.preventDefault();
                                            handleAddSkill(newSkillInput);
                                            setNewSkillInput('');
                                        }
                                    }}
                                    placeholder="Add new skill..."
                                    style={{
                                        flex: 1,
                                        padding: '6px 12px',
                                        border: '1px solid var(--neutral-border)',
                                        borderRadius: 6,
                                        fontSize: 13
                                    }}
                                />
                                <button
                                    onClick={() => {
                                        handleAddSkill(newSkillInput);
                                        setNewSkillInput('');
                                    }}
                                    className="btn-primary"
                                    style={{ fontSize: 12, padding: '6px 12px' }}
                                >
                                    Add Skill
                                </button>
                            </div>
                        </div>
                    </div>

                    <div style={{ marginBottom: 24 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '2px solid var(--neutral-border)', paddingBottom: 8, marginBottom: 12 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0, color: 'var(--black-beauty)' }}>Unmatched Skillset</h3>
                                {/* Status indicator - red before calculation, green after */}
                                <div style={{
                                    width: 10,
                                    height: 10,
                                    borderRadius: '50%',
                                    backgroundColor: unmatchedCalculated[resumeCandidate?.id] ? '#10b981' : '#ef4444'
                                }} title={unmatchedCalculated[resumeCandidate?.id] ? 'Calculated' : 'Not calculated'} />
                            </div>
                            <button 
                                onClick={handleCalculateUnmatched}
                                disabled={calculatingUnmatched}
                                className="btn-primary"
                                style={{ fontSize: 12, padding: '4px 12px' }}
                            >
                                {calculatingUnmatched ? 'Calculating...' : 'Calculate Unmatched'}
                            </button>
                        </div>
                        <div style={{ 
                            padding: 12, 
                            background: unmatchedCalculated[resumeCandidate?.id] && !resumeCandidate.lskillset ? '#f0fdf4' : '#fff5f5', 
                            borderRadius: 8, 
                            border: `1px solid ${unmatchedCalculated[resumeCandidate?.id] && !resumeCandidate.lskillset ? '#bbf7d0' : '#fed7d7'}`, 
                            minHeight: 60, 
                            lineHeight: '1.6' 
                        }}>
                             {unmatchedCalculated[resumeCandidate?.id] && !resumeCandidate.lskillset ? (
                                // Show "All skillsets are matched" message when calculated and no unmatched skills
                                <span style={{ color: '#15803d', fontSize: 13, fontWeight: 600 }}>All skillsets are matched.</span>
                             ) : resumeCandidate.lskillset ? (
                                // Show unmatched skills
                                String(resumeCandidate.lskillset)
                                    .replace(/Here are the skills present in the JD Skillset but missing or unmatched in the Candidate Skillset[:\s]*/i, '')
                                    .replace(/[\[\]"']/g, '') // Strips brackets and quotes
                                    .split(/[;,|]+/)
                                    .map((skill, i) => {
                                        const s = skill.trim();
                                        if(!s) return null;
                                        return (
                                            <span key={i} className="skill-bubble unmatched">
                                                {s}
                                            </span>
                                        );
                                })
                            ) : <span style={{ color: '#b91c1c', fontSize: 13 }}>Click calculate to compare against job requirements.</span>}
                        </div>
                    </div>

                    <div style={{ marginBottom: 24 }}>
                        <h3 style={{ fontSize: 16, fontWeight: 700, borderBottom: '2px solid var(--neutral-border)', paddingBottom: 8, marginBottom: 12, color: 'var(--black-beauty)' }}>Experience</h3>
                        <div style={{ whiteSpace: 'pre-wrap', fontSize: 14, lineHeight: 1.6, color: 'var(--muted)', background: '#fff', padding: 16, border: '1px solid var(--neutral-border)', borderRadius: 8 }}>
                            {resumeCandidate.experience || 'No experience details available.'}
                        </div>
                    </div>

                    {/* Education Section */}
                    {resumeCandidate.education && (
                        <div style={{ marginBottom: 24 }}>
                            <h3 style={{ fontSize: 16, fontWeight: 700, borderBottom: '2px solid var(--neutral-border)', paddingBottom: 8, marginBottom: 12, color: 'var(--black-beauty)' }}>Education</h3>
                            <div style={{ whiteSpace: 'pre-wrap', fontSize: 14, lineHeight: 1.6, color: 'var(--muted)', background: '#fff', padding: 16, border: '1px solid var(--neutral-border)', borderRadius: 8 }}>
                                {resumeCandidate.education}
                            </div>
                        </div>
                    )}

                    {/* Recruiter-Style Assessment Display */}
                    {resumeCandidate.rating && (
                        <div style={{ marginBottom: 24 }}>
                            <h3 style={{ fontSize: 16, fontWeight: 700, borderBottom: '2px solid var(--neutral-border)', paddingBottom: 8, marginBottom: 12, color: 'var(--black-beauty)' }}>Candidate Assessment</h3>
                            <div style={{ padding: 16, background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)', border: '1px solid #e5e7eb', borderRadius: 8, boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
                                {resumeCandidate.rating && typeof resumeCandidate.rating === 'object' && resumeCandidate.rating.assessment_level ? (
                                    // Structured assessment with recruiter-friendly format
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                                        {/* Assessment Header with Score */}
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: 12, borderBottom: '2px solid #e5e7eb' }}>
                                            <div>
                                                <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
                                                    Assessment Level
                                                </div>
                                                <div className="assessment-header" style={{ fontSize: 18, fontWeight: 700, color: '#1f2937' }}>
                                                    {resumeCandidate.rating.assessment_level}
                                                </div>
                                            </div>
                                            <div style={{ textAlign: 'right' }}>
                                                <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
                                                    Overall Score
                                                </div>
                                                <div style={{ fontSize: 24, fontWeight: 900, color: '#0ea5e9' }}>
                                                    {resumeCandidate.rating.total_score || 'N/A'}
                                                </div>
                                            </div>
                                        </div>
                                        
                                        {/* Star Rating */}
                                        {resumeCandidate.rating.stars && (
                                            <div>
                                                <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 6 }}>
                                                    Rating
                                                </div>
                                                <div className="assessment-stars" style={{ fontSize: 20 }}>
                                                    {resumeCandidate.rating.stars}
                                                </div>
                                            </div>
                                        )}
                                        
                                        {/* Overall Comment/Summary */}
                                        {resumeCandidate.rating.overall_comment && (
                                            <div>
                                                <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 6 }}>
                                                    Executive Summary
                                                </div>
                                                <div style={{ padding: 12, background: '#eff6ff', borderLeft: '4px solid #3b82f6', fontSize: 13, lineHeight: 1.6, color: '#1e40af', borderRadius: 4 }}>
                                                    {resumeCandidate.rating.overall_comment}
                                                </div>
                                            </div>
                                        )}
                                        
                                        {/* Detailed Comments */}
                                        {resumeCandidate.rating.comments && (
                                            <div>
                                                <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 6 }}>
                                                    Recruiter Notes
                                                </div>
                                                <div className="assessment-comments" style={{ fontSize: 13, lineHeight: 1.6, color: '#374151', padding: 12, background: '#f9fafb', borderRadius: 4, border: '1px solid #e5e7eb' }}>
                                                    {resumeCandidate.rating.comments}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                ) : resumeCandidate.rating ? (
                                    // Simple text rating with improved formatting
                                    <div>
                                        <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8 }}>
                                            Assessment Notes
                                        </div>
                                        <div style={{ fontSize: 14, lineHeight: 1.8, color: '#374151' }}>
                                            {String(resumeCandidate.rating).split('\n').map((para, idx) => (
                                                <p key={idx} style={{ marginBottom: 12, marginTop: 0, paddingLeft: 12, borderLeft: '3px solid #e5e7eb' }}>
                                                    {para}
                                                </p>
                                            ))}
                                        </div>
                                    </div>
                                ) : null}
                            </div>
                        </div>
                    )}

                </div>
            )}
        </div>
      </div>

      <div style={{ display: activeTab === 'chart' ? 'block' : 'none' }}>
        <OrgChartDisplay
          candidates={orgChartCandidates}
          jobFamilyOptions={jobFamilyOptions}
          selectedJobFamily={selectedJobFamily}
          onChangeJobFamily={setSelectedJobFamily}
          manualParentOverrides={manualParentOverrides}
          setManualParentOverrides={setManualParentOverrides}
          editingLayout={editingLayout}
          setEditingLayout={setEditingLayout}
          lastSavedOverrides={lastSavedOverrides}
          setLastSavedOverrides={setLastSavedOverrides}
          showOrgChart={showOrgChart}
          setShowOrgChart={setShowOrgChart}
          organisationOptions={organisationOptions}
          selectedOrganisation={selectedOrganisation}
          onChangeOrganisation={setSelectedOrganisation}
          countryOptions={countryOptions}
          selectedCountry={selectedCountry}
          onChangeCountry={setSelectedCountry}
        />
      </div>

      <EmailVerificationModal
        data={verifyModalData}
        email={verifyModalEmail}
        onClose={() => setVerifyModalData(null)}
      />
      
      <StatusManagerModal
        isOpen={statusModalOpen}
        onClose={() => setStatusModalOpen(false)}
        statuses={statusOptions}
        onAddStatus={handleAddStatus}
        onRemoveStatus={handleRemoveStatus}
      />
    </div>
  );
}