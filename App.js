
import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import html2canvas from 'html2canvas';
import { Tree, TreeNode } from 'react-organizational-chart';
import './cms.css'; // CMS theme with Resume Tab enhancements
import './print-org-chart.css'; // Print-only: restrict output to org chart tree
import './nav-sidebar.css'; // Left-column navigation sidebar
// Admin feature removed (AdminUploadButton not imported)

/* ========================= CONSTANTS ========================= */
// SSE Configuration
const SSE_RECONNECT_BASE_DELAY_MS = 1000;
const SSE_RECONNECT_MAX_DELAY_MS = 30000;
const SSE_MAX_RECONNECT_ATTEMPTS = 5;
const API_PORT = 4000;

/* ========================= CRYPTO SIGNING UTILITIES ========================= */
// Sign content with a fresh ECDSA P-256 key and return signature + public key (both base64).
async function signExportData(content) {
  const keyPair = await crypto.subtle.generateKey(
    { name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign', 'verify']
  );
  const encoder = new TextEncoder();
  const sig = await crypto.subtle.sign(
    { name: 'ECDSA', hash: { name: 'SHA-256' } }, keyPair.privateKey, encoder.encode(content)
  );
  const pubJwk = await crypto.subtle.exportKey('jwk', keyPair.publicKey);
  const sigB64 = btoa(Array.from(new Uint8Array(sig), b => String.fromCharCode(b)).join(''));
  const pubB64 = btoa(JSON.stringify(pubJwk));
  return { signature: sigB64, publicKey: pubB64 };
}

// Verify a signature produced by signExportData. Returns true/false.
async function verifyImportData(content, sigB64, pubB64) {
  try {
    const pubJwk = JSON.parse(atob(pubB64));
    const publicKey = await crypto.subtle.importKey(
      'jwk', pubJwk, { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']
    );
    const sigBin = Uint8Array.from(atob(sigB64), c => c.charCodeAt(0));
    const encoder = new TextEncoder();
    return await crypto.subtle.verify(
      { name: 'ECDSA', hash: { name: 'SHA-256' } }, publicKey, sigBin, encoder.encode(content)
    );
  } catch (e) {
    console.error('[Signature] Verification error:', e);
    return false;
  }
}

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
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
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
      background: 'rgba(34,37,41,0.65)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 9999
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
function EmailComposeModal({ isOpen, onClose, toAddresses, candidateName, candidateData, userData, smtpConfig, recipientCandidates = [], onSendSuccess, statusOptions = [] }) {
  const [from, setFrom] = useState('');
  const [cc, setCc] = useState('');
  const [bcc, setBcc] = useState('');
  const [subject, setSubject] = useState('');
  const [body, setBody] = useState('');
  const [files, setFiles] = useState([]);
  const [sending, setSending] = useState(false);
  const [directSending, setDirectSending] = useState(false); // State for Direct Send
  const [sendMode, setSendMode] = useState('individual'); // 'individual' (BCC-style) or 'group' (CC-style)
  const [recipientVisibilityExpanded, setRecipientVisibilityExpanded] = useState(false);

  // Calendar / Google Meet state
  const [addMeet, setAddMeet] = useState(false);
  const [calendarSlots, setCalendarSlots] = useState([]);
  const [slotsLoading, setSlotsLoading] = useState(false);
  const [selectedSlotIndex, setSelectedSlotIndex] = useState(null);
  const [creatingEvent, setCreatingEvent] = useState(false);
  const [meetLink, setMeetLink] = useState('');
  const [icsString, setIcsString] = useState('');
  const [calendarError, setCalendarError] = useState('');
  const [slotStartDate, setSlotStartDate] = useState('');
  const [slotEndDate, setSlotEndDate] = useState('');
  const [interviewDuration, setInterviewDuration] = useState(30);
  const [glossaryCopied, setGlossaryCopied] = useState(false);
  const [copiedTag, setCopiedTag] = useState('');
  const [glossaryLocked, setGlossaryLocked] = useState(false);
  const glossaryRef = useRef(null);

  // Template & AI State
  const [templates, setTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [showAiInput, setShowAiInput] = useState(false);
  const [aiPrompt, setAiPrompt] = useState('');
  const [aiLoading, setAiLoading] = useState(false);
  const [showTagGlossary, setShowTagGlossary] = useState(false);

  const [to, setTo] = useState(toAddresses);
  
  // Dismiss locked glossary when clicking outside it; also clear the tag highlight
  useEffect(() => {
    if (!glossaryLocked) return;
    const handler = e => {
      if (glossaryRef.current && !glossaryRef.current.contains(e.target)) {
        setGlossaryLocked(false);
        setShowTagGlossary(false);
        setCopiedTag('');
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [glossaryLocked]);

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
      setSlotStartDate('');
      setSlotEndDate('');
      setInterviewDuration(30);
      setGlossaryCopied(false);
      setCopiedTag('');
    }
  }, [isOpen]);

  if (!isOpen) return null;

  // Apply dynamic template tags from candidate and user data
  const getInterviewDateTimeStrings = () => {
    const selectedSlot = (selectedSlotIndex != null && calendarSlots[selectedSlotIndex]) ? calendarSlots[selectedSlotIndex] : null;
    const interviewDate = selectedSlot ? new Date(selectedSlot.start).toLocaleDateString(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' }) : '';
    const interviewTime = selectedSlot ? new Date(selectedSlot.start).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }) : '';
    return { interviewDate, interviewTime };
  };

  const applyTags = (text) => {
    let t = text;
    // Candidate tags
    t = t.replace(/\[Candidate Name\]/gi, candidateData?.name || candidateName || '');
    t = t.replace(/\[Job Title\]/gi, candidateData?.jobtitle || candidateData?.role || '');
    t = t.replace(/\[Company Name\]/gi, candidateData?.company || candidateData?.organisation || '');
    t = t.replace(/\[Country\]/gi, candidateData?.country || '');
    // Legacy [name] tag
    t = t.replace(/\[name\]/gi, candidateData?.name || candidateName || '');
    // User / sender tags
    t = t.replace(/\[Your Name\]/gi, userData?.full_name || userData?.username || '');
    t = t.replace(/\[Your Company Name\]/gi, userData?.corporation || '');
    // Interview / calendar tags
    const { interviewDate, interviewTime } = getInterviewDateTimeStrings();
    t = t.replace(/\[Date of Interview\]/gi, interviewDate);
    t = t.replace(/\[Time of Interview\]/gi, interviewTime);
    t = t.replace(/\[Video Conference Link\]/gi, meetLink || '');
    return t;
  };

  // Apply tags resolved against a specific candidate object (used for sequential multi-send)
  const applyTagsFor = (text, c) => {
    let t = text;
    t = t.replace(/\[Candidate Name\]/gi, c?.name || '');
    t = t.replace(/\[Job Title\]/gi, c?.jobtitle || c?.role || '');
    t = t.replace(/\[Company Name\]/gi, c?.company || c?.organisation || '');
    t = t.replace(/\[Country\]/gi, c?.country || '');
    t = t.replace(/\[name\]/gi, c?.name || '');
    t = t.replace(/\[Your Name\]/gi, userData?.full_name || userData?.username || '');
    t = t.replace(/\[Your Company Name\]/gi, userData?.corporation || '');
    // Interview / calendar tags (same slot for all recipients when bulk-sending)
    const { interviewDate, interviewTime } = getInterviewDateTimeStrings();
    t = t.replace(/\[Date of Interview\]/gi, interviewDate);
    t = t.replace(/\[Time of Interview\]/gi, interviewTime);
    t = t.replace(/\[Video Conference Link\]/gi, meetLink || '');
    return t;
  };

  // Read a File as base64 string (without data: prefix)
  const readFileAsBase64 = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

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
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
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
      const now = new Date();
      // Use user-selected dates if provided, otherwise default to next 3 days
      let startISO, endISO;
      if (slotStartDate) {
        startISO = new Date(slotStartDate + 'T00:00:00').toISOString();
      } else {
        startISO = now.toISOString();
      }
      if (slotEndDate) {
        endISO = new Date(slotEndDate + 'T23:59:59').toISOString();
      } else {
        endISO = new Date(now.getTime() + 3 * 24 * 60 * 60 * 1000).toISOString();
      }
      const res = await fetch('http://localhost:4000/calendar/freebusy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
        body: JSON.stringify({ startISO, endISO, durationMinutes: interviewDuration }),
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
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
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
    finalBody = applyTags(finalBody);
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

    // Read selected files as base64 attachments (shared across all sends)
    const attachments = await Promise.all(files.map(async (file) => ({
        filename: file.name,
        content: await readFileAsBase64(file),
        contentType: file.type || 'application/octet-stream'
    })));

    const isMulti = recipientCandidates && recipientCandidates.length > 1;

    try {
      if (isMulti && sendMode === 'individual') {
        // Sequential per-candidate dispatch — each email is fully personalised, recipient only sees their own address
        let sent = 0;
        const failures = [];
        for (const cand of recipientCandidates) {
          const candEmail = (cand.email || '').trim();
          if (!candEmail) {
            failures.push(`${cand.name || `id:${cand.id}`}: no email address`);
            continue;
          }
          const finalSubject = applyTagsFor(subject, cand);
          let finalBody = applyTagsFor(body, cand);
          if (meetLink && !finalBody.includes(meetLink)) {
            finalBody += '\n\nJoin meeting: ' + meetLink;
          }
          const payload = {
            to: candEmail,
            // Explicitly omit cc/bcc in individual send mode to prevent exposing other recipients
            subject: finalSubject,
            body: finalBody,
            from,
            smtpConfig,
          };
          if (icsString) {
            // Strip ATTENDEE lines for other recipients so the calendar invite
            // only lists the current recipient — prevents address exposure.
            const recipientLower = candEmail.toLowerCase();
            payload.ics = icsString
              .split(/\r?\n/)
              .filter(line => {
                if (/^ATTENDEE[;:]/i.test(line)) {
                  return line.toLowerCase().includes(`mailto:${recipientLower}`);
                }
                return true;
              })
              .join('\r\n');
          }
          if (attachments.length > 0) payload.attachments = attachments;
          try {
            const res = await fetch('http://localhost:4000/send-email', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
              body: JSON.stringify(payload),
              credentials: 'include'
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed');
            sent++;
          } catch (err) {
            failures.push(`${cand.name || candEmail}: ${err.message}`);
          }
        }
        if (failures.length === 0) {
          alert(`${sent} email${sent !== 1 ? 's' : ''} sent successfully!`);
          if (typeof onSendSuccess === 'function') onSendSuccess(recipientCandidates);
        } else {
          alert(`${sent} sent. ${failures.length} failed:\n${failures.join('\n')}`);
          if (sent > 0 && typeof onSendSuccess === 'function') {
            const sentCandidates = recipientCandidates.filter(c => {
              const e = (c.email || '').trim();
              return e && !failures.some(f => f.includes(e));
            });
            if (sentCandidates.length) onSendSuccess(sentCandidates);
          }
        }
        onClose();
      } else {
        // Single-candidate send or group send (all recipients see each other)
        let finalBody = applyTags(body);
        if (meetLink && !finalBody.includes(meetLink)) {
          finalBody += '\n\nJoin meeting: ' + meetLink;
        }
        const payload = {
          to, cc, bcc, subject,
          body: finalBody,
          from,
          smtpConfig,
        };
        if (icsString) payload.ics = icsString;
        if (attachments.length > 0) payload.attachments = attachments;
        const res = await fetch('http://localhost:4000/send-email', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
          body: JSON.stringify(payload),
          credentials: 'include'
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to send');
        alert('Email sent successfully!');
        if (typeof onSendSuccess === 'function') onSendSuccess(recipientCandidates);
        onClose();
      }
    } catch (e) {
        alert('Error sending email: ' + e.message);
    } finally {
        setDirectSending(false);
    }
  };

  const labelStyle = { display: 'block', marginBottom: 6, fontWeight: 700, fontSize: 13, color: 'var(--azure-dragon)' };
  const inputStyle = { width: '100%', padding: '8px 10px', boxSizing: 'border-box', border: '1px solid var(--desired-dawn)', borderRadius: 6, fontSize: 13, fontFamily: 'inherit', outline: 'none', transition: 'border-color 0.15s' };

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(216,216,216,0.85)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 10000
    }}>
      <div className="app-card" style={{
        width: 700, maxWidth: '95vw',
        display: 'flex', flexDirection: 'column', maxHeight: '90vh',
        borderRadius: 12, boxShadow: '0 8px 32px rgba(7,54,121,0.22)', overflow: 'hidden'
      }} onClick={e => e.stopPropagation()}>
        
        {/* Header */}
        <div style={{ padding: '16px 24px', background: 'linear-gradient(135deg,var(--azure-dragon),var(--cool-blue))', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0, fontSize: 18, color: '#fff', fontWeight: 700, letterSpacing: '0.3px' }}>✉ New Message</h3>
          <button onClick={onClose} style={{ background: 'rgba(255,255,255,0.15)', border: '1px solid rgba(255,255,255,0.3)', fontSize: 18, color: '#fff', cursor: 'pointer', borderRadius: '50%', width: 30, height: 30, display: 'flex', alignItems: 'center', justifyContent: 'center', lineHeight: 1, padding: 0 }} title="Close">×</button>
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

            {/* Recipient Visibility — collapsible bar, only shown when multiple recipients are selected */}
            {recipientCandidates && recipientCandidates.length > 1 && (
              <div style={{ marginBottom: 16 }}>
                <button
                  type="button"
                  onClick={() => setRecipientVisibilityExpanded(e => !e)}
                  aria-expanded={recipientVisibilityExpanded}
                  aria-label={`Recipient Visibility – ${sendMode === 'individual' ? 'Send individually' : 'Send as group'}. Click to ${recipientVisibilityExpanded ? 'collapse' : 'expand'}`}
                  style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#f0f9ff', border: '1px solid #bae6fd', borderRadius: 8, padding: '7px 14px', cursor: 'pointer', width: '100%', textAlign: 'left', color: '#0369a1', fontWeight: 700, fontSize: 13 }}
                >
                  <span style={{ fontSize: 11, transition: 'transform 0.15s', display: 'inline-block', transform: recipientVisibilityExpanded ? 'rotate(90deg)' : 'none' }}>▶</span>
                  Recipient Visibility
                  <span style={{ marginLeft: 'auto', fontWeight: 400, fontSize: 12, color: '#0369a1' }}>
                    {sendMode === 'individual' ? '✓ Send individually (default)' : 'Send as group'}
                  </span>
                </button>
                {recipientVisibilityExpanded && (
                  <div style={{ padding: '10px 14px', background: '#f0f9ff', borderRadius: '0 0 8px 8px', border: '1px solid #bae6fd', borderTop: 'none' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                      <label style={{ display: 'flex', alignItems: 'flex-start', gap: 8, cursor: 'pointer', fontSize: 13 }}>
                        <input type="radio" name="sendMode" value="individual" checked={sendMode === 'individual'} onChange={() => setSendMode('individual')} style={{ marginTop: 2 }} />
                        <span><b>Send individually</b> – each recipient gets a separate email and sees only their own address <span style={{ color: '#0369a1', fontSize: 12 }}>(default, recommended)</span></span>
                      </label>
                      <label style={{ display: 'flex', alignItems: 'flex-start', gap: 8, cursor: 'pointer', fontSize: 13 }}>
                        <input type="radio" name="sendMode" value="group" checked={sendMode === 'group'} onChange={() => setSendMode('group')} style={{ marginTop: 2 }} />
                        <span><b>Send as group</b> – one email to all recipients; everyone sees each other's address</span>
                      </label>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Template & AI Tools Section */}
            <div style={{ marginBottom: 16, padding: '12px', background: '#f8fafc', borderRadius: 8, border: '1px solid var(--neutral-border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <label style={{...labelStyle, marginBottom: 0}}>Email Template & AI Tools</label>
                <span
                  ref={glossaryRef}
                  tabIndex={0}
                  style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', cursor: 'pointer', fontSize: 13, outline: 'none' }}
                  onMouseEnter={() => { if (!glossaryLocked) setShowTagGlossary(true); }}
                  onMouseLeave={() => { if (!glossaryLocked) setShowTagGlossary(false); }}
                  onFocus={() => setShowTagGlossary(true)}
                  onBlur={() => { if (!glossaryLocked) setShowTagGlossary(false); }}
                  onClick={() => { setGlossaryLocked(l => !l); setShowTagGlossary(true); }}
                >
                  <span style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 18, height: 18, borderRadius: '50%', background: 'var(--black-beauty)', color: '#fff', fontWeight: 700, fontSize: 11, lineHeight: 1 }}>?</span>
                  <span style={{ marginLeft: 4, fontSize: 11, color: 'var(--muted)', fontWeight: 600 }}>Tag Glossary</span>
                  {showTagGlossary && (() => {
                    const TAG_GROUPS = [
                      { label: 'Candidate', color: '#6deaf9', tags: [
                        { tag: '[Candidate Name]', desc: "Candidate's full name" },
                        { tag: '[Job Title]', desc: "Candidate's professional role" },
                        { tag: '[Company Name]', desc: "Candidate's current employer" },
                        { tag: '[Country]', desc: "Candidate's geographic location" },
                      ]},
                      { label: 'Sender', color: '#86efac', tags: [
                        { tag: '[Your Name]', desc: "Your account's full name" },
                        { tag: '[Your Company Name]', desc: "Your registered company name" },
                      ]},
                      { label: 'Calendar / Interview', color: '#fde68a', tags: [
                        { tag: '[Date of Interview]', desc: 'Selected interview date (from calendar slot)' },
                        { tag: '[Time of Interview]', desc: 'Selected interview time (from calendar slot)' },
                        { tag: '[Video Conference Link]', desc: 'Google Meet link (after creating event)' },
                      ]},
                    ];
                    const allTagsText = TAG_GROUPS.flatMap(g => g.tags.map(t => t.tag)).join(', ');
                    const copyAll = (e) => {
                      e.stopPropagation();
                      navigator.clipboard.writeText(allTagsText).catch(() => {});
                      setGlossaryCopied(true);
                      setTimeout(() => setGlossaryCopied(false), 1500);
                    };
                    const copyTag = (e, tag) => {
                      e.stopPropagation();
                      navigator.clipboard.writeText(tag).catch(() => {});
                      setCopiedTag(tag);
                      setGlossaryLocked(true);
                    };
                    return (
                      <div
                        style={{ position: 'absolute', top: '110%', right: 0, zIndex: 9999, background: 'var(--black-beauty)', color: '#f1f5f9', borderRadius: 10, padding: '12px 14px', minWidth: 320, boxShadow: '0 6px 24px rgba(34,37,41,0.4)', fontSize: 12, lineHeight: 1.7 }}
                        onMouseEnter={() => setShowTagGlossary(true)}
                        onMouseLeave={() => { if (!glossaryLocked) setShowTagGlossary(false); }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, borderBottom: '1px solid rgba(255,255,255,0.2)', paddingBottom: 6 }}>
                          <span style={{ fontWeight: 700, fontSize: 12, letterSpacing: '0.4px' }}>Available Template Tags</span>
                          <button
                            type="button"
                            onClick={copyAll}
                            title="Copy all tags"
                            style={{ background: glossaryCopied ? '#6deaf9' : 'rgba(255,255,255,0.15)', border: 'none', borderRadius: 5, color: glossaryCopied ? '#073679' : '#fff', cursor: 'pointer', fontSize: 11, padding: '3px 8px', fontWeight: 700, transition: 'all 0.15s' }}
                          >
                            {glossaryCopied ? '✓ Copied!' : '⎘ Copy All'}
                          </button>
                        </div>
                        {TAG_GROUPS.map(g => (
                          <div key={g.label} style={{ marginBottom: 6 }}>
                            <div style={{ fontWeight: 700, fontSize: 10, color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 3 }}>{g.label}</div>
                            {g.tags.map(({ tag, desc }) => (
                              <div key={tag} style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginBottom: 2 }}>
                                <b
                                  onClick={e => copyTag(e, tag)}
                                  title="Click to copy this tag"
                                  style={{ color: copiedTag === tag ? '#6deaf9' : g.color, cursor: 'pointer', borderRadius: 4, padding: '0 3px', transition: 'background 0.15s', background: copiedTag === tag ? 'rgba(109,234,249,0.1)' : 'transparent', userSelect: 'none' }}
                                >{tag}</b>
                                <span style={{ color: 'rgba(255,255,255,0.6)', fontSize: 11 }}>– {desc}</span>
                              </div>
                            ))}
                          </div>
                        ))}
                        <div style={{ marginTop: 6, fontSize: 10, color: 'rgba(255,255,255,0.4)', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 5 }}>Click any tag to copy it · "⎘ Copy All" copies all tags{glossaryLocked ? ' · Click outside to close' : ''}</div>
                      </div>
                    );
                  })()}
                </span>
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
            <div style={{ marginBottom: 16, padding: '14px 16px', background: 'linear-gradient(135deg,#f0f4ff,#e8f0fb)', borderRadius: 10, border: '1px solid var(--cool-blue)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 16 }}>📅</span>
                  <span style={{ fontWeight: 700, color: 'var(--azure-dragon)', fontSize: 14 }}>Calendar & Google Meet</span>
                </div>
                <button
                  type="button"
                  onClick={handleConnectCalendar}
                  className="btn-secondary"
                  style={{ padding: '5px 10px', fontSize: 12 }}
                >
                  Connect Calendar
                </button>
              </div>

              {/* Row 1: checkbox + duration */}
              <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input type="checkbox" checked={addMeet} onChange={e => setAddMeet(e.target.checked)} />
                  <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--azure-dragon)' }}>Add Google Meet</span>
                </label>
                {addMeet && (
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
                    <span style={{ fontWeight: 600, color: 'var(--cool-blue)' }}>Duration:</span>
                    <select
                      value={interviewDuration}
                      onChange={e => setInterviewDuration(Number(e.target.value))}
                      style={{ padding: '4px 8px', border: '1px solid var(--cool-blue)', borderRadius: 6, fontSize: 13, background: '#fff', color: 'var(--azure-dragon)', fontWeight: 600 }}
                    >
                      <option value={15}>15 min</option>
                      <option value={30}>30 min</option>
                      <option value={45}>45 min</option>
                      <option value={60}>60 min</option>
                    </select>
                  </label>
                )}
              </div>

              {/* Row 2: date range + find slots */}
              {addMeet && (
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap', background: '#fff', padding: '8px 10px', borderRadius: 8, border: '1px solid var(--desired-dawn)' }}>
                  <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ color: 'var(--cool-blue)', fontWeight: 600 }}>From:</span>
                    <input
                      type="date"
                      value={slotStartDate}
                      min={new Date().toISOString().slice(0, 10)}
                      onChange={e => setSlotStartDate(e.target.value)}
                      style={{ padding: '4px 8px', border: '1px solid var(--desired-dawn)', borderRadius: 6, fontSize: 13 }}
                    />
                  </label>
                  <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ color: 'var(--cool-blue)', fontWeight: 600 }}>To:</span>
                    <input
                      type="date"
                      value={slotEndDate}
                      min={slotStartDate || new Date().toISOString().slice(0, 10)}
                      onChange={e => setSlotEndDate(e.target.value)}
                      style={{ padding: '4px 8px', border: '1px solid var(--desired-dawn)', borderRadius: 6, fontSize: 13 }}
                    />
                  </label>
                  <button
                    type="button"
                    onClick={handleFindSlots}
                    disabled={slotsLoading}
                    style={{ padding: '5px 12px', background: 'var(--azure-dragon)', color: '#fff', border: 'none', borderRadius: 6, cursor: slotsLoading ? 'not-allowed' : 'pointer', fontSize: 13, fontWeight: 600, opacity: slotsLoading ? 0.7 : 1 }}
                  >
                    {slotsLoading ? '⏳ Finding…' : '🔍 Find Slots'}
                  </button>
                </div>
              )}

              {calendarError && <div style={{ color: 'var(--danger)', fontSize: 13, marginBottom: 8, padding: '6px 8px', background: '#fff1f0', borderRadius: 6, border: '1px solid #fca5a5' }}>{calendarError}</div>}

              {/* Slots grouped by day */}
              {calendarSlots && calendarSlots.length > 0 && addMeet && (() => {
                // Group slots by date string
                const groups = {};
                calendarSlots.forEach((s, i) => {
                  const day = new Date(s.start).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
                  if (!groups[day]) groups[day] = [];
                  groups[day].push({ slot: s, idx: i });
                });
                return (
                  <div style={{ marginTop: 4 }}>
                    <div style={{ fontSize: 12, color: 'var(--argent)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      Select a slot ({interviewDuration} min) · {calendarSlots.length} available
                    </div>
                    {Object.entries(groups).map(([day, entries]) => (
                      <div key={day} style={{ marginBottom: 8 }}>
                        <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--cool-blue)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 4, paddingBottom: 3, borderBottom: '1px solid var(--desired-dawn)' }}>{day}</div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                          {entries.map(({ slot: s, idx: i }) => {
                            const isSelected = selectedSlotIndex === i;
                            const timeLabel = new Date(s.start).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' – ' + new Date(s.end).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                            return (
                              <button
                                key={i}
                                type="button"
                                onClick={() => setSelectedSlotIndex(i)}
                                style={{
                                  padding: '5px 10px', borderRadius: 20, fontSize: 12, fontWeight: isSelected ? 700 : 500, cursor: 'pointer', transition: 'all 0.15s',
                                  background: isSelected ? 'var(--azure-dragon)' : '#fff',
                                  color: isSelected ? '#fff' : 'var(--cool-blue)',
                                  border: isSelected ? '1.5px solid var(--azure-dragon)' : '1.5px solid var(--cool-blue)'
                                }}
                              >{timeLabel}</button>
                            );
                          })}
                        </div>
                      </div>
                    ))}

                    <div style={{ marginTop: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      <button type="button" onClick={handleCreateEvent} disabled={creatingEvent || selectedSlotIndex == null} style={{ padding: '6px 14px', background: 'var(--azure-dragon)', color: '#fff', border: 'none', borderRadius: 6, fontWeight: 700, fontSize: 13, cursor: selectedSlotIndex == null || creatingEvent ? 'not-allowed' : 'pointer', opacity: selectedSlotIndex == null ? 0.5 : 1 }}>
                        {creatingEvent ? 'Creating…' : '📌 Create Event & Add Link'}
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          if (meetLink) {
                            if (!body.includes(meetLink)) setBody(prev => prev + '\n\nJoin meeting: ' + meetLink);
                          } else {
                            alert('No meet link present. Create event first.');
                          }
                        }}
                        disabled={!meetLink}
                        style={{ padding: '6px 12px', background: meetLink ? 'var(--cool-blue)' : 'var(--desired-dawn)', color: '#fff', border: 'none', borderRadius: 6, fontWeight: 600, fontSize: 13, cursor: meetLink ? 'pointer' : 'not-allowed', opacity: meetLink ? 1 : 0.6 }}
                      >
                        Insert Meet Link into Message
                      </button>

                      {meetLink && (
                        <a href={meetLink} target="_blank" rel="noopener noreferrer" style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 4, fontSize: 13, color: 'var(--cool-blue)', textDecoration: 'none', fontWeight: 600 }}>
                          <span>🔗</span> Open Meet
                        </a>
                      )}
                    </div>
                  </div>
                );
              })()}
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
      background: 'rgba(34,37,41,0.65)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 10001
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
      background: 'rgba(34,37,41,0.65)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 10001
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

function CompensationCalculatorModal({ isOpen, onClose, onSave, initialValue }) {
  const COMP_KEYS = ['baseSalary', 'allowances', 'bonus', 'commission', 'rsu'];
  const emptyFields = Object.fromEntries(COMP_KEYS.map(k => [k, '']));
  const [fields, setFields] = useState(emptyFields);
  const [totalOverride, setTotalOverride] = useState('');
  const [manualTotal, setManualTotal] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setFields(emptyFields);
      const existing = initialValue != null && initialValue !== '' ? String(initialValue) : '';
      setTotalOverride(existing);
      setManualTotal(existing !== '');
    }
  }, [isOpen, initialValue]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!isOpen) return null;

  const autoTotal = COMP_KEYS.reduce((sum, k) => sum + (parseFloat(fields[k]) || 0), 0);
  const displayTotal = manualTotal ? totalOverride : (autoTotal === 0 ? '' : String(autoTotal));

  const handleChange = (key, value) => {
    if (value !== '' && !/^\d*\.?\d*$/.test(value)) return;
    setFields(prev => ({ ...prev, [key]: value }));
  };

  const handleTotalChange = (value) => {
    if (value !== '' && !/^\d*\.?\d*$/.test(value)) return;
    setManualTotal(true);
    setTotalOverride(value);
  };

  const handleSave = () => {
    const finalValue = manualTotal ? totalOverride : (autoTotal === 0 ? '' : String(autoTotal));
    onSave(finalValue);
    onClose();
  };

  const labelStyle = { display: 'block', marginBottom: 4, fontWeight: 600, fontSize: 12, color: 'var(--muted)' };
  const inputStyle = { width: '100%', boxSizing: 'border-box', padding: '6px 10px', font: 'inherit', fontSize: 13, background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: 6, marginBottom: 12 };
  const disabledInputStyle = { ...inputStyle, background: '#f1f5f9', color: '#94a3b8', cursor: 'not-allowed' };

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(34,37,41,0.65)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 10001
    }} onClick={onClose}>
      <div className="app-card" style={{ width: 420, padding: 24 }} onClick={e => e.stopPropagation()}>
        <button onClick={onClose} style={{ position: 'absolute', top: 12, right: 12, border: 'none', background: 'transparent', fontSize: 20, cursor: 'pointer', color: 'var(--argent)' }}>×</button>
        <h3 style={{ marginTop: 0, marginBottom: 20, color: 'var(--azure-dragon)', fontSize: 16 }}>Compensation Calculator</h3>
        {[
          { key: 'baseSalary', label: 'Annual Current Base Salary' },
          { key: 'allowances', label: 'Allowances' },
          { key: 'bonus', label: 'Bonus' },
          { key: 'commission', label: 'Commission' },
          { key: 'rsu', label: 'Restricted Stock Units (RSU)' },
        ].map(({ key, label }) => (
          <div key={key}>
            <label style={{ ...labelStyle, color: manualTotal ? '#94a3b8' : 'var(--muted)' }}>{label}</label>
            <input
              type="text"
              inputMode="decimal"
              placeholder="0"
              value={fields[key]}
              disabled={manualTotal}
              onChange={e => handleChange(key, e.target.value)}
              style={manualTotal ? disabledInputStyle : inputStyle}
            />
          </div>
        ))}
        <div style={{ borderTop: '2px solid var(--neutral-border)', marginBottom: 12, paddingTop: 12 }}>
          <label style={{ ...labelStyle, color: 'var(--azure-dragon)', fontWeight: 700 }}>
            Total Annual Remuneration {manualTotal ? <span style={{ fontWeight: 400, fontSize: 11, color: '#ef4444' }}>(manual – individual fields locked)</span> : <span style={{ fontWeight: 400, fontSize: 11, color: 'var(--argent)' }}>(auto-calculated)</span>}
          </label>
          <input
            type="text"
            inputMode="decimal"
            placeholder="0"
            value={displayTotal}
            onChange={e => handleTotalChange(e.target.value)}
            style={{ ...inputStyle, marginBottom: 0, fontWeight: 700, border: manualTotal ? '1px solid #ef4444' : '1px solid var(--azure-dragon)', background: manualTotal ? '#fff7f7' : '#f0f9ff' }}
          />
          {manualTotal && (
            <button
              onClick={() => { setManualTotal(false); setTotalOverride(''); }}
              style={{ marginTop: 6, fontSize: 11, color: '#3b82f6', background: 'none', border: 'none', cursor: 'pointer', padding: 0, textDecoration: 'underline' }}
            >Reset to auto-sum</button>
          )}
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 16 }}>
          <button onClick={onClose} className="btn-secondary" style={{ padding: '7px 18px', fontSize: 13 }}>Cancel</button>
          <button onClick={handleSave} className="btn-primary" style={{ padding: '7px 18px', fontSize: 13 }}>Save</button>
        </div>
      </div>
    </div>
  );
}


// Sticky column constants (defined outside component to avoid recreation on each render)
const FROZEN_ACTIONS_WIDTH = 80;
const CHECKBOX_COL_WIDTH = 36;
const FROZEN_EDGE_BORDER_COLOR = '#cbd5e1'; // subtle separator for permanent edge columns
const FROZEN_COL_BORDER_COLOR = '#93c5fd';  // blue separator for user-pinned columns (📌)

// Small component to display candidate avatar with graceful fallback on image error
function CandidateAvatar({ picSrc, initials, avatarBg, avatarText }) {
  const [imgFailed, setImgFailed] = React.useState(false);
  if (picSrc && !imgFailed) {
    return (
      <img
        src={picSrc}
        alt={initials}
        style={{ width: 28, height: 28, borderRadius: '50%', objectFit: 'cover', flexShrink: 0, border: '1px solid #e5e7eb' }}
        onError={() => setImgFailed(true)}
      />
    );
  }
  return (
    <span style={{ width: 28, height: 28, borderRadius: '50%', background: avatarBg, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, fontWeight: 700, color: avatarText, flexShrink: 0, letterSpacing: '0.5px' }}>{initials}</span>
  );
}

function CandidatesTable({
  candidates = [],
  onDelete, deleteError, onSave, onAutoSave, type, page, setPage, totalPages, editRows, setEditRows,
  skillsetMapping,
  searchExpanded, onToggleSearch, globalSearchInput, onGlobalSearchChange, onGlobalSearchSubmit, onClearSearch,
  onViewProfile, // NEW PROP to handle viewing profile
  statusOptions, // Prop for status options
  onOpenStatusModal, // Prop to open status modal
  allCandidates, // Passed for bulk verification/sync
  user, // Logged-in user for template tags
  onDockIn, // Callback to refresh candidates after DB Dock In import
  tokensLeft = 0, // Current token balance from parent App
}) {
  const DEFAULT_WIDTH = 140;
  const MIN_WIDTH = 90;
  const GLOBAL_MAX_WIDTH = 500;
  const COLUMN_WIDTHS_KEY = 'candidatesTableColWidths';
  const FIELD_MAX_WIDTHS = { skillset: 900 };

  const [selectedIds, setSelectedIds] = useState([]);
  const [deleting, setDeleting] = useState(false);
  const [colWidths, setColWidths] = useState({});
  const [savingAll, setSavingAll] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');
  const [saveError, setSaveError] = useState('');

  // DB Dock In state
  const dockInRef = useRef(null);
  const [dockInUploading, setDockInUploading] = useState(false);
  const [dockInError, setDockInError] = useState('');
  // DB Dock In 3-step wizard state
  const [dockInWizOpen, setDockInWizOpen] = useState(false);
  const [dockInWizStep, setDockInWizStep] = useState(1);
  const [dockInWizMode, setDockInWizMode] = useState(''); // 'normal' | 'analytic'
  const [dockInWizFile, setDockInWizFile] = useState(null);
  const [dockInAnalyticProgress, setDockInAnalyticProgress] = useState(''); // analytic stage message
  const [dockInAnalyticPct, setDockInAnalyticPct] = useState(0); // 0-100 progress bar
  const dockInWizFileRef = useRef(null);   // used by modal wizard (inside hidden table div)
  const dockInInlineFileRef = useRef(null); // used by inline empty-state wizard
  // DB Dock Out state
  const [dockOutClearing, setDockOutClearing] = useState(false);
  const [dockOutConfirmOpen, setDockOutConfirmOpen] = useState(false);
  const [dockOutNoWarning, setDockOutNoWarning] = useState(() => localStorage.getItem('dockOutSkipWarning') === '1');
  // Analytic DB: pre-file-parse confirmation state
  const [dockInAnalyticConfirm, setDockInAnalyticConfirm] = useState(false);
  const [dockInNewRecordCount, setDockInNewRecordCount] = useState(0);
  const [dockInRejectedRows, setDockInRejectedRows] = useState([]); // rows failing mandatory field validation
  const [dockInPeeking, setDockInPeeking] = useState(false); // true while parsing file for count
  // Step 3 — Resume Upload state
  const [dockInNewRecords, setDockInNewRecords] = useState([]); // [{tempId, name}] new records identified in Step 2
  const [dockInResumeFiles, setDockInResumeFiles] = useState([]); // File[] selected by user in Step 3
  const [dockInResumeMatches, setDockInResumeMatches] = useState([]); // [{record, file|null}] after name matching
  const dockInResumeMatchesRef = useRef([]); // ref copy to avoid stale-closure in handleDockIn
  const dockInResumeInlineRef = useRef(null); // hidden resume input – inline wizard
  const dockInResumeModalRef = useRef(null);  // hidden resume input – modal wizard
  // Step 3 (analytic mode) — Role & Skillset Confirmation
  const [dockInRoleTagPairs, setDockInRoleTagPairs] = useState([]); // [{roleTag, jskillset}] unique pairs from DB Copy
  const [dockInSelectedPair, setDockInSelectedPair] = useState(null); // {roleTag, jskillset} confirmed by user

  // Track newly-added candidate IDs for the "New" badge
  const [newCandidateIds, setNewCandidateIds] = useState(new Set());
  const prevCandidateIdsRef = useRef(null);
  
  // Sync Entries State
  const [syncLoading, setSyncLoading] = useState(false);
  const [syncMessage, setSyncMessage] = useState('');
  
  // Checkbox Rename Workflow State
  const [renameCheckboxId, setRenameCheckboxId] = useState(null);
  const [renameCategory, setRenameCategory] = useState('');
  const [renameValue, setRenameValue] = useState('');
  const [renameMessage, setRenameMessage] = useState('');
  const [renameError, setRenameError] = useState('');
  
  // Compensation calculator modal state
  const [compModalOpen, setCompModalOpen] = useState(false);
  const [compModalCandidateId, setCompModalCandidateId] = useState(null);
  const [compModalInitialValue, setCompModalInitialValue] = useState('');

  // Email modal & SMTP state
  const [emailModalOpen, setEmailModalOpen] = useState(false);
  const [composedToAddresses, setComposedToAddresses] = useState('');
  const [emailRecipients, setEmailRecipients] = useState([]);
  const [singleCandidateName, setSingleCandidateName] = useState('');
  const [singleCandidateData, setSingleCandidateData] = useState(null);
  const [smtpConfig, setSmtpConfig] = useState(null);
  const [smtpModalOpen, setSmtpModalOpen] = useState(false);


  // Load saved SMTP config from server when user logs in.
  // The login response already includes the full config (with password) so we
  // use it directly when present.  For sessions restored via cookie (user/resolve)
  // the config isn't bundled, so we fall back to the dedicated endpoint.
  useEffect(() => {
    if (!user || !user.username) return;
    if (user.smtpConfig) {
      setSmtpConfig(user.smtpConfig);
      return;
    }
    fetch('http://localhost:4000/smtp-config', { credentials: 'include' })
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && data.ok && data.config) {
          setSmtpConfig(data.config);
        }
      })
      .catch(() => {}); // ignore errors, user can configure manually
  }, [user]);

  const tableRef = useRef(null);

  // User-pinned middle columns (click header to toggle freeze)
  const [frozenMiddleCols, setFrozenMiddleCols] = useState(() => new Set());
  const toggleFrozenMiddleCol = key => setFrozenMiddleCols(prev => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  const fields = [
    { key: 'name', label: 'Name', type: 'text', editable: true },
    { key: 'role', label: 'Job Title', type: 'text', editable: true },
    { key: 'organisation', label: 'Company', type: 'text', editable: true },
    { key: 'type', label: 'Product', type: 'text', editable: false },
    { key: 'sector', label: 'Sector', type: 'text', editable: true },
    { key: 'seniority', label: 'Seniority', type: 'text', editable: true },
    { key: 'job_family', label: 'Job Family', type: 'text', editable: true },
    { key: 'skillset', label: 'Skillset', type: 'text', editable: false },
    { key: 'geographic', label: 'Geographic', type: 'text', editable: true },
    { key: 'country', label: 'Country', type: 'text', editable: true },
    { key: 'compensation', label: 'Compensation', type: 'number', editable: true },
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

  // Helper: remove a set of IDs from the newCandidateIds Set
  const dismissNewBadges = ids => setNewCandidateIds(prev => { const n = new Set(prev); ids.forEach(id => n.delete(id)); return n; });

  // When candidate list becomes empty, reset inline wizard to Step 1
  useEffect(() => {
    if (!allCandidates || allCandidates.length === 0) {
      setDockInWizStep(1);
      setDockInWizMode('');
      setDockInWizFile(null);
      setDockInError('');
      setDockInAnalyticProgress('');
      setDockInAnalyticPct(0);
      setDockInAnalyticConfirm(false);
      setDockInNewRecordCount(0);
      setDockInRejectedRows([]);
      setDockInNewRecords([]);
      setDockInResumeFiles([]);
      setDockInResumeMatches([]);
      dockInResumeMatchesRef.current = [];
      setDockInRoleTagPairs([]);
      setDockInSelectedPair(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allCandidates?.length]);

  // Detect newly added candidates and show "New" badge for 8 seconds
  useEffect(() => {
    const currentIds = new Set(candidates.map(c => String(c.id)));
    if (prevCandidateIdsRef.current !== null) {
      const added = [];
      currentIds.forEach(id => { if (!prevCandidateIdsRef.current.has(id)) added.push(id); });
      if (added.length) {
        setNewCandidateIds(prev => { const n = new Set(prev); added.forEach(id => n.add(id)); return n; });
        setTimeout(() => dismissNewBadges(added), 8000);
      }
    }
    prevCandidateIdsRef.current = currentIds;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidates]);

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
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
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

    if (renameCategory === 'Compensation' && isNaN(Number(renameValue.trim()))) {
      setRenameError('Compensation must be a numeric value.');
      return;
    }

    try {
      // Map frontend category names to database field names
      const fieldMap = {
        'Job Title': 'role',
        'Company': 'organisation',
        'Sector': 'sector',
        'Compensation': 'compensation',
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
    setEmailRecipients(selected);

    if (selected.length === 1) {
        setSingleCandidateName(selected[0].name || '');
        setSingleCandidateData(selected[0]);
    } else {
        setSingleCandidateName('');
        setSingleCandidateData(null);
    }

    setEmailModalOpen(true);
  };

  const handleEmailSendSuccess = (sentCandidates) => {
    if (!Array.isArray(sentCandidates) || !sentCandidates.length) return;
    if (!statusOptions.includes('Contacted')) return;
    sentCandidates.forEach(cand => {
      if (cand && cand.id !== null && cand.id !== undefined) {
        handleEditChange(cand.id, 'sourcing_status', 'Contacted');
      }
    });
  };

  const handleEditChange = (id, field, value) => {
    if (['skillset', 'type'].includes(field)) return;
    if (field === 'compensation' && value !== '' && !/^\d*\.?\d*$/.test(value)) return;

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

  const HEADER_ROW_HEIGHT = 38;

  // Computes { fieldKey: leftOffset } for all user-pinned middle columns in order
  const computePinnedLeftOffsets = useMemo(() => {
    const nameWidth = colWidths['name'] || DEFAULT_WIDTH;
    let acc = 44 + nameWidth;
    const map = {};
    visibleFields.forEach(f => {
      if (f.key === 'name' || f.key === 'sourcing_status') return;
      if (frozenMiddleCols.has(f.key)) {
        map[f.key] = acc;
        acc += colWidths[f.key] || DEFAULT_WIDTH;
      }
    });
    return map;
  }, [visibleFields, colWidths, frozenMiddleCols]);

  const getDisplayValue = (c, f) => {
    let v = editRows[c.id]?.[f.key] ?? '';
    if (v === '' || v == null) {
      v = f.key === 'type' ? (c.type ?? c.product ?? '') : (c[f.key] ?? '');
    }
    if (f.key === 'skillset') v = prettifySkillset(v);
    return v;
  };

  const openCompModal = (candidateId, value) => {
    setCompModalCandidateId(candidateId);
    setCompModalInitialValue(value);
    setCompModalOpen(true);
  };

  const renderBodyCell = (c, f, idx, frozen = false, extraStyle = {}) => {
    const readOnly = ['skillset', 'type'].includes(f.key);
    const maxForField = FIELD_MAX_WIDTHS[f.key] || GLOBAL_MAX_WIDTH;
    const displayValue = getDisplayValue(c, f);
    const cellBg = idx % 2 ? '#ffffff' : '#f9fafb';

    // Name cell: avatar circle + editable input + optional "New" badge
    if (f.key === 'name') {
      const rawName = displayValue || '';
      const initials = rawName.split(/\s+/).slice(0, 2).map(s => s[0]?.toUpperCase()).filter(Boolean).join('') || '?';
      const avatarPalette = ['#4c82b8', '#073679', '#6deaf9'];
      const avatarBg = avatarPalette[(rawName.charCodeAt(0) || 0) % 3];
      const avatarText = avatarBg === '#6deaf9' ? '#222529' : '#fff';
      const picSrc = c.pic && typeof c.pic === 'string' ? c.pic : null;
      const isNewCandidate = newCandidateIds.has(String(c.id));
      return (
        <td key={f.key} data-field={f.key} style={{ overflow: 'hidden', width: colWidths[f.key] || DEFAULT_WIDTH, maxWidth: maxForField, minWidth: MIN_WIDTH, padding: '4px 6px', verticalAlign: 'middle', fontSize: 13, color: 'var(--muted)', borderBottom: '1px solid #eef2f5', height: HEADER_ROW_HEIGHT, background: cellBg, ...extraStyle }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <CandidateAvatar picSrc={picSrc} initials={initials} avatarBg={avatarBg} avatarText={avatarText} />
            <input type="text" value={displayValue} onChange={e => handleEditChange(c.id, 'name', e.target.value)} style={{ flex: 1, minWidth: 0, boxSizing: 'border-box', padding: '4px 8px', font: 'inherit', fontSize: 12, background: '#ffffff' }} />
            {isNewCandidate && (
              <span
                title="Newly added profile"
                onMouseEnter={() => dismissNewBadges([String(c.id)])}
                style={{ flexShrink: 0, fontSize: 9, fontWeight: 800, letterSpacing: '0.5px', padding: '1px 5px', borderRadius: 6, background: 'var(--robins-egg, #6deaf9)', color: '#073679', textTransform: 'uppercase', cursor: 'default', userSelect: 'none', lineHeight: '14px' }}
              >New</span>
            )}
          </div>
        </td>
      );
    }

    return (
      <td key={f.key} data-field={f.key} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', width: colWidths[f.key] || DEFAULT_WIDTH, maxWidth: maxForField, minWidth: MIN_WIDTH, padding: '4px 6px', verticalAlign: 'middle', fontSize: 13, color: 'var(--muted)', borderBottom: '1px solid #eef2f5', height: HEADER_ROW_HEIGHT, background: cellBg, ...extraStyle }}>
        {readOnly
          ? <span style={{ display: 'block', width: '100%', background: f.key === 'skillset' ? '#fff' : '#f1f5f9', padding: '3px 8px', border: '1px solid var(--neutral-border)', borderRadius: 4, boxSizing: 'border-box', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12 }} title={displayValue}>{displayValue}</span>
          : f.key === 'sourcing_status'
            ? <select value={displayValue || ''} onChange={e => handleEditChange(c.id, f.key, e.target.value)} style={{ width: '100%', boxSizing: 'border-box', padding: '4px 8px', font: 'inherit', fontSize: 12, background: '#ffffff', border: '1px solid var(--desired-dawn)', borderRadius: 6 }}>
                <option value="">-- Select Status --</option>
                {statusOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
              </select>
            : f.key === 'seniority'
              ? <select value={displayValue || ''} onChange={e => handleEditChange(c.id, f.key, e.target.value)} style={{ width: '100%', boxSizing: 'border-box', padding: '4px 8px', font: 'inherit', fontSize: 12, background: '#ffffff', border: '1px solid var(--desired-dawn)', borderRadius: 6 }}>
                  <option value="">-- Select --</option>
                  <option value="Junior">Junior</option>
                  <option value="Mid">Mid</option>
                  <option value="Senior">Senior</option>
                  <option value="Lead">Lead</option>
                  <option value="Manager">Manager</option>
                  <option value="Director">Director</option>
                  <option value="Executive">Executive</option>
                </select>
              : f.key === 'compensation'
              ? <input type="text" inputMode="decimal" readOnly value={displayValue} onClick={() => openCompModal(c.id, displayValue)} onFocus={() => openCompModal(c.id, displayValue)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') openCompModal(c.id, displayValue); }} style={{ width: '100%', boxSizing: 'border-box', padding: '4px 8px', font: 'inherit', fontSize: 12, background: '#ffffff', cursor: 'pointer' }} />
              : f.key === 'geographic'
              ? <select value={displayValue || ''} onChange={e => handleEditChange(c.id, f.key, e.target.value)} style={{ width: '100%', boxSizing: 'border-box', padding: '4px 8px', font: 'inherit', fontSize: 12, background: '#ffffff', border: '1px solid var(--desired-dawn)', borderRadius: 6 }}>
                  <option value="">-- Select Region --</option>
                  <option value="North America">North America</option>
                  <option value="South America">South America</option>
                  <option value="Western Europe">Western Europe</option>
                  <option value="Eastern Europe">Eastern Europe</option>
                  <option value="Middle East">Middle East</option>
                  <option value="Asia">Asia</option>
                  <option value="Australia/Oceania">Australia/Oceania</option>
                  <option value="Africa">Africa</option>
                </select>
              : <input type={f.type} value={displayValue} onChange={e => handleEditChange(c.id, f.key, e.target.value)} style={{ width: '100%', boxSizing: 'border-box', padding: '4px 8px', font: 'inherit', fontSize: 12, background: '#ffffff' }} />
        }
      </td>
    );
  };

  // ── DB Dock In: import a DB Port export file and deploy ──
  const S1_TO_DB_DOCK = {
    name: 'name', company: 'company', jobtitle: 'jobtitle', country: 'country',
    linkedinurl: 'linkedinurl', product: 'product', sector: 'sector',
    jobfamily: 'jobfamily', geographic: 'geographic', seniority: 'seniority',
    skillset: 'skillset', sourcingstatus: 'sourcingstatus', email: 'email',
    mobile: 'mobile', office: 'office', comment: 'comment', compensation: 'compensation',
  };

  const handleDockIn = (file, analyticMode = false) => {
    if (!file) { setDockInError('❌ No file selected. Please choose an Excel file exported via DB Port.'); return; }
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext !== 'xlsx' && ext !== 'xls' && ext !== 'xml') {
      setDockInError(`❌ Rejected: "${file.name}" is not an Excel file. DB Dock In only accepts .xlsx, .xls, or .xml (XML Spreadsheet) files exported via DB Port.`);
      return;
    }
    setDockInUploading(true);
    setDockInError('');
    setDockInAnalyticPct(0);
    if (analyticMode) setDockInAnalyticProgress('Reading file…');
    file.arrayBuffer().then(async data => {
      const wb = XLSX.read(data);
      const dbCopyName = wb.SheetNames.find(n => n === 'DB Copy');
      if (!dbCopyName) {
        setDockInError(`❌ Rejected: No "DB Copy" sheet found. This file was not exported via DB Port, has been modified in a way that removed the DB Copy sheet, may be corrupted, or was incompletely downloaded.`);
        setDockInUploading(false);
        if (analyticMode) setDockInAnalyticProgress('');
        return;
      }
      const ws2  = wb.Sheets[dbCopyName];
      const raw  = XLSX.utils.sheet_to_json(ws2, { header: 1, defval: '' });
      if (!raw.length || String(raw[0][0]).trim() !== '__json_export_v1__') {
        setDockInError('❌ Rejected: DB Copy sheet is missing the required export sentinel ("__json_export_v1__"). Only original DB Port exports are accepted — the file may have been re-saved or modified.');
        setDockInUploading(false);
        if (analyticMode) setDockInAnalyticProgress('');
        return;
      }

      // ── Signature verification ──
      const sigSheetName = wb.SheetNames.find(n => n === 'Signature');
      if (sigSheetName) {
        try {
          const sigWs  = wb.Sheets[sigSheetName];
          const sigRaw = XLSX.utils.sheet_to_json(sigWs, { header: 1, defval: '' });
          const sigB64 = String((sigRaw[0] || [])[0] || '').trim();
          const pubB64 = String((sigRaw[1] || [])[0] || '').trim();
          if (!sigB64 || !pubB64) throw new Error('Signature sheet is incomplete — one or both signature fields are missing.');
          // Reconstruct the raw content that was signed: raw JSON strings joined
          const rawJsonStrings = raw.slice(1)
            .filter(row => row[0])
            .map(row => row.filter(c => c != null).join(''));
          const rawDbContent = rawJsonStrings.join('\n');
          const valid = await verifyImportData(rawDbContent, sigB64, pubB64);
          if (!valid) {
            setDockInError('❌ Rejected: Signature verification failed. The DB Copy data does not match the original export signature — the file may have been tampered with. Only the original signed export file can be imported.');
            setDockInUploading(false);
            if (analyticMode) setDockInAnalyticProgress('');
            return;
          }
        } catch (e) {
          setDockInError('❌ Rejected: Signature verification error — ' + (e && e.message ? e.message : 'Unknown error') + '. Only the original signed DB Port export file can be imported.');
          setDockInUploading(false);
          if (analyticMode) setDockInAnalyticProgress('');
          return;
        }
      }

      const dbRows = raw.slice(1)
        .filter(row => row[0])
        .map(row => {
          const fullJson = row.filter(c => c != null && String(c) !== '').join('');
          try { return JSON.parse(fullJson); }
          catch (e) { console.warn('[DB Dock In] Failed to parse DB Copy row:', e); return null; }
        })
        .filter(c => c != null);
      if (!dbRows.length) {
        setDockInError('❌ Rejected: No valid candidate records found in the DB Copy sheet. The export file may be empty or corrupted.');
        setDockInUploading(false);
        if (analyticMode) setDockInAnalyticProgress('');
        return;
      }
      const ws1    = wb.Sheets[wb.SheetNames[0]];
      const s1Rows = XLSX.utils.sheet_to_json(ws1, { defval: '' });
      // New records are recognised exclusively from Sheet 1 (Candidate Data tab).
      // DB Copy JSON is supplemental: it fills any field absent from Sheet 1.
      // DB Copy cannot introduce records not present in Sheet 1.
      const MANDATORY_DOCK_FIELDS = ['name', 'company', 'jobtitle', 'country'];
      const merged = s1Rows.map((s1Row, i) => {
        const dbRow = dbRows[i] || {};
        const out   = { ...dbRow }; // start with DB Copy metadata as base (userid, supplemental fields, etc.)
        for (const [s1Col, dbKey] of Object.entries(S1_TO_DB_DOCK)) {
          const v = s1Row[s1Col];
          if (v !== undefined && String(v).trim() !== '') out[dbKey] = v; // Sheet 1 overrides
        }
        // For new records (no userid), remove any client-side generated id to let the server
        // auto-assign via PostgreSQL sequence. This avoids integer overflow errors when a
        // timestamp-based id (e.g. 1773586616519) exceeds the 32-bit integer column range.
        if (!out.userid) {
          delete out.id;
        }
        return out;
      });
      // In Analytic DB mode, new records missing mandatory fields are rejected and must not be imported.
      // Existing records (with a userid) are always imported regardless of mode.
      const mergedToImport = analyticMode
        ? merged.filter(cand => {
            if (cand.userid) return true; // existing record — always import
            return MANDATORY_DOCK_FIELDS.every(f => String(cand[f] || '').trim() !== ''); // new: must pass mandatory check
          })
        : merged;
      if (analyticMode) { setDockInAnalyticProgress('Deploying candidates to database…'); setDockInAnalyticPct(5); }
      fetch('http://localhost:4000/candidates/bulk', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
        body:    JSON.stringify({ candidates: mergedToImport }),
        credentials: 'include',
      })
      .then(async res => {
        if (!res.ok) throw new Error(`Server returned status ${res.status} — check server logs for details.`);
        setDockInError('');
        // ── Resume upload: upload matched PDF/DOCX resumes to the webbridge ──
        const matchedResumes = dockInResumeMatchesRef.current.filter(m => m.file);
        if (matchedResumes.length > 0) {
          try {
            if (analyticMode) { setDockInAnalyticProgress('Uploading matched resumes…'); setDockInAnalyticPct(3); }
            const formData = new FormData();
            matchedResumes.forEach(m => formData.append('files', m.file));
            await fetch('http://localhost:8091/process/upload_multiple_cvs', {
              method: 'POST',
              credentials: 'include',
              body: formData,
            });
          } catch (resumeErr) {
            console.warn('[Dock In] Resume upload failed (non-fatal):', resumeErr && resumeErr.message);
            if (analyticMode) {
              setDockInAnalyticProgress(`⚠️ Resume upload failed (${resumeErr && resumeErr.message ? resumeErr.message : 'network error'}) — continuing with analysis…`);
            } else {
              setDockInError(`⚠️ Resume upload failed: ${resumeErr && resumeErr.message ? resumeErr.message : 'network error'}. Candidate data was imported successfully.`);
            }
          }
        }
        if (analyticMode) {
          // ── Analytic DB: trigger analysis for new records that passed mandatory-field validation ──
          // mergedToImport already excludes new records missing mandatory fields (they were rejected).
          // Eligible for analysis = new (no userid) records in mergedToImport.
          const ANALYTIC_COMPLETION_DISPLAY_MS = 2200;
          const extractCandidateSkills = (cand) => {
            try {
              const vs = cand.vskillset;
              if (Array.isArray(vs)) return vs.map(s => (typeof s === 'object' ? s.skill : s)).filter(Boolean);
              if (typeof vs === 'string') {
                const arr = JSON.parse(vs);
                return Array.isArray(arr) ? arr.map(s => (typeof s === 'object' ? s.skill : s)).filter(Boolean) : [];
              }
            } catch (_) { /* ignore — fall through to skillset string */ }
            const ss = cand.skillset || '';
            return typeof ss === 'string' ? ss.split(/[,;]+/).map(s => s.trim()).filter(Boolean) : [];
          };
          // Eligible = new records that were imported (mandatory fields already enforced by mergedToImport filter)
          const eligibleForAnalysis = mergedToImport.filter(cand => !cand.userid);
          const existingCount = mergedToImport.length - eligibleForAnalysis.length;
          setDockInAnalyticProgress(`Analysing ${eligibleForAnalysis.length} eligible record(s)${existingCount > 0 ? ` (${existingCount} existing record(s) skipped)` : ''} — this may take a moment…`);
          setDockInAnalyticPct(15);
          let analysed = 0;
          let skipped = 0;
          let failed = 0;
          const eligibleRecordsCount = eligibleForAnalysis.length || 1;
          const SKILLSET_START_PCT = 15;  // progress bar % at start of skillset inference
          const SKILLSET_RANGE_PCT = 55;  // 15% → 70% for per-record skillset phase
          // Step 1: Per-record skillset inference via /vskillset/infer (15% → 70%)
          for (const cand of eligibleForAnalysis) {
            const linkedinurl = cand.linkedinurl || '';
            const skills = extractCandidateSkills(cand);
            if (!linkedinurl || skills.length === 0) { skipped++; continue; }
            try {
              await fetch('http://localhost:8091/vskillset/infer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                // assessment_level L2: full Gemini-backed evaluation (recommended for Analytic DB)
                body: JSON.stringify({ linkedinurl, skills, assessment_level: 'L2', username: cand.username || '' }),
              });
              analysed++;
            } catch (analyticErr) {
              console.warn('[Analytic DB] Skillset analysis failed for record:', cand.linkedinurl, analyticErr && analyticErr.message);
              failed++;
            }
            const done = analysed + skipped + failed;
            setDockInAnalyticProgress(`Analysing skillsets… (${done}/${eligibleForAnalysis.length})`);
            // Progress bar: 15% (start) → 70% (end of skillset pass); bulk_assess gets 70%–95%
            setDockInAnalyticPct(SKILLSET_START_PCT + Math.round((done / eligibleRecordsCount) * SKILLSET_RANGE_PCT));
          }
          // Step 2: Bulk inference to populate extended attributes (Seniority, Job Family, Sector,
          // Product, Tenure, Rating, Skillset, Geographic) via /process/bulk_assess (70% → 95%)
          const eligibleLinkedinUrls = eligibleForAnalysis
            .map(c => c.linkedinurl)
            .filter(Boolean);
          if (eligibleLinkedinUrls.length > 0) {
            try {
              setDockInAnalyticProgress('Running extended inference (rating, seniority, sector, job family…)');
              setDockInAnalyticPct(72);
              await fetch('http://localhost:8091/process/bulk_assess', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                  linkedinurls: eligibleLinkedinUrls,
                  assessment_level: 'L2',
                  async: false,
                }),
              });
              setDockInAnalyticPct(95);
            } catch (bulkErr) {
              console.warn('[Analytic DB] Bulk extended inference failed:', bulkErr && bulkErr.message);
              setDockInAnalyticPct(95);
            }
          } else {
            setDockInAnalyticPct(95);
          }
          const summaryParts = [`${analysed} record(s) analysed`];
          if (skipped > 0) summaryParts.push(`${skipped} skipped (missing URL/skills)`);
          if (failed > 0) summaryParts.push(`${failed} failed`);
          if (existingCount > 0) summaryParts.push(`${existingCount} existing record(s) not re-analysed`);
          setDockInAnalyticPct(100);
          setDockInAnalyticProgress(`Analysis complete — ${summaryParts.join(', ')}.`);
          setTimeout(() => { setDockInAnalyticProgress(''); setDockInAnalyticPct(0); setDockInWizOpen(false); onDockIn && onDockIn(); }, ANALYTIC_COMPLETION_DISPLAY_MS);
        } else {
          setDockInWizOpen(false);
          onDockIn && onDockIn();
        }
      })
      .catch(err => setDockInError('❌ Deploy failed: ' + (err && err.message ? err.message : 'Network error')))
      .finally(() => setDockInUploading(false));
    }).catch(err => {
      setDockInError('❌ Failed to read Excel file: ' + (err && err.message ? err.message : 'The file may be corrupt or not a valid Excel workbook.'));
      setDockInUploading(false);
      if (analyticMode) setDockInAnalyticProgress('');
    });
  };

  // ── Resume name matching helper ──
  // Strips extension and normalises punctuation/case before comparing.
  // Requires an exact normalized match or that one fully contains the other (min 5 chars).
  const resumeMatchesRecord = (file, candidateName) => {
    const normalize = s => s.toLowerCase().replace(/[^a-z0-9]/g, '');
    const fn = normalize(file.name.replace(/\.[^.]+$/, ''));
    const cn = normalize(candidateName);
    if (!fn || !cn) return false;
    if (fn === cn) return true;
    // Substring check: only if the shorter key is ≥5 chars to avoid false positives (e.g. 'john' in 'johnson')
    const minLen = 5;
    if (cn.length >= minLen && fn.includes(cn)) return true;
    if (fn.length >= minLen && cn.includes(fn)) return true;
    return false;
  };

  // ── Pre-parse file to count records without vskillset (new records needing Analytic DB) ──
  const peekFileForNewRecords = (file, mode) => {
    setDockInPeeking(true);
    setDockInError('');
    file.arrayBuffer().then(data => {
      try {
        const wb = XLSX.read(data);
        const dbCopyName = wb.SheetNames.find(n => n === 'DB Copy');
        if (!dbCopyName) {
          // File invalid; let handleDockIn report the proper error
          setDockInPeeking(false);
          setDockInWizStep(mode === 'analytic' ? 5 : 4);
          handleDockIn(file, mode === 'analytic');
          return;
        }
        const ws2 = wb.Sheets[dbCopyName];
        const raw = XLSX.utils.sheet_to_json(ws2, { header: 1, defval: '' });
        if (!raw.length || String(raw[0][0]).trim() !== '__json_export_v1__') {
          setDockInPeeking(false);
          setDockInWizStep(mode === 'analytic' ? 5 : 4);
          handleDockIn(file, mode === 'analytic');
          return;
        }
        // New records are recognised exclusively from Sheet 1 (Candidate Data tab).
        // DB Copy JSON is supplemental: provides userid to identify existing records
        // and fallback values for any field absent in Sheet 1.
        const ws1 = wb.Sheets[wb.SheetNames[0]];
        const s1Rows = ws1 ? XLSX.utils.sheet_to_json(ws1, { defval: '' }) : [];
        const MANDATORY = ['name', 'company', 'jobtitle', 'country'];
        let newCount = 0;
        const rejected = [];
        const newRecordsList = [];
        // Build DB Copy objects array (index-aligned with Sheet 1) for supplemental lookups
        const dbCopyObjects = raw.slice(1)
          .filter(row => row && row.length && row[0])
          .map(row => {
            const jsonStr = row.filter(c => c != null && String(c) !== '').join('');
            try { return JSON.parse(jsonStr); } catch (_) { return null; }
          })
          .filter(Boolean);
        // Iterate over Sheet 1 rows — the authoritative source of new records
        const peekIdBase = Date.now() + Math.floor(Math.random() * 10000); // unique base per peek session
        s1Rows.forEach((s1Row, i) => {
          const dbObj = dbCopyObjects[i] || {};
          // Existing record: userid is present in the corresponding DB Copy JSON entry
          if (dbObj.userid) return;
          newCount++;
          const displayName = String(s1Row['name'] || dbObj.name || `Row ${i + 2}`).trim();
          // Generate a unique numeric ID for this new record (used for tracking through the wizard
          // and passed to the server as an explicit id so it is stored in the process table).
          const tempId = peekIdBase + i;
          newRecordsList.push({ tempId, name: displayName, row: i + 2 });
          // Validate mandatory fields: Sheet 1 is authoritative; DB Copy JSON as fallback
          const missing = MANDATORY.filter(f => String(s1Row[f] || dbObj[f] || '').trim() === '');
          if (missing.length > 0) {
            // +2: row 1 is the Sheet 1 header, rows are 1-based, so data row i → spreadsheet row i+2
            rejected.push({ row: i + 2, name: displayName, missing });
          }
        });
        setDockInNewRecords(newRecordsList);
        setDockInNewRecordCount(newCount);
        setDockInRejectedRows(rejected);
        setDockInPeeking(false);
        // Extract unique (role_tag, jskillset) pairs from DB Copy for Step 3 (analytic mode)
        const pairsMap = new Map();
        dbCopyObjects.forEach(obj => {
          const rt = (obj.role_tag || '').trim();
          const js = (obj.jskillset || '').trim();
          if (rt || js) {
            const key = `${rt}||${js}`;
            if (!pairsMap.has(key)) pairsMap.set(key, { roleTag: rt, jskillset: js });
          }
        });
        const roleTagPairs = Array.from(pairsMap.values());
        setDockInRoleTagPairs(roleTagPairs);
        setDockInSelectedPair(roleTagPairs.length === 1 ? roleTagPairs[0] : null);
        if (mode === 'analytic' && newCount > 0) {
          setDockInAnalyticConfirm(true); // show token-cost confirmation dialog for analytic mode
        } else if (mode === 'analytic') {
          // No new records in analytic mode: still show role/skillset confirmation (step 3) before deploy
          setDockInWizStep(3);
        } else if (newCount > 0) {
          // Normal mode with new records: go to resume upload step
          setDockInWizStep(3);
        } else {
          // Normal mode, no new records: skip resume step and deploy directly
          setDockInWizStep(4);
          handleDockIn(file, false);
        }
      } catch (_) {
        setDockInPeeking(false);
        setDockInWizStep(mode === 'analytic' ? 5 : 4);
        handleDockIn(file, mode === 'analytic');
      }
    }).catch(() => {
      setDockInPeeking(false);
      setDockInWizStep(mode === 'analytic' ? 5 : 4);
      handleDockIn(file, mode === 'analytic');
    });
  };

  // ── DB Dock Out: export + clear user's process table data ──
  const executeDockOut = async () => {
    setDockOutConfirmOpen(false);
    await handleDbPortExport();
    setDockOutClearing(true);
    // Clear any local candidate cache
    try { localStorage.removeItem('candidatesCache'); } catch (cacheErr) { console.warn('[DB Dock Out] Failed to clear cache:', cacheErr); }
    fetch('http://localhost:4000/candidates/clear-user', {
      method: 'DELETE',
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
      credentials: 'include',
    })
    .then(res => res.ok ? res.json() : Promise.reject())
    .then(() => { onDockIn && onDockIn(); })
    .catch(err => { console.warn('[DB Dock Out] Clear-user failed (export completed):', err); /* non-fatal: export already happened */ })
    .finally(() => setDockOutClearing(false));
  };

  const handleDockOut = () => {
    if (dockOutNoWarning) {
      executeDockOut();
    } else {
      setDockOutConfirmOpen(true);
    }
  };

  // ── DB Port: Excel export — SpreadsheetML XML format (native dropdown support) ──
  const handleDbPortExport = async () => {
    // Max cell length (SpreadsheetML / OOXML spec).
    const MAX_LEN = 32767;
    const cellStr = v => {
      const s = v == null ? '' : String(v);
      return s.length > MAX_LEN ? s.slice(0, MAX_LEN) : s;
    };
    // Escape special XML characters in cell content.
    const ex = s => String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

    // Sheet 1 column definitions (user-facing)
    const S1_COLS = [
      { header: 'name',           get: c => c.name || '' },
      { header: 'company',        get: c => c.company || c.organisation || '' },
      { header: 'jobtitle',       get: c => c.role || c.jobtitle || '' },
      { header: 'country',        get: c => c.country || '' },
      { header: 'linkedinurl',    get: c => c.linkedinurl || '' },
      { header: 'product',        get: c => c.type || c.product || '' },
      { header: 'sector',         get: c => c.sector || '' },
      { header: 'jobfamily',      get: c => c.job_family || c.jobfamily || '' },
      { header: 'geographic',     get: c => c.geographic || '' },
      { header: 'seniority',      get: c => c.seniority || '' },
      { header: 'skillset',       get: c => Array.isArray(c.skillset) ? c.skillset.join(', ') : (c.skillset || '') },
      { header: 'sourcingstatus', get: c => c.sourcing_status || c.sourcingstatus || '' },
      { header: 'email',          get: c => c.email || '' },
      { header: 'mobile',         get: c => c.mobile || '' },
      { header: 'office',         get: c => c.office || '' },
      { header: 'comment',        get: c => c.comment || '' },
      { header: 'compensation',   get: c => c.compensation || '' },
    ];

    // Build header + data rows for Sheet 1
    const headerRow = `<Row>${S1_COLS.map(col => `<Cell ss:StyleID="hdr"><Data ss:Type="String">${ex(col.header)}</Data></Cell>`).join('')}</Row>`;
    const dataRows  = (allCandidates || []).map(c =>
      `<Row>${S1_COLS.map(col => `<Cell><Data ss:Type="String">${ex(cellStr(col.get(c)))}</Data></Cell>`).join('')}</Row>`
    ).join('');
    const colDefs = S1_COLS.map(col =>
      `<Column ss:Width="${['linkedinurl','skillset'].includes(col.header) ? 200 : 110}"/>`
    ).join('');

    // Data validation — inline comma-separated list values using the x: namespace prefix.
    // This is the format Excel itself generates when saving as XML Spreadsheet 2003,
    // and avoids all cross-sheet reference / named-range resolution issues.
    const maxVRows = Math.max((allCandidates || []).length + 2, 1001);
    const geoCol    = S1_COLS.findIndex(c => c.header === 'geographic')     + 1;
    const senCol    = S1_COLS.findIndex(c => c.header === 'seniority')      + 1;
    const stCol     = S1_COLS.findIndex(c => c.header === 'sourcingstatus') + 1;
    const GEO_VALS  = ['North America','South America','Western Europe','Eastern Europe','Middle East','Asia','Australia/Oceania','Africa'];
    const SEN_VALS  = ['Junior','Mid','Senior','Lead','Manager','Director','Executive'];
    const ST_VALS_FALLBACK = ['Reviewing','Contacted','Unresponsive','Declined','Unavailable','Screened','Not Proceeding','Prospected'];
    const ST_VALS   = (statusOptions || []).length ? statusOptions : ST_VALS_FALLBACK;

    // Build each DataValidation block using a per-element namespace declaration.
    // <Value> must be a SINGLE quoted string containing comma-separated items:
    //   "Item1,Item2,Item3"  — the entire list wrapped in ONE pair of double quotes.
    // "Item1","Item2","Item3" is incorrect (multiple quoted items = "Bad Value" in Excel).
    // Double quotes within an item are doubled ("") per Excel formula convention.
    // & < > are XML-encoded; double quotes are literal in XML text content.
    const xmlSafe = s => String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const makeValidation = (col1, vals) => {
      if (!col1 || !vals || !vals.length) return '';
      const inner = vals.map(v => xmlSafe(v).replace(/"/g, '""')).join(',');
      return `<DataValidation xmlns="urn:schemas-microsoft-com:office:excel">\n` +
             ` <Range>R2C${col1}:R${maxVRows}C${col1}</Range>\n` +
             ` <Type>List</Type>\n` +
             ` <Value>"${inner}"</Value>\n</DataValidation>`;
    };
    const validationXml = [
      makeValidation(geoCol, GEO_VALS),
      makeValidation(senCol, SEN_VALS),
      makeValidation(stCol,  ST_VALS),
    ].filter(Boolean).join('\n');

    // Sheet 2: full candidate JSON rows — one JSON object per row, split across
    // multiple cells when the string exceeds the 32767-char cell limit.
    // The upload handler joins all cells in each row before JSON.parse.
    // The sheet is hidden so it doesn't clutter the workbook view.
    const jsonHeaderRow = `<Row><Cell><Data ss:Type="String">__json_export_v1__</Data></Cell></Row>`;
    // Raw JSON strings (before chunking) — signed for tamper-detection
    const rawJsonStrings = (allCandidates || []).map(c => {
      try { return JSON.stringify(c); } catch { return '{}'; }
    });
    const jsonRows = rawJsonStrings.map(s => {
      const chunks = [];
      for (let i = 0; i < s.length; i += MAX_LEN) chunks.push(s.slice(i, i + MAX_LEN));
      const cells = chunks.map(ch => `<Cell><Data ss:Type="String">${ex(ch)}</Data></Cell>`).join('');
      return `<Row>${cells}</Row>`;
    }).join('');

    // Sign the DB Copy content so Dock In can verify it hasn't been tampered with.
    const rawDbContent = rawJsonStrings.join('\n');
    const { signature: sigB64, publicKey: pubB64 } = await signExportData(rawDbContent);
    const sigSheet =
`<Worksheet ss:Name="Signature" ss:Visible="SheetHidden">\n` +
` <Table>\n` +
`  <Row><Cell><Data ss:Type="String">${ex(sigB64)}</Data></Cell></Row>\n` +
`  <Row><Cell><Data ss:Type="String">${ex(pubB64)}</Data></Cell></Row>\n` +
` </Table>\n` +
` <WorksheetOptions xmlns="urn:schemas-microsoft-com:office:excel"><Visible>SheetHidden</Visible></WorksheetOptions>\n` +
`</Worksheet>\n`;

    const xml = `<?xml version="1.0"?>\n<?mso-application progid="Excel.Sheet"?>\n` +
`<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"\n` +
` xmlns:o="urn:schemas-microsoft-com:office:office"\n` +
` xmlns:x="urn:schemas-microsoft-com:office:excel"\n` +
` xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"\n` +
` xmlns:html="http://www.w3.org/TR/REC-html40">\n` +
` <Styles><Style ss:ID="hdr"><Font ss:Bold="1"/></Style></Styles>\n` +
` <Worksheet ss:Name="Candidate Data">\n` +
`  <Table ss:DefaultColumnWidth="110">${colDefs}${headerRow}${dataRows}</Table>\n` +
`  <WorksheetOptions xmlns="urn:schemas-microsoft-com:office:excel">\n` +
`   <FreezePanes/>\n` +
`   <FrozenNoSplit/>\n` +
`   <SplitHorizontal>1</SplitHorizontal>\n` +
`   <TopRowBottomPane>1</TopRowBottomPane>\n` +
`   <ActivePane>2</ActivePane>\n` +
`  </WorksheetOptions>\n` +
`  ${validationXml}\n` +
` </Worksheet>\n` +
` <Worksheet ss:Name="DB Copy" ss:Visible="SheetHidden">\n` +
`  <Table>${jsonHeaderRow}${jsonRows}</Table>\n` +
`  <WorksheetOptions xmlns="urn:schemas-microsoft-com:office:excel"><Visible>SheetHidden</Visible></WorksheetOptions>\n` +
` </Worksheet>\n` +
sigSheet +
`</Workbook>`;

    const blob = new Blob([xml], { type: 'application/vnd.ms-excel' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `db_port_${new Date().toISOString().slice(0, 10)}.xls`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  };

  // ── Step-card shared style helper for inline wizard ──
  const wizCardStyle = (selected) => ({
    flex: 1, border: `2px solid ${selected ? '#4c82b8' : '#d8d8d8'}`,
    borderRadius: 10, padding: '16px 14px', cursor: 'pointer', transition: 'border-color 0.15s',
    background: selected ? 'rgba(76,130,184,0.07)' : '#fafafa',
    position: 'relative',
  });

  return (
    <>
      {/* ── Inline DB Dock In setup wizard (shown only when candidate list is empty) ── */}
      {(allCandidates || []).length === 0 && (() => {
        const isAnalyticWiz = dockInWizMode === 'analytic';
        const totalSteps = isAnalyticWiz ? 5 : 4;
        const stepLabels = isAnalyticWiz
          ? ['Choose Mode', 'Select File', 'Role & Skills', 'Upload Resumes', 'Deploy']
          : ['Choose Mode', 'Select File', 'Upload Resumes', 'Deploy'];
        const resumeStep = isAnalyticWiz ? 4 : 3;
        const deployStep = isAnalyticWiz ? 5 : 4;
        // Helper: value-based pair comparison to avoid stale object-reference issues
        const isPairSelected = (pair) => dockInSelectedPair !== null &&
          dockInSelectedPair.roleTag === pair.roleTag && dockInSelectedPair.jskillset === pair.jskillset;
        const needsPairSelection = dockInRoleTagPairs.length > 1 && !dockInSelectedPair;
        return (
        <div className="app-card" style={{ width: '100%', maxWidth: 640, margin: '40px auto', padding: '36px 40px' }}>
          <h2 style={{ margin: '0 0 6px', color: 'var(--azure-dragon)', fontSize: 20, fontWeight: 700 }}>📥 DB Dock In — Getting Started</h2>
          <p style={{ margin: '0 0 28px', color: '#666', fontSize: 14 }}>Complete the steps below to load your candidate database.</p>

          {/* Step indicator */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: 32 }}>
            {Array.from({ length: totalSteps }, (_, i) => i + 1).map(n => (
              <React.Fragment key={n}>
                <div style={{
                  width: 28, height: 28, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 12, fontWeight: 700, flexShrink: 0,
                  background: dockInWizStep > n ? '#073679' : dockInWizStep === n ? '#4c82b8' : '#e2e8f0',
                  color: dockInWizStep >= n ? '#fff' : '#87888a',
                  border: dockInWizStep === n ? '2px solid #073679' : '2px solid transparent',
                }}>
                  {dockInWizStep > n ? '✓' : n}
                </div>
                <div style={{ fontSize: 11, color: dockInWizStep === n ? '#073679' : '#87888a', fontWeight: dockInWizStep === n ? 600 : 400, marginLeft: 5, marginRight: 0, flex: n < totalSteps ? '1 1 0' : 'none' }}>
                  {stepLabels[n - 1]}
                </div>
                {n < totalSteps && <div style={{ flex: 1, height: 2, background: dockInWizStep > n ? '#073679' : '#e2e8f0', margin: '0 6px' }} />}
              </React.Fragment>
            ))}
          </div>

          {/* Step 1 — Choose Mode */}
          {dockInWizStep === 1 && (
            <div>
              <p style={{ margin: '0 0 18px', color: '#444', fontSize: 14 }}>Select how you want to import your DB Port export:</p>
              <div style={{ display: 'flex', gap: 14, marginBottom: 24 }}>
                <div role="button" tabIndex={0} onClick={() => setDockInWizMode('normal')} onKeyDown={e => e.key === 'Enter' && setDockInWizMode('normal')} style={wizCardStyle(dockInWizMode === 'normal')}>
                  {dockInWizMode === 'normal' && <div style={{ position: 'absolute', top: 8, right: 10, color: '#4c82b8', fontWeight: 700, fontSize: 15 }}>✓</div>}
                  <div style={{ fontSize: 28, marginBottom: 8 }}>📋</div>
                  <div style={{ fontWeight: 700, color: '#073679', marginBottom: 4 }}>Normal DB Dock In</div>
                  <div style={{ fontSize: 12, color: '#666', lineHeight: 1.5 }}>Import candidate data directly. Merges with existing records using the DB Copy schema.</div>
                </div>
                <div role="button" tabIndex={0} onClick={() => setDockInWizMode('analytic')} onKeyDown={e => e.key === 'Enter' && setDockInWizMode('analytic')} style={{ ...wizCardStyle(dockInWizMode === 'analytic'), border: `2px solid ${dockInWizMode === 'analytic' ? '#073679' : '#d8d8d8'}`, background: dockInWizMode === 'analytic' ? 'rgba(7,54,121,0.07)' : '#fafafa' }}>
                  {dockInWizMode === 'analytic' && <div style={{ position: 'absolute', top: 8, right: 10, color: '#073679', fontWeight: 700, fontSize: 15 }}>✓</div>}
                  <div style={{ fontSize: 28, marginBottom: 8 }}>🤖</div>
                  <div style={{ fontWeight: 700, color: '#073679', marginBottom: 4 }}>Analytic DB</div>
                  <div style={{ fontSize: 12, color: '#666', lineHeight: 1.5, marginBottom: 8 }}>Import and run advanced AI analysis on new records. Recommended for full Consulting Dashboard functions.</div>
                  <div style={{ fontSize: 11, color: '#444', lineHeight: 1.6, background: 'rgba(7,54,121,0.05)', borderRadius: 6, padding: '6px 8px' }}>
                    <div>📊 <strong>Candidate rating</strong> per record</div>
                    <div>🧠 <strong>Inferred skillset mapping</strong></div>
                    <div>📈 <strong>Seniority analysis</strong></div>
                    <div style={{ marginTop: 4, color: '#c0392b', fontWeight: 500 }}>⚡ 1 token consumed per new record</div>
                  </div>
                </div>
              </div>
              {dockInWizMode === 'analytic' && (
                <div style={{ fontSize: 13, color: '#666', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span>Your token balance:</span>
                  <strong style={{ color: tokensLeft < 5 ? '#c0392b' : '#073679' }}>{tokensLeft} token{tokensLeft !== 1 ? 's' : ''}</strong>
                </div>
              )}
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  disabled={!dockInWizMode}
                  onClick={() => setDockInWizStep(2)}
                  style={{ padding: '10px 28px', background: dockInWizMode ? 'var(--azure-dragon)' : '#ccc', color: '#fff', border: 'none', borderRadius: 6, cursor: dockInWizMode ? 'pointer' : 'not-allowed', fontWeight: 600, fontSize: 15 }}
                >
                  Next: Select File →
                </button>
              </div>
            </div>
          )}

          {/* Step 2 — Select File */}
          {dockInWizStep === 2 && (
            <div>
              {/* Hidden file input for inline wizard */}
              <input
                type="file"
                accept=".xlsx,.xls,.xml"
                style={{ display: 'none' }}
                onChange={e => {
                  const f = e.target.files[0];
                  e.target.value = '';
                  if (f) {
                    setDockInWizFile(f);
                    setDockInAnalyticConfirm(false);
                    setDockInResumeFiles([]);
                    setDockInResumeMatches([]);
                    dockInResumeMatchesRef.current = [];
                    peekFileForNewRecords(f, dockInWizMode);
                  }
                }}
              />
              <p style={{ margin: '0 0 18px', color: '#444', fontSize: 14 }}>Choose the <strong>DB Port export file</strong> (.xlsx / .xls / .xml) to dock.</p>
              <div
                role="button" tabIndex={0}
                onClick={() => dockInInlineFileRef.current && dockInInlineFileRef.current.click()}
                onKeyDown={e => e.key === 'Enter' && dockInInlineFileRef.current && dockInInlineFileRef.current.click()}
                style={{ border: '2px dashed #4c82b8', borderRadius: 10, padding: '40px 24px', textAlign: 'center', cursor: dockInPeeking ? 'wait' : 'pointer', marginBottom: 20, background: '#f7fbff' }}
              >
                {dockInPeeking ? (
                  <>
                    <div style={{ fontSize: 36, marginBottom: 10 }}>⏳</div>
                    <div style={{ fontWeight: 600, color: '#073679' }}>Reading file…</div>
                  </>
                ) : (
                  <>
                    <div style={{ fontSize: 40, marginBottom: 10 }}>📂</div>
                    <div style={{ fontWeight: 600, color: '#073679', marginBottom: 4 }}>Click to browse for a DB Port export</div>
                    <div style={{ fontSize: 12, color: '#87888a' }}>Accepts .xlsx, .xls, and .xml (XML Spreadsheet) files</div>
                  </>
                )}
              </div>
              {dockInError && <div style={{ color: 'var(--danger)', fontSize: 13, marginBottom: 12, lineHeight: 1.5 }}>{dockInError}</div>}
              <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                <button onClick={() => { setDockInError(''); setDockInWizStep(1); }} style={{ padding: '8px 18px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}>← Back</button>
              </div>
            </div>
          )}

          {/* Step 3 (analytic) — Role & Skillset Confirmation */}
          {dockInWizStep === 3 && isAnalyticWiz && (
            <div>
              <p style={{ margin: '0 0 14px', color: '#444', fontSize: 14 }}>
                Confirm the <strong>role tag &amp; job skillset</strong> to use for bulk assessment. These are read from the DB Copy tab.
              </p>
              {dockInRoleTagPairs.length === 0 && (
                <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '12px 16px', marginBottom: 16, color: '#555', fontSize: 13 }}>
                  ⚠️ No role_tag / jskillset data found in DB Copy. The system will use your account's default configuration during assessment.
                </div>
              )}
              {dockInRoleTagPairs.length === 1 && (
                <div style={{ background: '#f0f7ff', border: '1px solid #4c82b8', borderRadius: 8, padding: '12px 16px', marginBottom: 16 }}>
                  <div style={{ fontWeight: 600, color: '#073679', fontSize: 13, marginBottom: 4 }}>✅ Confirmed pair:</div>
                  <div style={{ fontSize: 13, color: '#333' }}>
                    <strong>Role Tag:</strong> {dockInRoleTagPairs[0].roleTag || '(none)'}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <strong>Job Skillset:</strong> {dockInRoleTagPairs[0].jskillset ? dockInRoleTagPairs[0].jskillset.slice(0, 80) + (dockInRoleTagPairs[0].jskillset.length > 80 ? '…' : '') : '(none)'}
                  </div>
                </div>
              )}
              {dockInRoleTagPairs.length > 1 && (
                <div style={{ marginBottom: 16 }}>
                  <p style={{ margin: '0 0 10px', fontSize: 13, color: '#555' }}>
                    {dockInRoleTagPairs.length} unique role/skillset pair{dockInRoleTagPairs.length !== 1 ? 's' : ''} detected. Select the one to use for assessment:
                  </p>
                  {dockInRoleTagPairs.map((pair, idx) => (
                    <div
                      key={idx}
                      role="button" tabIndex={0}
                      onClick={() => setDockInSelectedPair(pair)}
                      onKeyDown={e => e.key === 'Enter' && setDockInSelectedPair(pair)}
                      style={{
                        border: `2px solid ${isPairSelected(pair) ? '#073679' : '#d8d8d8'}`,
                        borderRadius: 8, padding: '10px 14px', marginBottom: 8, cursor: 'pointer',
                        background: isPairSelected(pair) ? 'rgba(7,54,121,0.07)' : '#fafafa',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: isPairSelected(pair) ? '#073679' : '#ccc', fontWeight: 700 }}>
                          {isPairSelected(pair) ? '●' : '○'}
                        </span>
                        <div>
                          <div style={{ fontSize: 13, fontWeight: 600, color: '#073679' }}>
                            Role: {pair.roleTag || '(none)'}
                          </div>
                          <div style={{ fontSize: 12, color: '#555' }}>
                            Skillset: {pair.jskillset ? pair.jskillset.slice(0, 80) + (pair.jskillset.length > 80 ? '…' : '') : '(none)'}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <button onClick={() => { setDockInError(''); setDockInWizStep(2); }} style={{ padding: '8px 18px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}>← Back</button>
                <button
                  disabled={needsPairSelection}
                  onClick={() => setDockInWizStep(resumeStep)}
                  style={{ padding: '10px 24px', background: needsPairSelection ? '#ccc' : 'var(--azure-dragon)', color: '#fff', border: 'none', borderRadius: 6, cursor: needsPairSelection ? 'not-allowed' : 'pointer', fontWeight: 600 }}
                >
                  Confirm & Continue →
                </button>
              </div>
            </div>
          )}

          {/* Resume Upload step (step 3 in normal mode, step 4 in analytic mode) */}
          {dockInWizStep === resumeStep && (
            <div>
              {/* Hidden resume directory input for inline wizard */}
              <input
                type="file"
                accept=".pdf,.doc,.docx"
                multiple
                ref={dockInResumeInlineRef}
                style={{ display: 'none' }}
                onChange={e => {
                  const files = Array.from(e.target.files || []);
                  e.target.value = '';
                  setDockInResumeFiles(files);
                  const matches = dockInNewRecords.map(rec => ({
                    record: rec,
                    file: files.find(f => resumeMatchesRecord(f, rec.name)) || null,
                  }));
                  setDockInResumeMatches(matches);
                  dockInResumeMatchesRef.current = matches;
                }}
              />
              <p style={{ margin: '0 0 14px', color: '#444', fontSize: 14 }}>
                <strong>Upload resume files</strong> for the {dockInNewRecords.length} new record{dockInNewRecords.length !== 1 ? 's' : ''} identified.
                Files are matched to candidates by name. You can skip this step if resumes are not available.
              </p>
              <div
                role="button" tabIndex={0}
                onClick={() => dockInResumeInlineRef.current && dockInResumeInlineRef.current.click()}
                onKeyDown={e => e.key === 'Enter' && dockInResumeInlineRef.current && dockInResumeInlineRef.current.click()}
                style={{ border: '2px dashed #4c82b8', borderRadius: 10, padding: '28px 20px', textAlign: 'center', cursor: 'pointer', marginBottom: 16, background: '#f7fbff' }}
              >
                <div style={{ fontSize: 36, marginBottom: 8 }}>📎</div>
                <div style={{ fontWeight: 600, color: '#073679', marginBottom: 4 }}>Click to select resume files (PDF / DOC / DOCX)</div>
                <div style={{ fontSize: 12, color: '#87888a' }}>{dockInResumeFiles.length > 0 ? `${dockInResumeFiles.length} file(s) selected` : 'Select one or more resume files'}</div>
              </div>
              {dockInResumeMatches.length > 0 && (
                <div style={{ marginBottom: 14, background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '10px 14px' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 8 }}>Match results:</div>
                  {dockInResumeMatches.map((m, idx) => (
                    <div key={idx} style={{ fontSize: 13, color: m.file ? '#15803d' : '#6b7280', marginBottom: 3, display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span>{m.file ? '✅' : '⚪'}</span>
                      <span><strong>{m.record.name}</strong> {m.file ? `→ ${m.file.name}` : '— no match'}</span>
                    </div>
                  ))}
                </div>
              )}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <button onClick={() => { setDockInError(''); setDockInWizStep(isAnalyticWiz ? 3 : 2); }} style={{ padding: '8px 18px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}>← Back</button>
                <div style={{ display: 'flex', gap: 10 }}>
                  <button onClick={() => { setDockInResumeMatches([]); dockInResumeMatchesRef.current = []; setDockInWizStep(deployStep); handleDockIn(dockInWizFile, isAnalyticWiz); }} style={{ padding: '8px 18px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}>Skip Resumes →</button>
                  <button onClick={() => { setDockInWizStep(deployStep); handleDockIn(dockInWizFile, isAnalyticWiz); }} style={{ padding: '10px 24px', background: 'var(--azure-dragon)', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600 }}>Deploy →</button>
                </div>
              </div>
            </div>
          )}

          {/* Deploy step (step 4 in normal mode, step 5 in analytic mode) */}
          {dockInWizStep === deployStep && (
            <div style={{ textAlign: 'center' }}>
              {dockInWizFile && <p style={{ margin: '0 0 18px', color: '#444', fontSize: 14 }}>📄 <strong>{dockInWizFile.name}</strong></p>}
              {dockInUploading && (
                <div style={{ margin: '24px 0' }}>
                  <div style={{ color: '#073679', fontWeight: 600, fontSize: 15, marginBottom: 14 }}>{dockInAnalyticProgress || 'Deploying candidates to database…'}</div>
                  {isAnalyticWiz && (
                    <div style={{ width: '100%', maxWidth: 420, margin: '0 auto' }}>
                      <div style={{ background: '#e2e8f0', borderRadius: 8, height: 14, overflow: 'hidden' }}>
                        <div style={{ height: '100%', borderRadius: 8, background: 'linear-gradient(90deg, #073679, #4c82b8)', transition: 'width 0.4s ease', width: `${dockInAnalyticPct}%` }} />
                      </div>
                      <div style={{ fontSize: 12, color: '#666', marginTop: 6 }}>{dockInAnalyticPct}%</div>
                    </div>
                  )}
                  {!isAnalyticWiz && <div style={{ fontSize: 30, marginTop: 8 }}>⏳</div>}
                </div>
              )}
              {!dockInUploading && dockInAnalyticProgress && (
                <div style={{ margin: '24px 0', color: '#27ae60', fontWeight: 600, fontSize: 15 }}>✅ {dockInAnalyticProgress}</div>
              )}
              {dockInError && (
                <div style={{ color: 'var(--danger)', fontSize: 13, margin: '16px 0', lineHeight: 1.5, textAlign: 'left' }}>{dockInError}</div>
              )}
              {!dockInUploading && dockInError && (
                <div style={{ display: 'flex', gap: 10, justifyContent: 'center', marginTop: 16 }}>
                  <button onClick={() => { setDockInError(''); setDockInWizStep(2); }} style={{ padding: '8px 18px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}>← Try Another File</button>
                </div>
              )}
            </div>
          )}
        </div>
        );
      })()}

      {/* ── Normal table view (hidden when candidate list is empty) ── */}
      <div className="app-card" style={{
        width: '100%', maxWidth: '100%', position: 'relative', padding: 16,
        display: (allCandidates || []).length === 0 ? 'none' : undefined,
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
            style={{ padding: '8px 16px' }}
          >{savingAll ? 'Saving  ' : 'Save'}</button>

          {/* Right-aligned group: Configure SMTP, DB Dock In, DB Dock Out */}
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              onClick={() => setSmtpModalOpen(true)}
              className="btn-secondary"
              style={{ padding: '8px 16px' }}
            >
              Configure SMTP
            </button>

            {/* DB Dock In / DB Dock Out — replaces the old DB Port button */}
            {/* Hidden file inputs: one for legacy direct import, one for wizard */}
            <input
              type="file"
              accept=".xlsx,.xls,.xml"
              ref={dockInRef}
              style={{ display: 'none' }}
              onChange={e => { const f = e.target.files[0]; e.target.value = ''; if (f) handleDockIn(f); }}
            />
            <input
              type="file"
              accept=".xlsx,.xls,.xml"
              ref={dockInWizFileRef}
              style={{ display: 'none' }}
              onChange={e => {
                const f = e.target.files[0];
                e.target.value = '';
                if (f) {
                  setDockInWizFile(f);
                  setDockInAnalyticConfirm(false);
                  setDockInResumeFiles([]);
                  setDockInResumeMatches([]);
                  dockInResumeMatchesRef.current = [];
                  // Pre-parse for both normal and analytic modes:
                  // identifies new records, generates temp IDs, then routes to Step 3 (resume)
                  peekFileForNewRecords(f, dockInWizMode);
                }
              }}
            />
            {selectedIds.length === 0 && (
              <button
                onClick={() => {
                  setDockInWizMode('');
                  setDockInWizFile(null);
                  setDockInWizStep(1);
                  setDockInError('');
                  setDockInAnalyticProgress('');
                  setDockInWizOpen(true);
                }}
                disabled={dockInUploading}
                id="dockInBtn"
                title="Import a DB Port export file and deploy candidates"
                style={{ padding: '8px 16px', background: 'var(--cool-blue)', color: '#fff', border: 'none', borderRadius: 4, cursor: dockInUploading ? 'not-allowed' : 'pointer' }}
              >
                {dockInUploading ? 'Deploying…' : '📥 DB Dock In'}
              </button>
            )}
            {(allCandidates || []).length > 0 && (
              <button
                onClick={handleDockOut}
                disabled={dockOutClearing}
                id="dockOutBtn"
                title="Export all candidates and clear this user's data from the system"
                style={{ padding: '8px 16px', background: 'var(--azure-dragon)', color: '#fff', border: 'none', borderRadius: 4, cursor: dockOutClearing ? 'not-allowed' : 'pointer' }}
              >
                {dockOutClearing ? 'Clearing…' : '📤 DB Dock Out'}
              </button>
            )}
          </div>
          {dockInError && <div style={{ color: 'var(--danger)', fontSize: 13, marginLeft: 4 }}>{dockInError}</div>}
          
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
              <option value="Compensation">Compensation</option>
              <option value="Job Family">Job Family</option>
              <option value="Geographic">Geographic</option>
              <option value="Country">Country</option>
            </select>

            {renameCategory && (
              <>
                <input
                  type="text"
                  inputMode={renameCategory === 'Compensation' ? 'decimal' : undefined}
                  value={renameValue}
                  onChange={(e) => {
                    if (renameCategory === 'Compensation' && e.target.value !== '' && !/^\d*\.?\d*$/.test(e.target.value)) return;
                    setRenameValue(e.target.value);
                  }}
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

        {/* Search bar — collapsible, styled like SourcingVerify.html */}
        <div className="vskillset-section" style={{ marginBottom: 8 }}>
          <div className="vskillset-header" onClick={onToggleSearch} style={{ cursor: 'pointer' }}>
            <span className="vskillset-title">🔍 Search Candidates</span>
            <span className="vskillset-arrow">{searchExpanded ? '▼' : '▶'}</span>
          </div>
          {searchExpanded && (
            <div style={{ padding: '10px 12px', background: '#fff', border: '1px solid var(--neutral-border)', borderTop: 0, borderRadius: '0 0 6px 6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, position: 'relative' }}>
                <div style={{ position: 'relative', flex: 1, maxWidth: 540 }}>
                  <input
                    type="search"
                    value={globalSearchInput}
                    onChange={e => onGlobalSearchChange(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && onGlobalSearchSubmit()}
                    placeholder="Search by name, job title, company, skills…"
                    autoComplete="off"
                    style={{ width: '100%', padding: '7px 36px 7px 12px', border: '1px solid #d0d7de', borderRadius: 6, fontSize: 14, boxSizing: 'border-box', outline: 'none' }}
                    aria-label="Search candidates"
                  />
                  {globalSearchInput && (
                    <span
                      onClick={onClearSearch}
                      style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', cursor: 'pointer', color: '#57606a', fontSize: 16 }}
                      title="Clear search"
                    >✕</span>
                  )}
                </div>
                <button
                  type="button"
                  onClick={onGlobalSearchSubmit}
                  style={{ padding: '7px 16px', background: '#0969da', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 14, whiteSpace: 'nowrap' }}
                >Search</button>
                <button
                  type="button"
                  onClick={onClearSearch}
                  disabled={!globalSearchInput}
                  className="btn-secondary"
                  style={{ padding: '7px 14px', fontSize: 14, whiteSpace: 'nowrap' }}
                >Clear Search</button>
              </div>
            </div>
          )}
        </div>

        {/* Single table: checkbox+Name sticky-left, Sourcing Status+Actions sticky-right, middle scrolls */}
        {/* Middle columns can be user-pinned by clicking their header (📌 toggle) */}
        <div ref={tableRef} className="candidates-grid-wrap" style={{ overflowX: 'auto', marginBottom: 12, border: '1px solid var(--neutral-border)', borderRadius: 10, boxShadow: '0 4px 14px rgba(7,54,121,0.08)' }}>
          <table className="candidates-grid" style={{ tableLayout: 'fixed', borderCollapse: 'separate', borderSpacing: 0, overflow: 'visible', border: 0, background: 'transparent', borderRadius: 0, boxShadow: 'none' }}>
            <thead>
              {/* Row 1: column labels */}
              <tr style={{ height: HEADER_ROW_HEIGHT }}>
                <th style={{ position: 'sticky', left: 0, top: 0, zIndex: 40, width: CHECKBOX_COL_WIDTH, minWidth: CHECKBOX_COL_WIDTH, textAlign: 'center', background: '#f1f5f9', userSelect: 'none', borderRight: `1px solid ${FROZEN_EDGE_BORDER_COLOR}`, borderBottom: '1px solid var(--neutral-border)', fontFamily: 'Orbitron', height: HEADER_ROW_HEIGHT }}
                    onDoubleClick={(e) => handleHeaderDoubleClick(e, '__ALL__')}>
                  <div style={{ height: HEADER_ROW_HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <input type="checkbox" checked={candidates.length > 0 && selectedIds.length === candidates.length} onChange={handleSelectAll} style={{ cursor: 'pointer' }} />
                  </div>
                </th>
                {(() => {
                  return visibleFields.map(f => {
                    const isLeft = f.key === 'name';
                    const isRight = f.key === 'sourcing_status';
                    const isPinned = !isLeft && !isRight && frozenMiddleCols.has(f.key);
                    const maxForField = FIELD_MAX_WIDTHS[f.key] || GLOBAL_MAX_WIDTH;
                    let frozenStyle;
                    if (isLeft) {
                      frozenStyle = { position: 'sticky', left: CHECKBOX_COL_WIDTH, zIndex: 40, borderRight: `1px solid ${FROZEN_EDGE_BORDER_COLOR}`, background: '#f1f5f9' };
                    } else if (isRight) {
                      frozenStyle = { position: 'sticky', right: FROZEN_ACTIONS_WIDTH, zIndex: 40, borderLeft: `1px solid ${FROZEN_EDGE_BORDER_COLOR}`, background: '#f1f5f9' };
                    } else if (isPinned) {
                      frozenStyle = { position: 'sticky', left: computePinnedLeftOffsets[f.key], zIndex: 30, borderRight: `2px solid ${FROZEN_COL_BORDER_COLOR}`, background: '#f1f5f9' };
                    } else {
                      frozenStyle = { background: '#f1f5f9' };
                    }
                    return (
                      <th key={f.key} data-field={f.key}
                          onClick={(!isLeft && !isRight) ? () => toggleFrozenMiddleCol(f.key) : undefined}
                          onDoubleClick={(e) => handleHeaderDoubleClick(e, f.key)}
                          style={{ position: 'sticky', top: 0, zIndex: (isLeft || isRight) ? 40 : (isPinned ? 30 : 20), width: colWidths[f.key] || DEFAULT_WIDTH, minWidth: MIN_WIDTH, maxWidth: maxForField, userSelect: 'none', padding: '6px 8px 4px', verticalAlign: 'bottom', fontSize: 12, fontWeight: 700, color: 'var(--muted)', borderBottom: '1px solid var(--neutral-border)', borderRight: '1px solid var(--neutral-border)', fontFamily: 'Orbitron', cursor: (!isLeft && !isRight) ? 'pointer' : 'default', height: HEADER_ROW_HEIGHT, ...frozenStyle }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 4 }}>
                          <span className="header-label" style={{ flex: '1 1 auto' }}>{f.label}{isPinned ? ' 📌' : ''}</span>
                          <span role="separator" tabIndex={0} style={{ cursor: 'col-resize', padding: '0 4px', userSelect: 'none', height: '100%', display: 'flex', alignItems: 'center', fontSize: 14, lineHeight: 1, color: 'var(--argent)' }}
                                onMouseDown={e => { e.stopPropagation(); onMouseDown(f.key, e); }}
                                onKeyDown={e => handleResizerKey(e, f.key)}>▕</span>
                        </div>
                      </th>
                    );
                  });
                })()}
                <th style={{ position: 'sticky', right: 0, top: 0, zIndex: 40, width: FROZEN_ACTIONS_WIDTH, background: '#f1f5f9', fontSize: 12, fontWeight: 700, color: 'var(--muted)', borderBottom: '1px solid var(--neutral-border)', borderLeft: `1px solid ${FROZEN_EDGE_BORDER_COLOR}`, fontFamily: 'Orbitron', height: HEADER_ROW_HEIGHT, textAlign: 'center' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((c, idx) => {
                const rowBg = idx % 2 ? '#ffffff' : '#f9fafb';
                return (
                  <tr key={c.id} style={{ height: HEADER_ROW_HEIGHT, background: rowBg }}>
                    <td style={{ position: 'sticky', left: 0, zIndex: 10, textAlign: 'center', background: rowBg, minWidth: CHECKBOX_COL_WIDTH, width: CHECKBOX_COL_WIDTH, height: HEADER_ROW_HEIGHT, overflow: 'hidden', borderRight: `1px solid ${FROZEN_EDGE_BORDER_COLOR}` }}>
                      <input type="checkbox" checked={selectedIds.includes(c.id)} onChange={() => handleCheckboxChange(c.id)} style={{ cursor: 'pointer' }} />
                    </td>
                    {visibleFields.map(f => {
                      const isLeft = f.key === 'name';
                      const isRight = f.key === 'sourcing_status';
                      const isPinned = !isLeft && !isRight && frozenMiddleCols.has(f.key);
                      let extraStyle;
                      if (isLeft) {
                        extraStyle = { position: 'sticky', left: CHECKBOX_COL_WIDTH, zIndex: 10, borderRight: `1px solid ${FROZEN_EDGE_BORDER_COLOR}`, background: rowBg };
                      } else if (isRight) {
                        extraStyle = { position: 'sticky', right: FROZEN_ACTIONS_WIDTH, zIndex: 10, borderLeft: `1px solid ${FROZEN_EDGE_BORDER_COLOR}`, background: rowBg };
                      } else if (isPinned) {
                        extraStyle = { position: 'sticky', left: computePinnedLeftOffsets[f.key], zIndex: 5, borderRight: `2px solid ${FROZEN_COL_BORDER_COLOR}`, background: rowBg };
                      } else {
                        extraStyle = {};
                      }
                      return renderBodyCell(c, f, idx, false, extraStyle);
                    })}
                    <td style={{ position: 'sticky', right: 0, zIndex: 10, textAlign: 'center', borderBottom: '1px solid #eef2f5', borderLeft: `1px solid ${FROZEN_EDGE_BORDER_COLOR}`, height: HEADER_ROW_HEIGHT, background: rowBg, overflow: 'hidden' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                        <button onClick={() => onViewProfile && onViewProfile(c)} title="View Resume & Profile"
                                style={{ background: 'var(--azure-dragon)', color: '#fff', border: 'none', padding: '5px 8px', borderRadius: 6, cursor: 'pointer', fontSize: 11, fontWeight: 700 }}>
                          Profile
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
              {!candidates.length && (
                <tr>
                  <td colSpan={visibleFields.length + 2} style={{ padding: 16, textAlign: 'center', color: 'var(--argent)', fontSize: 14 }}>
                    No candidates match the current search.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', gap: 14, marginBottom: 4, alignItems: 'center' }}>
          <button disabled={page <= 1} onClick={() => setPage(page - 1)} className="btn-secondary" style={{ padding: '6px 14px' }}>Prev</button>
          <span style={{ fontSize: 13, color: 'var(--muted)', fontFamily: 'Orbitron' }}>Page {page} of {totalPages}</span>
          <button disabled={page >= totalPages} onClick={() => setPage(page + 1)} className="btn-secondary" style={{ padding: '6px 14px' }}>Next</button>
        </div>
      </div>
      <EmailComposeModal 
        isOpen={emailModalOpen}
        onClose={() => setEmailModalOpen(false)}
        toAddresses={composedToAddresses}
        candidateName={singleCandidateName}
        candidateData={singleCandidateData}
        userData={user}
        smtpConfig={smtpConfig}
        recipientCandidates={emailRecipients}
        onSendSuccess={handleEmailSendSuccess}
        statusOptions={statusOptions}
      />
      <SmtpConfigModal
        isOpen={smtpModalOpen}
        onClose={() => setSmtpModalOpen(false)}
        onSave={(cfg) => {
          setSmtpConfig(cfg);
          setSmtpModalOpen(false);
          // Persist to server so config survives page reloads
          fetch('http://localhost:4000/smtp-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            credentials: 'include',
            body: JSON.stringify(cfg),
          }).catch(() => {}); // ignore errors silently
        }}
        currentConfig={smtpConfig}
      />
      <CompensationCalculatorModal
        isOpen={compModalOpen}
        onClose={() => setCompModalOpen(false)}
        initialValue={compModalInitialValue}
        onSave={(total) => {
          if (compModalCandidateId != null) handleEditChange(compModalCandidateId, 'compensation', total);
        }}
      />

      {/* ── DB Dock In wizard modal ── */}
      {dockInWizOpen && (() => {
        const isAnalyticWiz = dockInWizMode === 'analytic';
        const totalSteps = isAnalyticWiz ? 5 : 4;
        const stepLabels = isAnalyticWiz
          ? ['Choose Mode', 'Select File', 'Role & Skills', 'Upload Resumes', 'Deploy']
          : ['Choose Mode', 'Select File', 'Upload Resumes', 'Deploy'];
        const resumeStep = isAnalyticWiz ? 4 : 3;
        const deployStep = isAnalyticWiz ? 5 : 4;
        // Helper: value-based pair comparison to avoid stale object-reference issues
        const isPairSelected = (pair) => dockInSelectedPair !== null &&
          dockInSelectedPair.roleTag === pair.roleTag && dockInSelectedPair.jskillset === pair.jskillset;
        const needsPairSelection = dockInRoleTagPairs.length > 1 && !dockInSelectedPair;
        return (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 9999,
          background: 'rgba(34,37,41,0.65)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            background: '#fff', borderRadius: 12, padding: '28px 32px', maxWidth: 540, width: '92%',
            boxShadow: '0 8px 40px rgba(0,0,0,0.28)',
          }}>
            {/* ── Header ── */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
              <h3 style={{ margin: 0, color: '#073679', fontSize: 18, fontWeight: 700 }}>📥 DB Dock In</h3>
              <button
                onClick={() => { if (!dockInUploading) setDockInWizOpen(false); }}
                disabled={dockInUploading}
                style={{ background: 'none', border: 'none', fontSize: 20, cursor: 'pointer', color: '#87888a', lineHeight: 1 }}
                title="Close"
              >×</button>
            </div>

            {/* ── Step indicator ── */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: 24 }}>
              {Array.from({ length: totalSteps }, (_, i) => i + 1).map(n => (
                <React.Fragment key={n}>
                  <div style={{
                    width: 26, height: 26, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 11, fontWeight: 700, flexShrink: 0,
                    background: dockInWizStep > n ? '#073679' : dockInWizStep === n ? '#4c82b8' : '#e2e8f0',
                    color: dockInWizStep >= n ? '#fff' : '#87888a',
                    border: dockInWizStep === n ? '2px solid #073679' : '2px solid transparent',
                  }}>
                    {dockInWizStep > n ? '✓' : n}
                  </div>
                  <div style={{ fontSize: 10, color: dockInWizStep === n ? '#073679' : '#87888a', fontWeight: dockInWizStep === n ? 600 : 400, marginLeft: 4, flex: n < totalSteps ? '1 1 0' : 'none', minWidth: 0 }}>
                    {stepLabels[n - 1]}
                  </div>
                  {n < totalSteps && <div style={{ flex: 1, height: 2, background: dockInWizStep > n ? '#073679' : '#e2e8f0', margin: '0 4px' }} />}
                </React.Fragment>
              ))}
            </div>

            {/* ── Step 1: Choose Mode ── */}
            {dockInWizStep === 1 && (
              <div>
                <p style={{ margin: '0 0 16px', color: '#444', fontSize: 14 }}>
                  Select how you want to import the DB Port export:
                </p>
                <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
                  {/* Normal mode card */}
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => setDockInWizMode('normal')}
                    onKeyDown={e => e.key === 'Enter' && setDockInWizMode('normal')}
                    style={{
                      flex: 1, border: `2px solid ${dockInWizMode === 'normal' ? '#4c82b8' : '#d8d8d8'}`,
                      borderRadius: 10, padding: '16px 14px', cursor: 'pointer', transition: 'border-color 0.15s',
                      background: dockInWizMode === 'normal' ? 'rgba(76,130,184,0.07)' : '#fafafa',
                      position: 'relative',
                    }}
                  >
                    {dockInWizMode === 'normal' && (
                      <div style={{ position: 'absolute', top: 8, right: 10, color: '#4c82b8', fontWeight: 700, fontSize: 15 }}>✓</div>
                    )}
                    <div style={{ fontSize: 24, marginBottom: 6 }}>📋</div>
                    <div style={{ fontWeight: 700, color: '#073679', marginBottom: 4, fontSize: 14 }}>Normal DB Dock In</div>
                    <div style={{ fontSize: 12, color: '#666', lineHeight: 1.5 }}>
                      Import candidate data directly. Merges with existing records using the DB Copy schema.
                    </div>
                  </div>
                  {/* Analytic DB card */}
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => setDockInWizMode('analytic')}
                    onKeyDown={e => e.key === 'Enter' && setDockInWizMode('analytic')}
                    style={{
                      flex: 1, border: `2px solid ${dockInWizMode === 'analytic' ? '#073679' : '#d8d8d8'}`,
                      borderRadius: 10, padding: '16px 14px', cursor: 'pointer', transition: 'border-color 0.15s',
                      background: dockInWizMode === 'analytic' ? 'rgba(7,54,121,0.07)' : '#fafafa',
                      position: 'relative',
                    }}
                  >
                    {dockInWizMode === 'analytic' && (
                      <div style={{ position: 'absolute', top: 8, right: 10, color: '#073679', fontWeight: 700, fontSize: 15 }}>✓</div>
                    )}
                    <div style={{ fontSize: 24, marginBottom: 6 }}>🤖</div>
                    <div style={{ fontWeight: 700, color: '#073679', marginBottom: 4, fontSize: 14 }}>Analytic DB</div>
                    <div style={{ fontSize: 12, color: '#666', lineHeight: 1.5, marginBottom: 8 }}>
                      Import and run advanced analysis on all records. Recommended for full Consulting Dashboard functions.
                    </div>
                    <div style={{ fontSize: 11, color: '#444', lineHeight: 1.6, background: 'rgba(7,54,121,0.05)', borderRadius: 6, padding: '6px 8px' }}>
                      <div>📊 <strong>Candidate rating</strong> per record</div>
                      <div>🧠 <strong>Inferred skillset mapping</strong></div>
                      <div>📈 <strong>Seniority analysis</strong></div>
                      <div style={{ marginTop: 4, color: '#c0392b', fontWeight: 500 }}>
                        ⚡ 1 token consumed per analysed record
                      </div>
                    </div>
                  </div>
                </div>
                {dockInWizMode === 'analytic' && (
                  <div style={{ fontSize: 12, color: '#666', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span>Your balance:</span>
                    <strong style={{ color: tokensLeft < 5 ? '#c0392b' : '#073679' }}>{tokensLeft} token{tokensLeft !== 1 ? 's' : ''}</strong>
                  </div>
                )}
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
                  <button
                    onClick={() => setDockInWizOpen(false)}
                    style={{ padding: '8px 18px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}
                  >Cancel</button>
                  <button
                    disabled={!dockInWizMode}
                    onClick={() => setDockInWizStep(2)}
                    style={{
                      padding: '8px 20px', background: dockInWizMode ? 'var(--azure-dragon)' : '#ccc',
                      color: '#fff', border: 'none', borderRadius: 6,
                      cursor: dockInWizMode ? 'pointer' : 'not-allowed', fontWeight: 600,
                    }}
                  >
                    Next: Select File →
                  </button>
                </div>
              </div>
            )}

            {/* ── Step 2: File Selection ── */}
            {dockInWizStep === 2 && (
              <div>
                <p style={{ margin: '0 0 16px', color: '#444', fontSize: 14 }}>
                  Choose the <strong>DB Port export file</strong> (.xlsx / .xls / .xml) to dock.
                </p>
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => dockInWizFileRef.current && dockInWizFileRef.current.click()}
                  onKeyDown={e => e.key === 'Enter' && dockInWizFileRef.current && dockInWizFileRef.current.click()}
                  style={{
                    border: '2px dashed #4c82b8', borderRadius: 10, padding: '32px 24px',
                    textAlign: 'center', cursor: 'pointer', marginBottom: 18, background: '#f7fbff',
                  }}
                >
                  <div style={{ fontSize: 36, marginBottom: 8 }}>📂</div>
                  <div style={{ fontWeight: 600, color: '#073679', marginBottom: 4 }}>Click to browse for a DB Port export</div>
                  <div style={{ fontSize: 12, color: '#87888a' }}>Accepts .xlsx, .xls, and .xml (XML Spreadsheet) files</div>
                </div>
                {dockInWizFile && (
                  <div style={{ fontSize: 13, color: '#444', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
                    📄 <strong>{dockInWizFile.name}</strong>
                  </div>
                )}
                {dockInError && <div style={{ color: 'var(--danger)', fontSize: 13, marginBottom: 12, lineHeight: 1.5 }}>{dockInError}</div>}
                <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                  <button onClick={() => { setDockInError(''); setDockInWizStep(1); }} style={{ padding: '8px 18px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}>← Back</button>
                </div>
              </div>
            )}

            {/* ── Step 3 (Analytic): Role & Skillset Confirmation ── */}
            {dockInWizStep === 3 && isAnalyticWiz && (
              <div>
                <p style={{ margin: '0 0 14px', color: '#444', fontSize: 14 }}>
                  Confirm the <strong>role tag &amp; job skillset</strong> to use for bulk assessment, read from your DB Copy tab.
                </p>
                {dockInRoleTagPairs.length === 0 && (
                  <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '12px 16px', marginBottom: 16, color: '#555', fontSize: 13 }}>
                    ⚠️ No role_tag / jskillset data found in DB Copy. The system will use your account's default configuration.
                  </div>
                )}
                {dockInRoleTagPairs.length === 1 && (
                  <div style={{ background: '#f0f7ff', border: '1px solid #4c82b8', borderRadius: 8, padding: '12px 16px', marginBottom: 16 }}>
                    <div style={{ fontWeight: 600, color: '#073679', fontSize: 13, marginBottom: 4 }}>✅ Confirmed pair:</div>
                    <div style={{ fontSize: 13, color: '#333' }}>
                      <strong>Role Tag:</strong> {dockInRoleTagPairs[0].roleTag || '(none)'}
                      &nbsp;&nbsp;|&nbsp;&nbsp;
                      <strong>Job Skillset:</strong> {dockInRoleTagPairs[0].jskillset ? dockInRoleTagPairs[0].jskillset.slice(0, 80) + (dockInRoleTagPairs[0].jskillset.length > 80 ? '…' : '') : '(none)'}
                    </div>
                  </div>
                )}
                {dockInRoleTagPairs.length > 1 && (
                  <div style={{ marginBottom: 16 }}>
                    <p style={{ margin: '0 0 10px', fontSize: 13, color: '#555' }}>
                      {dockInRoleTagPairs.length} unique pairs detected — select one for assessment:
                    </p>
                    {dockInRoleTagPairs.map((pair, idx) => (
                      <div
                        key={idx}
                        role="button" tabIndex={0}
                        onClick={() => setDockInSelectedPair(pair)}
                        onKeyDown={e => e.key === 'Enter' && setDockInSelectedPair(pair)}
                        style={{
                          border: `2px solid ${isPairSelected(pair) ? '#073679' : '#d8d8d8'}`,
                          borderRadius: 8, padding: '8px 12px', marginBottom: 6, cursor: 'pointer',
                          background: isPairSelected(pair) ? 'rgba(7,54,121,0.07)' : '#fafafa',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ color: isPairSelected(pair) ? '#073679' : '#ccc', fontWeight: 700 }}>
                            {isPairSelected(pair) ? '●' : '○'}
                          </span>
                          <div>
                            <div style={{ fontSize: 13, fontWeight: 600, color: '#073679' }}>Role: {pair.roleTag || '(none)'}</div>
                            <div style={{ fontSize: 12, color: '#555' }}>Skillset: {pair.jskillset ? pair.jskillset.slice(0, 80) + (pair.jskillset.length > 80 ? '…' : '') : '(none)'}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <button onClick={() => setDockInWizStep(2)} style={{ padding: '8px 16px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500, fontSize: 13 }}>← Back</button>
                  <button
                    disabled={needsPairSelection}
                    onClick={() => setDockInWizStep(resumeStep)}
                    style={{ padding: '8px 18px', background: needsPairSelection ? '#ccc' : 'var(--azure-dragon)', color: '#fff', border: 'none', borderRadius: 6, cursor: needsPairSelection ? 'not-allowed' : 'pointer', fontWeight: 600, fontSize: 13 }}
                  >
                    Confirm & Continue →
                  </button>
                </div>
              </div>
            )}

            {/* ── Resume Upload step (step 3 normal / step 4 analytic) ── */}
            {dockInWizStep === resumeStep && (
              <div>
                {/* Hidden resume input for modal wizard */}
                <input
                  type="file"
                  accept=".pdf,.doc,.docx"
                  multiple
                  ref={dockInResumeModalRef}
                  style={{ display: 'none' }}
                  onChange={e => {
                    const files = Array.from(e.target.files || []);
                    e.target.value = '';
                    setDockInResumeFiles(files);
                    const matches = dockInNewRecords.map(rec => ({
                      record: rec,
                      file: files.find(f => resumeMatchesRecord(f, rec.name)) || null,
                    }));
                    setDockInResumeMatches(matches);
                    dockInResumeMatchesRef.current = matches;
                  }}
                />
                <p style={{ margin: '0 0 12px', color: '#444', fontSize: 14 }}>
                  <strong>Upload resume files</strong> for the {dockInNewRecords.length} new record{dockInNewRecords.length !== 1 ? 's' : ''} identified.
                  Files are matched to candidates by name.
                </p>
                <div
                  role="button" tabIndex={0}
                  onClick={() => dockInResumeModalRef.current && dockInResumeModalRef.current.click()}
                  onKeyDown={e => e.key === 'Enter' && dockInResumeModalRef.current && dockInResumeModalRef.current.click()}
                  style={{ border: '2px dashed #4c82b8', borderRadius: 10, padding: '24px 20px', textAlign: 'center', cursor: 'pointer', marginBottom: 14, background: '#f7fbff' }}
                >
                  <div style={{ fontSize: 30, marginBottom: 6 }}>📎</div>
                  <div style={{ fontWeight: 600, color: '#073679', marginBottom: 4, fontSize: 13 }}>Click to select resume files (PDF / DOC / DOCX)</div>
                  <div style={{ fontSize: 12, color: '#87888a' }}>{dockInResumeFiles.length > 0 ? `${dockInResumeFiles.length} file(s) selected` : 'Select one or more resume files'}</div>
                </div>
                {dockInResumeMatches.length > 0 && (
                  <div style={{ marginBottom: 12, background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 7, padding: '8px 12px', maxHeight: 140, overflowY: 'auto' }}>
                    {dockInResumeMatches.map((m, idx) => (
                      <div key={idx} style={{ fontSize: 12, color: m.file ? '#15803d' : '#6b7280', marginBottom: 2, display: 'flex', alignItems: 'center', gap: 5 }}>
                        <span>{m.file ? '✅' : '⚪'}</span>
                        <span><strong>{m.record.name}</strong> {m.file ? `→ ${m.file.name}` : '— no match'}</span>
                      </div>
                    ))}
                  </div>
                )}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <button onClick={() => setDockInWizStep(isAnalyticWiz ? 3 : 2)} style={{ padding: '8px 16px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500, fontSize: 13 }}>← Back</button>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button onClick={() => { setDockInResumeMatches([]); dockInResumeMatchesRef.current = []; setDockInWizStep(deployStep); handleDockIn(dockInWizFile, isAnalyticWiz); }} style={{ padding: '8px 14px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500, fontSize: 13 }}>Skip →</button>
                    <button onClick={() => { setDockInWizStep(deployStep); handleDockIn(dockInWizFile, isAnalyticWiz); }} style={{ padding: '8px 18px', background: 'var(--azure-dragon)', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600, fontSize: 13 }}>Deploy →</button>
                  </div>
                </div>
              </div>
            )}

            {/* ── Deploy step (step 4 normal / step 5 analytic) ── */}
            {dockInWizStep === deployStep && (
              <div>
                <p style={{ margin: '0 0 10px', color: '#444', fontSize: 14 }}>
                  {dockInWizFile ? (
                    <span>📄 <strong>{dockInWizFile.name}</strong></span>
                  ) : 'Deploying…'}
                </p>
                {dockInUploading && (
                  <div style={{ margin: '18px 0', textAlign: 'center' }}>
                    <div style={{ color: '#073679', fontWeight: 600, fontSize: 14, marginBottom: 12 }}>
                      {dockInAnalyticProgress || 'Deploying candidates to database…'}
                    </div>
                    {isAnalyticWiz && (
                      <div style={{ width: '100%', maxWidth: 360, margin: '0 auto' }}>
                        <div style={{ background: '#e2e8f0', borderRadius: 8, height: 12, overflow: 'hidden' }}>
                          <div style={{ height: '100%', borderRadius: 8, background: 'linear-gradient(90deg, #073679, #4c82b8)', transition: 'width 0.4s ease', width: `${dockInAnalyticPct}%` }} />
                        </div>
                        <div style={{ fontSize: 12, color: '#666', marginTop: 5 }}>{dockInAnalyticPct}%</div>
                      </div>
                    )}
                    {!isAnalyticWiz && <div style={{ fontSize: 28, marginTop: 8 }}>⏳</div>}
                  </div>
                )}
                {!dockInUploading && dockInAnalyticProgress && (
                  <div style={{ margin: '18px 0', textAlign: 'center', color: '#27ae60', fontWeight: 600, fontSize: 14 }}>
                    ✅ {dockInAnalyticProgress}
                  </div>
                )}
                {dockInError && (
                  <div style={{ color: 'var(--danger)', fontSize: 13, margin: '12px 0', lineHeight: 1.5 }}>{dockInError}</div>
                )}
                {!dockInUploading && dockInError && (
                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 14 }}>
                    <button
                      onClick={() => setDockInWizStep(2)}
                      style={{ padding: '8px 18px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}
                    >← Try Another File</button>
                    <button
                      onClick={() => setDockInWizOpen(false)}
                      style={{ padding: '8px 18px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}
                    >Close</button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        );
      })()}

      {/* ── DB Dock Out confirmation dialog ── */}
      {dockOutConfirmOpen && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 9999,
          background: 'rgba(34,37,41,0.65)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            background: '#fff', borderRadius: 10, padding: '28px 32px', maxWidth: 480, width: '90%',
            boxShadow: '0 8px 32px rgba(0,0,0,0.22)',
          }}>
            <h3 style={{ margin: '0 0 12px', color: '#222529', fontSize: 18 }}>⚠️ DB Dock Out — Confirm</h3>
            <p style={{ margin: '0 0 10px', lineHeight: 1.6, color: '#444' }}>
              All candidate data for your account will be <strong>permanently deleted</strong> from the system after export.
            </p>
            <p style={{ margin: '0 0 16px', lineHeight: 1.6, color: '#c0392b', fontWeight: 500 }}>
              Do not lose the exported file — only the original signed export can be re-imported via DB Dock In.
            </p>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20, cursor: 'pointer', fontSize: 14 }}>
              <input
                type="checkbox"
                checked={dockOutNoWarning}
                onChange={e => {
                  const v = e.target.checked;
                  setDockOutNoWarning(v);
                  if (v) localStorage.setItem('dockOutSkipWarning', '1');
                  else localStorage.removeItem('dockOutSkipWarning');
                }}
              />
              Don't show this warning again
            </label>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
              <button
                onClick={() => setDockOutConfirmOpen(false)}
                style={{ padding: '8px 20px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}
              >
                Return
              </button>
              <button
                onClick={executeDockOut}
                style={{ padding: '8px 20px', background: 'var(--azure-dragon)', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600 }}
              >
                Proceed with Dock Out
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Analytic DB — token cost confirmation dialog ── */}
      {dockInAnalyticConfirm && (() => {
        const validNewCount = dockInNewRecordCount - dockInRejectedRows.length;
        const rejectedCount = dockInRejectedRows.length;
        return (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 10000,
          background: 'rgba(34,37,41,0.65)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            background: '#fff', borderRadius: 12, padding: '28px 32px', maxWidth: 500, width: '92%',
            boxShadow: '0 8px 40px rgba(0,0,0,0.28)', maxHeight: '85vh', overflowY: 'auto',
          }}>
            <h3 style={{ margin: '0 0 14px', color: '#073679', fontSize: 17, fontWeight: 700 }}>🤖 Analytic DB — Confirm Analysis</h3>
            <p style={{ margin: '0 0 10px', lineHeight: 1.6, color: '#444', fontSize: 14 }}>
              <strong>{dockInNewRecordCount}</strong> new record{dockInNewRecordCount !== 1 ? 's' : ''} (rows without a user ID) were found in this file.
            </p>
            {rejectedCount > 0 && (
              <div style={{ margin: '0 0 12px', background: '#fff8e1', border: '1px solid #f0c040', borderRadius: 7, padding: '10px 14px' }}>
                <p style={{ margin: '0 0 6px', fontWeight: 600, color: '#8a6000', fontSize: 13 }}>
                  ⚠️ {rejectedCount} row{rejectedCount !== 1 ? 's' : ''} rejected — missing mandatory fields (will NOT be imported):
                </p>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {dockInRejectedRows.slice(0, 10).map((r, idx) => (
                    <li key={idx} style={{ fontSize: 12, color: '#555', marginBottom: 2 }}>
                      Row {r.row}: <strong>{r.name}</strong> — missing: {r.missing.join(', ')}
                    </li>
                  ))}
                  {dockInRejectedRows.length > 10 && (
                    <li style={{ fontSize: 12, color: '#888' }}>…and {dockInRejectedRows.length - 10} more</li>
                  )}
                </ul>
              </div>
            )}
            <p style={{ margin: '0 0 10px', lineHeight: 1.6, color: '#444', fontSize: 14 }}>
              <strong>{validNewCount}</strong> eligible record{validNewCount !== 1 ? 's' : ''} will be analysed (candidate rating, skillset mapping, seniority analysis), consuming <strong>1 token each</strong>.
            </p>
            <p style={{ margin: '0 0 20px', lineHeight: 1.6, fontSize: 14 }}>
              <span style={{ color: '#c0392b', fontWeight: 600 }}>Total token cost: {validNewCount} token{validNewCount !== 1 ? 's' : ''}</span>
              {' '}(current balance: <strong style={{ color: tokensLeft < validNewCount ? '#c0392b' : '#073679' }}>{tokensLeft}</strong>)
            </p>
            {tokensLeft < validNewCount && (
              <p style={{ margin: '0 0 16px', color: '#c0392b', fontWeight: 500, fontSize: 13 }}>
                ⚠️ Insufficient tokens. You have {tokensLeft} token{tokensLeft !== 1 ? 's' : ''} but need {validNewCount}. The import will proceed but analysis will be partial.
              </p>
            )}
            <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
              <button
                onClick={() => { setDockInAnalyticConfirm(false); setDockInWizStep(2); setDockInWizFile(null); setDockInRejectedRows([]); }}
                style={{ padding: '8px 20px', background: '#e2e8f0', color: '#333', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 500 }}
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setDockInAnalyticConfirm(false);
                  setDockInWizStep(3); // → Role & Skillset Confirmation (analytic mode step 3)
                }}
                style={{ padding: '8px 20px', background: 'var(--azure-dragon)', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600 }}
              >
                Proceed with Analysis
              </button>
            </div>
          </div>
        </div>
        );
      })()}
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
          jobtitle:(p.jobtitle||'').trim(),
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

      /* NodeCard: show job title directly from process table jobtitle field; seniority shown only in badge */
      const NodeCard=({node})=>{
        // Title: use jobtitle from process table directly, then fallback to personal, roleTag, raw.role
        const title = (node.jobtitle||'').trim()
          || (node.roleTag||'').trim()
          || (node.raw?.role ? String(node.raw.role).trim() : '')
          || '';

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
  }, [candidates, draggingId, editingLayout, manualParentOverrides, pruneOverrides, setManualParentOverrides]);

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
    const id = requestAnimationFrame(adjustCentering);
    return ()=> cancelAnimationFrame(id);
  },[orgChart, adjustCentering, editingLayout, draggingId]);

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
    // Target only the org chart tree content, not the toolbar/buttons
    const treeEl = chartRef.current.querySelector('#org-chart-content') || chartRef.current;
    // Expand treeEl itself plus all containers that may clip the chart
    const clippedElems = Array.from(treeEl.querySelectorAll(
      '.org-chart-scroll,.org-tree-root,.org-center-wrapper,.org-group,.org,.org li'
    ));
    const allElems = [treeEl, ...clippedElems];
    const originals = allElems.map(el => ({
      el,
      overflow: el.style.overflow,
      overflowX: el.style.overflowX,
      overflowY: el.style.overflowY,
      width: el.style.width,
      height: el.style.height,
      maxWidth: el.style.maxWidth,
      maxHeight: el.style.maxHeight
    }));
    try{
      allElems.forEach(el=>{
        el.style.overflow='visible';
        el.style.overflowX='visible';
        el.style.overflowY='visible';
        el.style.maxWidth='none';
        el.style.maxHeight='none';
      });
      // For .org-chart-scroll elements, also set explicit pixel dimensions
      clippedElems.filter(el=>el.classList.contains('org-chart-scroll')).forEach(el=>{
        const sw=el.scrollWidth;
        const sh=el.scrollHeight;
        if(sw>el.clientWidth) el.style.width=sw+'px';
        if(sh>el.clientHeight) el.style.height=sh+'px';
      });
      // Wait for fonts and images to finish loading before capturing
      await document.fonts.ready;
      const imgs=Array.from(treeEl.querySelectorAll('img'));
      await Promise.all(imgs.map(img=>img.complete ? Promise.resolve() : new Promise(r=>{ img.onload=r; img.onerror=r; })));
      // Two rAF frames for layout to fully settle after overflow expansion
      await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
      const fullWidth=treeEl.scrollWidth;
      const fullHeight=treeEl.scrollHeight;
      const canvas=await html2canvas(treeEl,{
        backgroundColor:'#ffffff',
        useCORS:true,
        allowTaint:true,
        foreignObjectRendering:false,
        logging:false,
        imageTimeout:0,
        scale: (typeof window !== 'undefined' ? window.devicePixelRatio : 1) || 1,
        width:fullWidth,
        height:fullHeight,
        scrollX: 0,
        scrollY: 0
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
        o.el.style.overflowX=o.overflowX;
        o.el.style.overflowY=o.overflowY;
        o.el.style.width=o.width;
        o.el.style.height=o.height;
        o.el.style.maxWidth=o.maxWidth;
        o.el.style.maxHeight=o.maxHeight;
      });
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
          <button
            onClick={()=>{
              if(editingLayout){
                setEditingLayout(false); setDraggingId(null);
              } else { setEditingLayout(true); }
            }}
            style={{
              background: editingLayout ? 'var(--argent,#87888a)' : 'var(--azure-dragon,#073679)',
              color:'#fff', border:'none', padding:'6px 14px',
              borderRadius:6, cursor:'pointer', fontWeight:700, fontSize:13
            }}
          >
            {editingLayout ? 'Finish Editing' : 'Edit Layout'}
          </button>
          <button
            onClick={handleSaveLayout}
            disabled={!unsavedChanges}
            style={{
              background: !unsavedChanges ? 'var(--desired-dawn,#d8d8d8)' : 'var(--cool-blue,#4c82b8)',
              color: !unsavedChanges ? '#87888a' : '#fff',
              border: `1px solid ${!unsavedChanges ? '#c8c8c8' : 'var(--azure-dragon,#073679)'}`,
              padding:'6px 14px', borderRadius:6,
              cursor: !unsavedChanges ? 'not-allowed':'pointer', fontWeight:700, fontSize:13
            }}
          >Save Layout</button>
          <button
            onClick={handleCancelLayout}
            disabled={!unsavedChanges}
            style={{
              background: !unsavedChanges ? 'var(--desired-dawn,#d8d8d8)' : '#fff',
              color: !unsavedChanges ? '#87888a' : 'var(--azure-dragon,#073679)',
              border: `1px solid ${!unsavedChanges ? '#c8c8c8' : 'var(--cool-blue,#4c82b8)'}`,
              padding:'6px 14px', borderRadius:6,
              cursor: !unsavedChanges ? 'not-allowed':'pointer', fontWeight:700, fontSize:13
            }}
          >Cancel</button>
          <button
            onClick={handleResetManual}
            disabled={!Object.keys(manualParentOverrides||{}).length}
            style={{
              background: !Object.keys(manualParentOverrides||{}).length ? 'var(--desired-dawn,#d8d8d8)' : '#fff',
              color: !Object.keys(manualParentOverrides||{}).length ? '#87888a' : '#c0392b',
              border: `1px solid ${!Object.keys(manualParentOverrides||{}).length ? '#c8c8c8' : '#c0392b'}`,
              padding:'6px 14px', borderRadius:6,
              cursor: !Object.keys(manualParentOverrides||{}).length ? 'not-allowed':'pointer', fontWeight:700, fontSize:13
            }}
          >Reset Manual</button>
          <button
            onClick={handleGenerateChart}
            disabled={loading}
            style={{
              background: loading ? 'var(--desired-dawn,#d8d8d8)' : 'var(--azure-dragon,#073679)',
              color: loading ? '#87888a' : '#fff',
              border:'none',
              padding:'6px 14px', borderRadius:6, cursor:loading?'not-allowed':'pointer', fontWeight:700, fontSize:13
            }}
          >{loading ? 'Regenerating...' : 'Regenerate'}</button>
          {orgChart.length>0 && (
            <>
              <button
                onClick={handleDownload}
                style={{
                  background:'var(--cool-blue,#4c82b8)', color:'#fff',
                  border:'1px solid var(--azure-dragon,#073679)',
                  padding:'6px 14px', borderRadius:6, cursor:'pointer', fontWeight:700, fontSize:13
                }}
              >Export PNG</button>
              <button
                onClick={handlePrint}
                style={{
                  background:'#fff', color:'var(--azure-dragon,#073679)',
                  border:'1px solid var(--cool-blue,#4c82b8)',
                  padding:'6px 14px', borderRadius:6, cursor:'pointer', fontWeight:700, fontSize:13
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
            style={{
              padding:'6px 10px',
              borderRadius:6,
              border:'1px solid var(--cool-blue,#4c82b8)',
              background:'#fff',
              cursor:'pointer', fontSize:13
            }}
        >
          {jobFamilyOptions.map(jf=> <option key={jf} value={jf}>{jf}</option>)}
        </select>

        <label htmlFor="organisation-dropdown" style={{ fontWeight:700, color: 'var(--muted)' }}>Organisation</label>
        <select
          id="organisation-dropdown"
          value={selectedOrganisation}
          onChange={e=>onChangeOrganisation(e.target.value)}
          style={{
            padding:'6px 10px',
            borderRadius:6,
            border:'1px solid var(--cool-blue,#4c82b8)',
            background:'#fff',
            cursor:'pointer', fontSize:13
          }}
        >
          {organisationOptions.map(opt=> <option key={opt} value={opt}>{opt}</option>)}
        </select>

        <label htmlFor="country-dropdown" style={{ fontWeight:700, color: 'var(--muted)' }}>Country</label>
        <select
          id="country-dropdown"
          value={selectedCountry}
          onChange={e=>onChangeCountry(e.target.value)}
          style={{
            padding:'6px 10px',
            borderRadius:6,
            border:'1px solid var(--cool-blue,#4c82b8)',
            background:'#fff',
            cursor:'pointer', fontSize:13
          }}
        >
          {countryOptions.map(opt=> <option key={opt} value={opt}>{opt}</option>)}
        </select>

        <span style={{ fontSize:12, color:'#64748b' }}>
          {editingLayout ? 'Drag to re-parent (drop on Make Root to promote)' : 'Click Edit Layout to enable dragging.'}
        </span>
        {unsavedChanges && <span style={{ fontSize:12, color:'#dc2626', fontWeight:600 }}>Unsaved changes</span>}
      </div>

      <div id="org-chart-content" style={{ marginTop:12 }}>
        {orgChart.length ? orgChart : <span style={{ color:'#64748b' }}>No org chart generated yet.</span>}
      </div>
    </div>
  );
}

/* ========================= UPLOAD ========================= */
function CandidateUpload({ onUpload }) {
  const [file,setFile] = useState(null);
  const [uploading,setUploading] = useState(false);
  const [error,setError] = useState('');
  const [expanded, setExpanded] = useState(false);

  // Exact process-table column names (in the order defined in the DB schema).
  // Uploads are accepted only if the spreadsheet headers exactly match these names.
  const UPLOAD_FIELDS = [
    'id', 'name', 'company', 'jobtitle', 'country', 'linkedinurl', 'username', 'userid',
    'product', 'sector', 'jobfamily', 'geographic', 'seniority', 'skillset', 'sourcingstatus',
    'email', 'mobile', 'office', 'role_tag', 'experience', 'cv', 'education', 'exp', 'rating',
    'pic', 'tenure', 'comment', 'vskillset', 'compensation', 'lskillset', 'jskillset',
    'rating_level', 'rating_updated_at', 'rating_version', 'personal'
  ];

  const validateUploadHeaders = (headers) => {
    if (!headers.includes('id')) {
      return 'Upload rejected: the "id" column is required but was not found in the file.';
    }
    if (!headers.includes('userid') && !headers.includes('username')) {
      return 'Upload rejected: the file must contain either a "userid" or "username" column.';
    }
    return null;
  };

  const mapRow = (row) => {
    const out = {};
    for (const f of UPLOAD_FIELDS) {
      if (Object.prototype.hasOwnProperty.call(row, f)) {
        const v = row[f];
        if (v != null && String(v).trim() !== '') out[f] = v;
      }
    }
    if (out.vskillset && typeof out.vskillset === 'string') {
      try { out.vskillset = JSON.parse(out.vskillset); }
      catch (e) { console.warn('[parseRow] Failed to parse vskillset:', out.vskillset, e); out.vskillset = null; }
    }
    return out;
  };

  const handleFileChange = e => { setFile(e.target.files[0]); setError(''); };

  // S1 column names (from Candidate Data tab) mapped to the field name used in the
  // DB Copy JSON — used to overlay Sheet 1 editable values on top of DB Copy metadata.
  const S1_TO_DB = {
    name: 'name', company: 'company', jobtitle: 'jobtitle', country: 'country',
    linkedinurl: 'linkedinurl', product: 'product', sector: 'sector',
    jobfamily: 'jobfamily', geographic: 'geographic', seniority: 'seniority',
    skillset: 'skillset', sourcingstatus: 'sourcingstatus', email: 'email',
    mobile: 'mobile', office: 'office', comment: 'comment', compensation: 'compensation',
  };

  const handleUpload = () => {
    if (!file) { setError('Please select an Excel file exported via DB Port.'); return; }
    setUploading(true);
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext !== 'xlsx' && ext !== 'xls' && ext !== 'xml') {
      setError('DB Dock & Deploy only accepts Excel files (.xlsx / .xls / .xml) exported via DB Port.');
      setUploading(false);
      return;
    }
    file.arrayBuffer().then(data => {
      const wb = XLSX.read(data);

      // ── Require DB Copy sheet with __json_export_v1__ sentinel ───────────────
      const dbCopyName = wb.SheetNames.find(n => n === 'DB Copy');
      if (!dbCopyName) {
        setError('This file was not exported via DB Port. Please use a DB Port export (must contain a "DB Copy" sheet).');
        setUploading(false);
        return;
      }
      const ws2  = wb.Sheets[dbCopyName];
      const raw  = XLSX.utils.sheet_to_json(ws2, { header: 1, defval: '' });
      if (!raw.length || String(raw[0][0]).trim() !== '__json_export_v1__') {
        setError('DB Copy sheet is missing or unrecognised. Only DB Port exports are accepted.');
        setUploading(false);
        return;
      }

      // Parse DB Copy rows → provides id, userid, vskillset, jskillset, experience, etc.
      // JSON may be chunked across multiple cells (each ≤32767 chars) — join all
      // non-empty cells in the row before parsing to reconstruct the full string.
      const dbRows = raw.slice(1)
        .filter(row => row[0])
        .map(row => {
          const fullJson = row.filter(c => c != null && String(c) !== '').join('');
          try { return JSON.parse(fullJson); }
          catch (e) { console.warn('[DB Dock] Failed to parse DB Copy row:', e); return null; }
        })
        .filter(c => c != null);  // keep all parseable rows (id may be null for new records)

      if (!dbRows.length) {
        setError('No valid candidates found in DB Copy.');
        setUploading(false);
        return;
      }

      // Parse Candidate Data (Sheet 1) → provides editable field values.
      // Row order matches DB Copy — aligned by index.
      const ws1       = wb.Sheets[wb.SheetNames[0]];
      const s1Rows    = XLSX.utils.sheet_to_json(ws1, { defval: '' });

      // Merge: DB Copy supplies metadata; Sheet 1 overrides editable columns.
      const candidates = dbRows.map((dbRow, i) => {
        const s1Row  = s1Rows[i] || {};
        const merged = { ...dbRow };
        for (const [s1Col, dbKey] of Object.entries(S1_TO_DB)) {
          const v = s1Row[s1Col];
          if (v !== undefined && String(v).trim() !== '') merged[dbKey] = v;
        }
        return merged;
      });
      // ─────────────────────────────────────────────────────────────────────────

      fetch('http://localhost:4000/candidates/bulk', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
        body:    JSON.stringify({ candidates }),
        credentials: 'include',
      })
      .then(res => {
        if (!res.ok) throw new Error(`DB Dock & Deploy failed with status ${res.status}`);
        setFile(null); setError(''); onUpload && onUpload();
      })
      .catch(() => setError('Failed to deploy candidates.'))
      .finally(() => setUploading(false));
    }).catch(() => {
      setError('Failed to parse Excel file.');
      setUploading(false);
    });
  };
  return (
    <div className="vskillset-section">
      <div
        className="vskillset-header"
        onClick={() => setExpanded(!expanded)}
        style={{ cursor: 'pointer' }}
      >
        <span className="vskillset-title">DB Dock &amp; Deploy</span>
        <span className="vskillset-arrow">{expanded ? '▼' : '▶'}</span>
      </div>
      {expanded && (
        <div style={{ padding: '8px 0' }}>
          <input type="file" accept=".xlsx,.xls,.xml" onChange={handleFileChange}/>
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
          >{uploading?'Deploying...':'Deploy'}</button>
          {error && <div style={{ color:'var(--danger)', marginTop:8 }}>{error}</div>}
          <div style={{ fontSize:12, marginTop:8, color: 'var(--argent)' }}>
            Accepts DB Port exports only (.xlsx / .xls / .xml). Column schema is sourced from the DB Copy sheet; values are taken from the Candidate Data tab.
          </div>
        </div>
      )}
    </div>
  );
}

/* ========================= MAIN APP ========================= */
/* ========================= NAV SIDEBAR COMPONENT ========================= */
function NavSidebar({ activePage = 'candidate-management' }) {
  const [servicesExpanded, setServicesExpanded] = useState(false);

  return (
    <nav className="nav-sidebar" aria-label="Main navigation">
      <a href="http://localhost:3000/" className="nav-sidebar__brand">FIOE</a>
      <ul className="nav-sidebar__list">

        <li className="nav-sidebar__item">
          <a href="http://localhost:3000/" className="nav-sidebar__link">
            <svg className="nav-sidebar__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>
            </svg>
            <span>Home</span>
          </a>
        </li>

        <li className="nav-sidebar__divider"></li>

        <li
          className="nav-sidebar__item nav-sidebar__item--has-sub"
          onMouseEnter={() => setServicesExpanded(true)}
          onMouseLeave={() => setServicesExpanded(false)}
        >
          <span
            className="nav-sidebar__link"
            role="button"
            tabIndex={0}
            aria-haspopup="true"
            aria-expanded={servicesExpanded ? 'true' : 'false'}
            onKeyDown={e => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                setServicesExpanded(v => !v);
              }
            }}
          >
            <svg className="nav-sidebar__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
              <rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
            </svg>
            <span>Services</span>
            <svg className="nav-sidebar__chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" width="11" height="11">
              <polyline points="9 18 15 12 9 6"/>
            </svg>
          </span>
          <ul className="nav-sidebar__submenu" role="menu" style={{ maxHeight: servicesExpanded ? '300px' : undefined }}>
            <li><a href="http://localhost:8091/AutoSourcing.html" className="nav-sidebar__submenu-link" role="menuitem">Autosourcing</a></li>
            <li><a href="http://localhost:8091/SourcingVerify.html" className="nav-sidebar__submenu-link" role="menuitem">Talent Evaluation</a></li>
            <li><a href="http://localhost:3000/" className={'nav-sidebar__submenu-link' + (activePage === 'candidate-management' ? ' active' : '')} role="menuitem">Candidate Management</a></li>
            <li><a href="http://localhost:5000/LookerDashboard.html" className="nav-sidebar__submenu-link" role="menuitem">Consulting Dashboard</a></li>
          </ul>
        </li>

        <li className="nav-sidebar__divider"></li>

        <li className="nav-sidebar__item">
          <a href="#ai-agent" className="nav-sidebar__link">
            <svg className="nav-sidebar__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
            </svg>
            <span>AI Agent</span>
          </a>
        </li>

        <li className="nav-sidebar__item">
          <a href="http://localhost:8091/api_porting.html" className="nav-sidebar__link">
            <svg className="nav-sidebar__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/>
              <polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/>
            </svg>
            <span>API Port</span>
          </a>
        </li>

        <li className="nav-sidebar__item">
          <a href="#community" className="nav-sidebar__link">
            <svg className="nav-sidebar__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/>
              <circle cx="9" cy="7" r="4"/>
              <path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/>
            </svg>
            <span>Community</span>
          </a>
        </li>

        <li className="nav-sidebar__item">
          <a href="#contact" className="nav-sidebar__link">
            <svg className="nav-sidebar__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
              <polyline points="22,6 12,13 2,6"/>
            </svg>
            <span>Contact Us</span>
          </a>
        </li>

      </ul>
    </nav>
  );
}

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

  // Tabs state
  const [activeTab, setActiveTab] = useState('list'); // 'list' or 'chart' or 'resume'

  // NEW: Resume tab state
  const [resumeCandidate, setResumeCandidate] = useState(null);
  const [resumePicError, setResumePicError] = useState(false);
  
  // State for resume email updating
  const [resumeEmailList, setResumeEmailList] = useState([]);

  // State for email generation/verification in Resume Tab
  const [generatingEmails, setGeneratingEmails] = useState(false);
  const [verifyingEmail, setVerifyingEmail] = useState(false);
  const [verifyModalData, setVerifyModalData] = useState(null);
  const [verifyModalEmail, setVerifyModalEmail] = useState('');
  const [tokenConfirmOpen, setTokenConfirmOpen] = useState(false);
  const [pendingVerifyEmail, setPendingVerifyEmail] = useState(null);
  // State for calculating unmatched skills
  const [calculatingUnmatched, setCalculatingUnmatched] = useState(false);
  const [unmatchedCalculated, setUnmatchedCalculated] = useState({});  // Store by candidate ID

  // State for skillset management
  const [newSkillInput, setNewSkillInput] = useState('');
  const [vskillsetExpanded, setVskillsetExpanded] = useState(false);

  // Auto-expand Verified Skillset panel when resumeCandidate changes and has vskillset entries
  // Only expand on initial load or when candidate changes, don't force re-expansion after manual collapse
  useEffect(() => {
    try {
      if (resumeCandidate?.vskillset?.length > 0) {
        setVskillsetExpanded(true);
      }
    } catch (e) {
      // defensive: don't crash UI if something unexpected is present
      console.warn('[vskillset] auto-expand check failed', e);
    }
  }, [resumeCandidate?.id]); // Only depend on candidate ID to allow manual collapse

  // Category colors for verified skillset
  const VSKILLSET_CATEGORY_COLORS = {
    'High': '#10b981',
    'Medium': '#f59e0b',
    'Low': '#6b7280',
    'Unknown': '#9ca3af'
  };

  // Helper function to render star rating from text or number
  const renderStarRating = (starsValue) => {
    if (!starsValue) return null;
    
    // If it's a string with star characters, count them
    let starCount = 0;
    if (typeof starsValue === 'string') {
      // Count ★ or ⭐ characters
      const fullStars = (starsValue.match(/[★⭐]/g) || []).length;
      // Also try to parse as number if it's like "4.5" or "5"
      const numMatch = starsValue.match(/(\d+\.?\d*)/);
      if (numMatch) {
        starCount = parseFloat(numMatch[1]);
      } else {
        starCount = fullStars;
      }
    } else if (typeof starsValue === 'number') {
      starCount = starsValue;
    }
    
    // Cap at 5 stars
    starCount = Math.min(starCount, 5);
    
    // Star styling constants
    const fullStarStyle = { color: '#fbbf24', fontSize: 20 };
    const emptyStarStyle = { color: '#d1d5db', fontSize: 20 };
    
    // Generate star display
    const stars = [];
    for (let i = 1; i <= 5; i++) {
      if (i <= Math.floor(starCount)) {
        // Full star
        stars.push(<span key={i} style={fullStarStyle}>★</span>);
      } else if (i === Math.ceil(starCount) && starCount % 1 !== 0) {
        // Half star - use hollow star with lighter color for better cross-browser support
        stars.push(<span key={i} style={{ color: '#fbbf24', fontSize: 20, opacity: 0.5 }}>★</span>);
      } else {
        // Empty star
        stars.push(<span key={i} style={emptyStarStyle}>★</span>);
      }
    }
    
    return (
      <div 
        style={{ display: 'flex', gap: 2 }}
        role="img"
        aria-label={`${starCount} out of 5 stars`}
      >
        {stars}
      </div>
    );
  };

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

  const [searchExpanded, setSearchExpanded] = useState(false);
  const [globalSearchInput, setGlobalSearchInput] = useState('');
  const [globalSearch, setGlobalSearch] = useState('');

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
    fetch('http://localhost:4000/logout', { method: 'POST', credentials: 'include', headers: { 'X-Requested-With': 'XMLHttpRequest' } })
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
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
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
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
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
      if (res.status === 401) {
        // Session cookie is missing or expired; force re-login
        setUser(null);
        setLoading(false);
        return;
      }
      if (!res.ok) {
        console.error('[fetchCandidates] server error', res.status);
        setCandidates([]);
        setLoading(false);
        return;
      }
      const raw=await res.json();
      const candidatesList = Array.isArray(raw)?raw:[];
      
      // Log vskillset data for debugging
      const withVskillset = candidatesList.filter(c => c.vskillset);
      if (withVskillset.length > 0) {
        console.log(`[fetchCandidates] Found ${withVskillset.length} candidates with vskillset data`);
        console.log('[fetchCandidates] Sample vskillset:', withVskillset[0].vskillset);
      } else {
        console.log('[fetchCandidates] No candidates with vskillset data found');
      }
      
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
    const q = (globalSearch || '').trim().toLowerCase();
    if (!q) return intersectionFiltered;
    return intersectionFiltered.filter(c => {
      return Object.values(c).some(v => v != null && String(v).toLowerCase().includes(q));
    });
  },[intersectionFiltered, globalSearch]);

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
        headers:{'Content-Type':'application/json','X-Requested-With':'XMLHttpRequest'},
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
          headers:{'Content-Type':'application/json','X-Requested-With':'XMLHttpRequest'},
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
          headers:{'Content-Type':'application/json','X-Requested-With':'XMLHttpRequest'},
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
    console.log('[handleViewProfile] Candidate data:', {
      id: candidate.id,
      name: candidate.name,
      hasVskillset: !!candidate.vskillset,
      vskillsetType: typeof candidate.vskillset,
      vskillsetLength: Array.isArray(candidate.vskillset) ? candidate.vskillset.length : 'N/A',
      vskillsetSample: candidate.vskillset ? (Array.isArray(candidate.vskillset) ? candidate.vskillset[0] : candidate.vskillset) : null
    });
    
    const rawEmails = (candidate.email || '').split(/[;,]+/).map(s => s.trim()).filter(Boolean);
    // Initialize emails list with check state, and default confidence for existing emails (N/A)
    setResumeEmailList(rawEmails.map(e => ({ value: e, checked: false, confidence: 'Stored (N/A)' })));
    setResumePicError(false);
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
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
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
  const handleVerifySelectedEmail = () => {
    const selected = resumeEmailList.filter(item => item.checked);
    if (selected.length === 0) { alert('Please select an email to verify.'); return; }
    if (selected.length > 1) { alert('Please verify one email at a time.'); return; }
    if (tokensLeft < 2) { alert('Insufficient tokens. You need at least 2 tokens to verify an email.'); return; }
    setPendingVerifyEmail(selected[0].value);
    setTokenConfirmOpen(true);
  };

  const handleConfirmVerify = async () => {
    setTokenConfirmOpen(false);
    const emailToVerify = pendingVerifyEmail;
    setPendingVerifyEmail(null);
    setVerifyingEmail(true);
    setVerifyModalEmail(emailToVerify);
    setVerifyModalData(null);
    try {
      const res = await fetch('http://localhost:4000/verify-email-details', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
        body: JSON.stringify({ email: emailToVerify }),
        credentials: 'include'
      });
      if (!res.ok) throw new Error('Verification failed');
      const data = await res.json();
      setVerifyModalData(data);
      // Deduct 2 tokens on successful verification
      fetch('http://localhost:4000/deduct-tokens', { method: 'POST', credentials: 'include', headers: { 'X-Requested-With': 'XMLHttpRequest' } })
        .then(r => r.json())
        .then(t => {
          if (t.tokensLeft !== undefined) setTokensLeft(t.tokensLeft);
          if (t.accountTokens !== undefined) setAccountTokens(t.accountTokens);
        })
        .catch(err => console.error('Token deduction failed:', err));
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
              headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
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

  // Handler to move skill from Unmatched to Skillset (via drag-and-drop)
  const handleMoveToSkillset = (skillToMove) => {
    if (!resumeCandidate) return;
    
    // Remove from unmatched
    const currentUnmatched = resumeCandidate.lskillset ? String(resumeCandidate.lskillset).split(/[;,|]+/).map(s => s.trim()).filter(Boolean) : [];
    const updatedUnmatched = currentUnmatched.filter(s => s !== skillToMove);
    const newLSkillset = updatedUnmatched.join(', ');
    
    // Add to skillset
    const currentSkills = resumeCandidate.skillset ? String(resumeCandidate.skillset).split(/[;,|]+/).map(s => s.trim()).filter(Boolean) : [];
    if (!currentSkills.some(s => s.toLowerCase() === skillToMove.toLowerCase())) {
      currentSkills.push(skillToMove);
    }
    const newSkillset = currentSkills.join(', ');
    
    const id = resumeCandidate.id;
    
    // Update state
    setCandidates(prev => prev.map(c => String(c.id) === String(id) ? { ...c, skillset: newSkillset, lskillset: newLSkillset } : c));
    setResumeCandidate(prev => ({ ...prev, skillset: newSkillset, lskillset: newLSkillset }));
    
    // Save to backend
    saveCandidateDebounced(id, { skillset: newSkillset, lskillset: newLSkillset });
  };

  // Handler to move skill from Skillset to Unmatched (via drag-and-drop)
  const handleMoveToUnmatched = (skillToMove) => {
    if (!resumeCandidate) return;
    
    // Remove from skillset
    const currentSkills = resumeCandidate.skillset ? String(resumeCandidate.skillset).split(/[;,|]+/).map(s => s.trim()).filter(Boolean) : [];
    const updatedSkills = currentSkills.filter(s => s !== skillToMove);
    const newSkillset = updatedSkills.join(', ');
    
    // Add to unmatched
    const currentUnmatched = resumeCandidate.lskillset ? String(resumeCandidate.lskillset).split(/[;,|]+/).map(s => s.trim()).filter(Boolean) : [];
    if (!currentUnmatched.some(s => s.toLowerCase() === skillToMove.toLowerCase())) {
      currentUnmatched.push(skillToMove);
    }
    const newLSkillset = currentUnmatched.join(', ');
    
    const id = resumeCandidate.id;
    
    // Update state
    setCandidates(prev => prev.map(c => String(c.id) === String(id) ? { ...c, skillset: newSkillset, lskillset: newLSkillset } : c));
    setResumeCandidate(prev => ({ ...prev, skillset: newSkillset, lskillset: newLSkillset }));
    
    // Save to backend
    saveCandidateDebounced(id, { skillset: newSkillset, lskillset: newLSkillset });
  };

  // Helper to normalize vskillset array
  const normalizeVskillArray = (raw) => {
    if (!raw) return [];
    if (Array.isArray(raw)) return raw.filter(Boolean);
    if (typeof raw === 'object') {
      // If it's an object with numeric keys, convert to array
      const keys = Object.keys(raw).filter(k => !isNaN(k)).sort((a, b) => Number(a) - Number(b));
      return keys.map(k => raw[k]).filter(Boolean);
    }
    return [];
  };

  // Helper to parse skillset string into array
  const parseSkillsetString = (skillsetStr) => {
    return skillsetStr ? String(skillsetStr).split(/[;,|]+/).map(s => s.trim()).filter(Boolean) : [];
  };

  // Helper to update candidate in both state locations
  const updateCandidateState = (id, updates) => {
    setCandidates(prev => prev.map(c => String(c.id) === String(id) ? { ...c, ...updates } : c));
    setResumeCandidate(prev => ({ ...prev, ...updates }));
  };

  // Handler to accept a verified skill (move to main skillset)
  const handleAcceptVskill = (vskillItem) => {
    if (!resumeCandidate) return;
    
    const skillName = vskillItem.skill || (typeof vskillItem === 'string' ? vskillItem : '');
    if (!skillName) return;
    
    // Add to main skillset
    const currentSkills = parseSkillsetString(resumeCandidate.skillset);
    if (!currentSkills.some(s => s.toLowerCase() === skillName.toLowerCase())) {
      currentSkills.push(skillName);
    }
    const newSkillset = currentSkills.join(', ');
    
    // Remove from vskillset
    const currentVskills = normalizeVskillArray(resumeCandidate.vskillset);
    const updatedVskills = currentVskills.filter(v => {
      const vName = v.skill || (typeof v === 'string' ? v : '');
      return vName.toLowerCase() !== skillName.toLowerCase();
    });
    
    const id = resumeCandidate.id;
    const updates = { skillset: newSkillset, vskillset: updatedVskills };
    
    // Update state
    updateCandidateState(id, updates);
    
    // Save to backend
    saveCandidateDebounced(id, { skillset: newSkillset, vskillset: JSON.stringify(updatedVskills) });
  };

  // Handler to dismiss a verified skill (remove from vskillset)
  const handleDismissVskill = (vskillItem) => {
    if (!resumeCandidate) return;
    
    const skillName = vskillItem.skill || (typeof vskillItem === 'string' ? vskillItem : '');
    if (!skillName) return;
    
    // Remove from vskillset
    const currentVskills = normalizeVskillArray(resumeCandidate.vskillset);
    const updatedVskills = currentVskills.filter(v => {
      const vName = v.skill || (typeof v === 'string' ? v : '');
      return vName.toLowerCase() !== skillName.toLowerCase();
    });
    
    const id = resumeCandidate.id;
    const updates = { vskillset: updatedVskills };
    
    // Update state
    updateCandidateState(id, updates);
    
    // Save to backend
    saveCandidateDebounced(id, { vskillset: JSON.stringify(updatedVskills) });
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
    <div className="page-shell">
      <NavSidebar activePage="candidate-management" />
      <div className="page-main" style={{
        padding: 24,
        boxSizing: 'border-box',
        background:'var(--bg)',
        color: 'var(--muted)',
        overflowX: 'hidden'
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
          onClick={() => { setActiveTab('chart'); }}
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
        <div style={{ marginTop:32 }}>
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
                searchExpanded={searchExpanded}
                onToggleSearch={() => setSearchExpanded(prev => !prev)}
                globalSearchInput={globalSearchInput}
                onGlobalSearchChange={v => { setGlobalSearchInput(v); }}
                onGlobalSearchSubmit={() => { setGlobalSearch(globalSearchInput); setPage(1); }}
                onClearSearch={() => { setGlobalSearchInput(''); setGlobalSearch(''); setPage(1); }}
                onViewProfile={handleViewProfile}
                statusOptions={statusOptions}
                onOpenStatusModal={() => setStatusModalOpen(true)}
                allCandidates={candidates}
                user={user}
                onDockIn={fetchCandidates}
                tokensLeft={tokensLeft}
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
                            {resumeCandidate.pic && typeof resumeCandidate.pic === 'string' && !resumePicError ? (
                                <img 
                                    src={(() => {
                                        const p = resumeCandidate.pic.trim();
                                        if (p.startsWith('http://') || p.startsWith('https://') || p.startsWith('data:')) return p;
                                        // Strip any embedded whitespace (e.g., line-breaks in base64)
                                        const b64 = p.replace(/\s/g, '');
                                        return !b64.startsWith('data:image/') ? `data:image/jpeg;base64,${b64}` : b64;
                                    })()}
                                    alt={resumeCandidate.name || 'Candidate'}
                                    style={{
                                        width: 60,
                                        height: 60,
                                        borderRadius: '50%',
                                        objectFit: 'cover',
                                        border: '2px solid #e5e7eb',
                                        display: 'block'
                                    }}
                                    onError={() => setResumePicError(true)}
                                />
                            ) : (
                                // Placeholder for missing or failed image
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
                                    <label style={{ display: 'block', fontSize: 15, fontWeight: 700, marginBottom: 4 }}>Seniority</label>
                                    <div style={{ fontSize: 16 }}>{resumeCandidate.seniority || '—'}</div>
                                </div>
                                <div>
                                    <label style={{ display: 'block', fontSize: 15, fontWeight: 700, marginBottom: 4 }}>Total Experience</label>
                                    {/* Switched to use .exp */}
                                    <div style={{ fontSize: 16 }}>{resumeCandidate.exp ? `${resumeCandidate.exp} Years` : '—'}</div>
                                </div>
                                <div>
                                    <label style={{ display: 'block', fontSize: 15, fontWeight: 700, marginBottom: 4 }}>Job Family</label>
                                    <div style={{ fontSize: 16 }}>{resumeCandidate.job_family || '—'}</div>
                                </div>
                                <div>
                                    <label style={{ display: 'block', fontSize: 15, fontWeight: 700, marginBottom: 4 }}>Sector</label>
                                    <div style={{ fontSize: 16 }}>{resumeCandidate.sector || '—'}</div>
                                </div>
                                
                                {/* AVG Tenure field added above Location */}
                                <div>
                                    <label style={{ display: 'block', fontSize: 15, fontWeight: 700, marginBottom: 4 }}>AVG Tenure</label>
                                    <div style={{ fontSize: 16, color: 'var(--muted)' }}>
                                        {resumeCandidate.tenure ? `${resumeCandidate.tenure} Years` : '—'}
                                    </div>
                                </div>
                                
                                {/* Location field */}
                                <div>
                                    <label style={{ display: 'block', fontSize: 15, fontWeight: 700, marginBottom: 4 }}>Location</label>
                                    <div style={{ fontSize: 16, color: 'var(--muted)' }}>
                                        {[resumeCandidate.city, resumeCandidate.country].filter(Boolean).join(', ') || '—'}
                                    </div>
                                </div>
                                
                                {/* Mobile field moved below Location */}
                                <div>
                                    <label style={{ display: 'block', fontSize: 15, fontWeight: 700, marginBottom: 4 }}>Mobile</label>
                                    <div style={{ fontSize: 16, color: 'var(--muted)' }}>{resumeCandidate.mobile || '—'}</div>
                                </div>
                                
                                {/* Office field added below Mobile */}
                                <div>
                                    <label style={{ display: 'block', fontSize: 15, fontWeight: 700, marginBottom: 4 }}>Office</label>
                                    <div style={{ fontSize: 16, color: 'var(--muted)' }}>{resumeCandidate.office || '—'}</div>
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
                        <h3 className="skillset-header">Skillset (Drag skills here or from here)</h3>
                        <div 
                            className="skillset-container"
                            title="Drag Skills here or from here"
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={(e) => {
                                e.preventDefault();
                                const skill = e.dataTransfer.getData('skill');
                                const source = e.dataTransfer.getData('source');
                                if (skill && source === 'unmatched') {
                                    handleMoveToSkillset(skill);
                                }
                            }}
                        >
                            {resumeCandidate.skillset ? (
                                String(resumeCandidate.skillset).split(/[;,|]+/).map((skill, i) => {
                                    const s = skill.trim();
                                    if(!s) return null;
                                    return (
                                        <span 
                                            key={i} 
                                            className="skill-bubble"
                                            draggable="true"
                                            onDragStart={(e) => {
                                                e.dataTransfer.setData('skill', s);
                                                e.dataTransfer.setData('source', 'skillset');
                                            }}
                                        >
                                            {s}
                                            <button
                                                onClick={() => handleRemoveSkill(s)}
                                                className="remove-btn"
                                                title="Remove skill"
                                            >
                                                ×
                                            </button>
                                        </span>
                                    );
                                })
                            ) : <span style={{ color: '#9ca3af', fontSize: 13 }}>No skills listed.</span>}
                        </div>
                        
                        {/* Textbox to manually add new skills */}
                        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                            <input
                                type="text"
                                value={newSkillInput}
                                onChange={(e) => setNewSkillInput(e.target.value)}
                                onKeyPress={(e) => {
                                    if (e.key === 'Enter' && newSkillInput.trim()) {
                                        handleAddSkill(newSkillInput);
                                        setNewSkillInput('');
                                    }
                                }}
                                placeholder="Enter a new skill and press Enter"
                                style={{
                                    flex: 1,
                                    padding: '8px 12px',
                                    fontSize: 14,
                                    border: '1px solid var(--neutral-border)',
                                    borderRadius: 6,
                                    outline: 'none'
                                }}
                            />
                            <button
                                onClick={() => {
                                    if (newSkillInput.trim()) {
                                        handleAddSkill(newSkillInput);
                                        setNewSkillInput('');
                                    }
                                }}
                                disabled={!newSkillInput.trim()}
                                style={{
                                    padding: '8px 16px',
                                    fontSize: 14,
                                    fontWeight: 600,
                                    background: newSkillInput.trim() ? '#2563eb' : '#e5e7eb',
                                    color: newSkillInput.trim() ? 'white' : '#9ca3af',
                                    border: 'none',
                                    borderRadius: 6,
                                    cursor: newSkillInput.trim() ? 'pointer' : 'not-allowed',
                                    transition: 'background 0.2s'
                                }}
                            >
                                Add Skill
                            </button>
                        </div>
                    </div>

                    <div style={{ marginBottom: 24 }}>
                        <h3 className="skillset-header">Unmatched Skillset (Drag skills here or from here)</h3>
                        <div 
                            className="skillset-container"
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={(e) => {
                                e.preventDefault();
                                const skill = e.dataTransfer.getData('skill');
                                const source = e.dataTransfer.getData('source');
                                if (skill && source === 'skillset') {
                                    handleMoveToUnmatched(skill);
                                }
                            }}
                        >
                             {unmatchedCalculated[resumeCandidate?.id] && !resumeCandidate.lskillset ? (
                                // Show "All skillsets are matched" message when calculated and no unmatched skills
                                <span style={{ color: '#15803d', fontSize: 13, fontWeight: 600 }}>All skillsets are matched.</span>
                             ) : resumeCandidate.lskillset ? (
                                // Show unmatched skills with drag-and-drop
                                String(resumeCandidate.lskillset)
                                    .replace(/Here are the skills present in the JD Skillset but missing or unmatched in the Candidate Skillset[:\s]*/i, '')
                                    .replace(/[\[\]"']/g, '') // Strips brackets and quotes
                                    .split(/[;,|]+/)
                                    .map((skill, i) => {
                                        const s = skill.trim();
                                        if(!s) return null;
                                        return (
                                            <span 
                                                key={i} 
                                                className="skill-bubble unmatched"
                                                draggable="true"
                                                onDragStart={(e) => {
                                                    e.dataTransfer.setData('skill', s);
                                                    e.dataTransfer.setData('source', 'unmatched');
                                                }}
                                            >
                                                {s}
                                            </span>
                                        );
                                })
                            ) : (
                                <div style={{ width: '100%' }}>
                                    <span style={{ color: '#b91c1c', fontSize: 13 }}>Click calculate to compare against job requirements.</span>
                                    <button 
                                        onClick={handleCalculateUnmatched}
                                        disabled={calculatingUnmatched}
                                        className="btn-primary btn-calculate"
                                    >
                                        {calculatingUnmatched ? 'Calculating...' : 'Calculate Unmatched'}
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Verified Skillset Details Section - Always visible, purely visualization */}
                    <div className="vskillset-section">
                        <div 
                            className="vskillset-header"
                            onClick={() => setVskillsetExpanded(!vskillsetExpanded)}
                            style={{ cursor: 'pointer' }}
                        >
                            <span className="vskillset-title">Verified Skillset Details</span>
                            <span className="vskillset-arrow">{vskillsetExpanded ? '▼' : '▶'}</span>
                        </div>
                        {vskillsetExpanded && (
                            <div style={{ overflowX: 'auto' }}>
                                {resumeCandidate.vskillset && Array.isArray(resumeCandidate.vskillset) && resumeCandidate.vskillset.length > 0 ? (
                                    <table className="vskillset-table">
                                        <thead>
                                            <tr>
                                                <th>Skill</th>
                                                <th>Probability</th>
                                                <th>Category</th>
                                                <th>Reason</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {resumeCandidate.vskillset.map((item, idx) => {
                                                const category = item.category || 'Unknown';
                                                // Handle probability: if value is 0-1 (decimal), convert to percentage
                                                let probabilityValue = typeof item.probability !== 'undefined' ? item.probability : null;
                                                if (probabilityValue !== null) {
                                                    if (probabilityValue >= 0 && probabilityValue <= 1) {
                                                        probabilityValue = probabilityValue * 100;
                                                    }
                                                    probabilityValue = `${Math.round(probabilityValue)}%`;
                                                } else {
                                                    probabilityValue = 'N/A';
                                                }
                                                const categoryColor = VSKILLSET_CATEGORY_COLORS[category] || VSKILLSET_CATEGORY_COLORS['Unknown'];
                                                
                                                return (
                                                    <tr key={idx}>
                                                        <td style={{ color: '#1f2937' }}>{item.skill || ''}</td>
                                                        <td style={{ color: '#1f2937' }}>{probabilityValue}</td>
                                                        <td>
                                                            <span style={{ 
                                                                color: categoryColor, 
                                                                fontWeight: 600,
                                                                fontSize: 11
                                                            }}>
                                                                {category}
                                                            </span>
                                                        </td>
                                                        <td style={{ color: '#6b7280', fontSize: 11 }}>{item.reason || ''}</td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                ) : (
                                    <div style={{ 
                                        padding: '20px', 
                                        textAlign: 'center', 
                                        color: '#6b7280',
                                        fontSize: 14
                                    }}>
                                        No verified skills available for this candidate.
                                    </div>
                                )}
                            </div>
                        )}
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

                    {/* Professional Assessment Table Display - Robust version with JSON parsing */}
                    {(() => {
                        if (!resumeCandidate || !resumeCandidate.rating) return null;

                        // Normalize rating to an object if possible
                        let ratingRaw = resumeCandidate.rating;
                        let ratingObj = null;
                        if (typeof ratingRaw === 'string') {
                            // attempt to parse JSON safely
                            try {
                                ratingObj = JSON.parse(ratingRaw);
                            } catch (e) {
                                // not JSON — leave ratingObj as null, will use ratingRaw as string
                                ratingObj = null;
                            }
                        } else if (typeof ratingRaw === 'object') {
                            ratingObj = ratingRaw;
                        }

                        // If we have a structured rating object with assessment_level, render the professional table
                        if (ratingObj && ratingObj.assessment_level) {
                            const r = ratingObj;
                            return (
                                <div style={{ marginBottom: 24 }}>
                                    <h3 className="skillset-header">Candidate Assessment</h3>
                                    <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', borderRadius: 8, overflow: 'hidden' }}>
                                        <table className="assessment-table">
                                            <thead>
                                                <tr>
                                                    <th>CATEGORY</th>
                                                    <th>DETAILS</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr>
                                                    <td style={{ fontWeight: 600, color: '#374151', width: '25%' }}>Assessment Level</td>
                                                    <td style={{ fontWeight: 600 }}>
                                                        <span className="assessment-badge">
                                                            {r.assessment_level}
                                                        </span>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td style={{ fontWeight: 600, color: '#374151' }}>Overall Score</td>
                                                    <td style={{ color: '#4c82b8', fontWeight: 700, fontSize: 24 }}>
                                                        {r.total_score || 'N/A'}
                                                    </td>
                                                </tr>
                                                {r.stars && (
                                                    <tr>
                                                        <td style={{ fontWeight: 600, color: '#374151' }}>Rating</td>
                                                        <td>
                                                            {renderStarRating(r.stars)}
                                                        </td>
                                                    </tr>
                                                )}
                                                {r.overall_comment && (
                                                    <tr>
                                                        <td style={{ fontWeight: 600, color: '#374151', verticalAlign: 'top' }}>Executive Summary</td>
                                                        <td>
                                                            <div style={{ 
                                                                padding: 16, 
                                                                background: '#dbeafe', 
                                                                borderLeft: '4px solid #4c82b8', 
                                                                borderRadius: 4,
                                                                fontSize: 14,
                                                                color: '#1e40af',
                                                                lineHeight: 1.6
                                                            }}>
                                                                {r.overall_comment}
                                                            </div>
                                                        </td>
                                                    </tr>
                                                )}
                                                {r.comments && (
                                                    <tr>
                                                        <td style={{ fontWeight: 600, color: '#374151', verticalAlign: 'top' }}>AI Assessment</td>
                                                        <td>
                                                            <div style={{ 
                                                                padding: 16, 
                                                                background: '#f9fafb', 
                                                                borderRadius: 4,
                                                                border: '1px solid #e5e7eb',
                                                                whiteSpace: 'pre-wrap',
                                                                fontSize: 14,
                                                                color: '#374151',
                                                                lineHeight: 1.6
                                                            }}>
                                                                {r.comments}
                                                            </div>
                                                        </td>
                                                    </tr>
                                                )}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            );
                        }

                        // If we have an object but no assessment_level, pretty-print the object for readability
                        if (ratingObj && typeof ratingObj === 'object') {
                            return (
                                <div style={{ marginBottom: 24 }}>
                                    <h3 className="skillset-header">Candidate Assessment</h3>
                                    <div style={{ 
                                        padding: 12, 
                                        background: '#fff', 
                                        border: '1px solid var(--neutral-border)', 
                                        borderRadius: 8, 
                                        fontFamily: 'monospace', 
                                        whiteSpace: 'pre-wrap', 
                                        fontSize: 12, 
                                        color: '#334155'
                                    }}>
                                        {JSON.stringify(ratingObj, null, 2)}
                                    </div>
                                </div>
                            );
                        }

                        // Fallback: rating is a string (non-JSON) — preserve previous behavior but render with paragraph styling
                        return (
                            <div style={{ marginBottom: 24 }}>
                                <h3 className="skillset-header">Candidate Assessment</h3>
                                <div style={{ padding: 12, background: '#fff', border: '1px solid var(--neutral-border)', borderRadius: 8 }}>
                                    <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8 }}>
                                        Assessment Notes
                                    </div>
                                    <div style={{ fontSize: 14, lineHeight: 1.8, color: '#374151' }}>
                                        {String(ratingRaw).split('\n').map((para, idx) => (
                                            <p key={`para-${idx}-${para.substring(0, 20)}`} style={{ marginBottom: 12, marginTop: 0, paddingLeft: 12, borderLeft: '3px solid #e5e7eb' }}>
                                                {para}
                                            </p>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        );
                    })()}

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

      {tokenConfirmOpen && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(34,37,41,0.65)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 10002 }}
             onClick={() => setTokenConfirmOpen(false)}>
          <div className="app-card" style={{ width: 420, padding: 28 }} onClick={e => e.stopPropagation()}>
            <h3 style={{ marginTop: 0, marginBottom: 12, color: 'var(--azure-dragon)', fontSize: 16 }}>Confirm Verified Selection</h3>
            <p style={{ fontSize: 14, color: 'var(--muted)', marginBottom: 24, lineHeight: 1.6 }}>
              Are you sure you want to proceed?&nbsp;
              <strong>2 tokens will be deducted</strong> from your account for this verified selection.
            </p>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button onClick={() => setTokenConfirmOpen(false)} className="btn-secondary" style={{ padding: '8px 20px', fontSize: 13 }}>Cancel</button>
              <button onClick={handleConfirmVerify} className="btn-primary" style={{ padding: '8px 20px', fontSize: 13 }}>Continue</button>
            </div>
          </div>
        </div>
      )}

      <StatusManagerModal
        isOpen={statusModalOpen}
        onClose={() => setStatusModalOpen(false)}
        statuses={statusOptions}
        onAddStatus={handleAddStatus}
        onRemoveStatus={handleRemoveStatus}
      />
      </div>
    </div>
  );
}