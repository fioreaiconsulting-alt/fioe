const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');
const http = require('http'); // added for socket server
const crypto = require('crypto'); // Built-in node crypto for password hashing
const dns = require('dns').promises; // Built-in DNS for MX checks
const net = require('net'); // Built-in Net for SMTP handshake
const nodemailer = require('nodemailer'); // Added for sending emails

// Lazy-load Gemini SDK so the server still boots if it isn't installed
let GoogleGenerativeAIClass = null;
try {
  ({ GoogleGenerativeAI: GoogleGenerativeAIClass } = require('@google/generative-ai'));
} catch (e) {
  console.warn("[WARN] '@google/generative-ai' not installed. /verify-data will return an informative error until it's installed.");
}

// Lazy-load Google APIs for Looker/Sheets integration
let google = null;
try {
  ({ google } = require('googleapis'));
} catch (e) {
  console.warn("[WARN] 'googleapis' not installed. Port to Looker Studio features will fail.");
}

const app = express();
const port = 4000;

// Enable parsing cookies
const cookieParser = require('cookie-parser');
app.use(cookieParser());

app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NEW: Serve images from 'image' directory
app.use('/image', express.static(path.join(__dirname, 'image')));

// Update CORS to allow credentials (cookies)
const allowedOrigins = ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:8000', 'http://127.0.0.1:8000'];
app.use(cors({
  origin: allowedOrigins,
  credentials: true
}));

const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'candidate_db',
  password: 'orlha',
  port: 5432,
});

const mappingPath = path.resolve(__dirname, 'skillset-mapping.json');


// ========================= HELPERS: COMPANY & JOB TITLE NORMALIZATION =========================

// Small alias map for common company variants (extend as needed)
const COMPANY_ALIAS_MAP = [
  { re: /\bnexon(?:\s+games)?\b/i, canonical: 'Nexon' },
  { re: /\bmihoyo\b|\bmiho?yo\b/i, canonical: 'Mihoyo' },
  { re: /\btencent(?:\s+(?:gaming|games|cloud|music|video|pictures|entertainment))?\b/i, canonical: 'Tencent' },
  { re: /\bgarena\b/i, canonical: 'Garena' },
  { re: /\boppo\b/i, canonical: 'Oppo' },
  { re: /\blilith\b/i, canonical: 'Lilith Games' },
  { re: /\bla?rian\b/i, canonical: 'Larian Studios' },
  // add more known brand normalizations here
];

// Remove common legal suffixes and noise, then apply alias map and Title Case result
function normalizeCompanyName(raw) {
  if (raw == null) return null;
  const s = String(raw).trim();
  if (!s) return null;
  // If already matches an alias exactly, return that canonical
  for (const a of COMPANY_ALIAS_MAP) {
    if (a.re.test(s)) return a.canonical;
  }
  // Remove punctuation and known suffixes/words that are noise
  let cleaned = s
    .replace(/\b(Co|Co\.|Company|LLC|Inc|Inc\.|Ltd|Ltd\.|GmbH|AG|S\.A\.|Pty Ltd|Sdn Bhd|SAS|S\.A\.S\.|KK|BV)\b/gi, '')
    .replace(/\b(Group|Studios|Studio|Games|Entertainment|Interactive)\b/gi, '')
    .replace(/[,()"]/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim();

  // Remove special characters (non-alphanumeric except spaces)
  cleaned = cleaned.replace(/[^a-zA-Z0-9\s]/g, '').replace(/\s{2,}/g, ' ').trim();

  // map again after cleaning
  for (const a of COMPANY_ALIAS_MAP) {
    if (a.re.test(cleaned)) return a.canonical;
  }

  // Title case the cleaned name
  cleaned = cleaned.split(' ').map(w => {
    if (!w) return '';
    return w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
  }).join(' ').trim();

  return cleaned || null;
}

// Canonicalize a job title into a concise, common form.
// Preserves seniority/lead tokens when detected.
function canonicalJobTitle(rawTitle) {
  if (rawTitle == null) return null;
  const t = String(rawTitle).trim();
  if (!t) return null;
  const lower = t.toLowerCase();

  // detect seniority prefix/suffix
  const seniorityMatch = lower.match(/\b(senior|sr|lead|principal|manager|director|jr|junior|mid|expert)\b/);
  let seniorityPrefix = '';
  if (seniorityMatch) {
    const v = seniorityMatch[0];
    if (/\b(senior|sr)\b/.test(v)) seniorityPrefix = 'Senior ';
    else if (/\b(lead)\b/.test(v)) seniorityPrefix = 'Lead ';
    else if (/\b(principal|expert)\b/.test(v)) seniorityPrefix = 'Expert ';
    else if (/\b(jr|junior)\b/.test(v)) seniorityPrefix = 'Junior ';
    else if (/\b(mid)\b/.test(v)) seniorityPrefix = 'Mid ';
    else if (/\b(manager)\b/.test(v)) seniorityPrefix = 'Manager ';
    else if (/\b(director)\b/.test(v)) seniorityPrefix = 'Director ';
  }

  // graphics-related normalization
  if (/\b(graphic|graphics|gfx)\b/.test(lower)) {
    if (/\b(programm(er|ing)|engine)\b/.test(lower)) {
      // prefer "Graphics Programmer" for programmer-like titles
      return (seniorityPrefix + 'Graphics Programmer').trim();
    }
    if (/\b(engineer|engineering)\b/.test(lower) && !/\b(programm(er|ing))\b/.test(lower)) {
      return (seniorityPrefix + 'Graphics Engineer').trim();
    }
    // fallback
    return (seniorityPrefix + 'Graphics Engineer').trim();
  }

  // cloud-related normalization (Cloud Specialist, Cloud Developer → Cloud Engineer)
  // Exception: Cloud Architect remains separate due to distinct expertise level
  if (/\b(cloud)\b/.test(lower)) {
    if (/\b(architect)\b/.test(lower)) {
      return (seniorityPrefix + 'Cloud Architect').trim();
    }
    if (/\b(specialist|developer|engineer|consultant|analyst)\b/.test(lower)) {
      return (seniorityPrefix + 'Cloud Engineer').trim();
    }
  }

  // engine programmer / game engine
  if (/\b(engine programmer|engineer programmer|engineer|engine programmer|game engine)\b/.test(lower) || /\b(engine)\b/.test(lower) && /\b(programmer|program)\b/.test(lower)) {
    return (seniorityPrefix + 'Engine Programmer').trim();
  }

  // general programmer vs engineer detection
  if (/\b(programm(er|ing))\b/.test(lower)) {
    return (seniorityPrefix + 'Programmer').trim();
  }
  if (/\b(engineer|software eng|swe|eng)\b/.test(lower)) {
    return (seniorityPrefix + 'Engineer').trim();
  }

  if (/\b(technical artist|tech artist)\b/.test(lower)) {
    return (seniorityPrefix + 'Technical Artist').trim();
  }

  // manager/director
  if (/\b(manager|mgr)\b/.test(lower)) return (seniorityPrefix + 'Manager').trim();
  if (/\b(director|dir)\b/.test(lower)) return (seniorityPrefix + 'Director').trim();

  // default: compact and title-case the original, but prefer some token normalization
  const cleaned = t.replace(/\s{2,}/g, ' ').split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
  return (seniorityPrefix + cleaned).trim();
}

// 'Junior', 'Mid', 'Senior', 'Lead', 'Manager', 'Director', 'Expert', 'Executive' or null
function standardizeSeniority(raw) {
  if (!raw) return null;
  // Normalize: lowercase, remove punctuation that separates tokens, convert hyphens/underscores to spaces
  let s = String(raw).trim().toLowerCase();
  s = s.replace(/[.,]/g, '');            // remove commas/dots
  s = s.replace(/[_\-\/]+/g, ' ');       // convert hyphen/underscore/slash to space
  s = s.replace(/\s{2,}/g, ' ').trim();  // collapse multiple spaces

  // Exact/Strong matches (tokenized)
  if (/^(junior|jr)$/.test(s)) return 'Junior';
  if (/^(mid|middle|mid level|mid-level|midlevel|intermediate)$/.test(s)) return 'Mid';
  if (/^(senior|sr)$/.test(s)) return 'Senior';
  if (/^(lead)$/.test(s)) return 'Lead';
  if (/^(manager|mgr)$/.test(s)) return 'Manager';
  if (/^(director|dir)$/.test(s)) return 'Director';
  if (/^(expert|principal|staff)$/.test(s)) return 'Expert';
  if (/^(executive|exec|vp|cxo|chief|head|svp)$/.test(s)) return 'Executive';

  // Fuzzy / contains checks for multi-word or noisy strings
  if (/\b(junior|jr)\b/.test(s)) return 'Junior';
  if (/\b(mid|middle|intermediate|mid level|mid-level|midlevel)\b/.test(s)) return 'Mid';
  if (/\b(senior|sr)\b/.test(s)) return 'Senior';
  if (/\blead\b/.test(s)) return 'Lead';
  if (/\b(manager|mgr)\b/.test(s)) return 'Manager';
  if (/\bdirector\b/.test(s)) return 'Director';
  if (/\b(expert|principal|staff)\b/.test(s)) return 'Expert';
  if (/\b(executive|exec|vp|cxo|chief|head|svp)\b/.test(s)) return 'Executive';

  return null;
}

// Remove special characters (non-alphanumeric) from a string, keeping only letters, numbers, and spaces
function removeSpecialCharacters(text) {
  if (text == null) return null;
  const s = String(text).trim();
  if (!s) return null;
  // Keep only alphanumeric characters and spaces
  return s.replace(/[^a-zA-Z0-9\s]/g, '').replace(/\s{2,}/g, ' ').trim();
}

// Load and cache country code mapping
let countryCodeMap = null;
function loadCountryCodeMap() {
  if (countryCodeMap) return countryCodeMap;
  try {
    const fs = require('fs');
    const countryCodePath = path.resolve(__dirname, 'countrycode.JSON');
    const data = fs.readFileSync(countryCodePath, 'utf8');
    countryCodeMap = JSON.parse(data);
    return countryCodeMap;
  } catch (err) {
    console.warn('[COUNTRY] Failed to load countrycode.JSON:', err.message);
    return {};
  }
}

// Normalize country name using countrycode.JSON mapping
function normalizeCountry(raw) {
  if (raw == null) return null;
  const s = String(raw).trim();
  if (!s) return null;
  
  const countryMap = loadCountryCodeMap();
  const lower = s.toLowerCase();
  
  // Check for exact match in values (case-insensitive)
  for (const [code, name] of Object.entries(countryMap)) {
    if (name.toLowerCase() === lower) {
      return name;
    }
  }
  
  // Check for common aliases
  const aliases = {
    'south korea': 'Korea',
    'republic of korea': 'Korea',
    'rok': 'Korea',
    'united states of america': 'United States',
    'usa': 'United States',
    'us': 'United States',
    'uk': 'United Kingdom',
    'great britain': 'United Kingdom',
    'uae': 'United Arab Emirates',
    'emirates': 'United Arab Emirates'
  };
  
  if (aliases[lower]) {
    return aliases[lower];
  }
  
  // Check for partial matches (e.g., "South Korea" contains "Korea")
  for (const [code, name] of Object.entries(countryMap)) {
    if (lower.includes(name.toLowerCase()) || name.toLowerCase().includes(lower)) {
      return name;
    }
  }
  
  // Return original if no match found, but title-cased
  return s.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
}

// Utility: update process row's company/personal fields if canonicalization suggests change.
// Returns an object { company, personal } with canonical values (may be null or same as inputs).
async function ensureCanonicalFieldsForId(id, currentCompany, currentJobTitle, currentPersonal) {
  const canonicalCompany = normalizeCompanyName(currentCompany || '');
  const canonicalPersonal = (currentPersonal && String(currentPersonal).trim()) ? String(currentPersonal).trim() : (currentJobTitle ? canonicalJobTitle(currentJobTitle) : null);

  // Build SET clauses only if a meaningful change is required.
  const sets = [];
  const values = [];
  let idx = 1;
  if (canonicalCompany != null && String(canonicalCompany).trim() !== String(currentCompany || '').trim()) {
    sets.push(`company = $${idx}`); values.push(canonicalCompany); idx++;
  }
  // If canonicalPersonal is null explicitly and currentPersonal is non-null, we avoid clearing unless explicitly asked.
  if (canonicalPersonal != null && String(canonicalPersonal).trim() !== String(currentPersonal || '').trim()) {
    sets.push(`personal = $${idx}`); values.push(canonicalPersonal); idx++;
  }

  if (sets.length) {
    values.push(id);
    const sql = `UPDATE "process" SET ${sets.join(', ')} WHERE id = $${idx}`;
    try {
      await pool.query(sql, values);
    } catch (err) {
      console.warn('[CANON] failed to persist canonical fields for id', id, err && err.message);
    }
  }

  return { company: canonicalCompany, personal: canonicalPersonal };
}

// Helper to determine region from country name for validation
function getRegionFromCountry(country) {
  if (!country) return null;
  const c = String(country).trim().toLowerCase();
  // common mappings (extend as needed)
  const asia = ['singapore','china','japan','india','south korea','korea','hong kong','taiwan','thailand','philippines','vietnam','malaysia','indonesia'];
  const northAmerica = ['united states','usa','us','canada','mexico'];
  const westernEurope = ['united kingdom','uk','england','france','germany','spain','italy','netherlands','belgium','sweden','norway','finland','denmark','switzerland','austria','ireland','portugal'];
  const easternEurope = ['russia','poland','ukraine','czech','hungary','slovakia','romania','bulgaria','serbia','croatia','latvia','lithuania','estonia'];
  const middleEast = ['saudi arabia','uae','qatar','israel','iran','iraq','oman','kuwait','jordan','lebanon','bahrain','syria','yemen'];
  const southAmerica = ['brazil','argentina','colombia','chile','peru','venezuela','uruguay','paraguay','bolivia','ecuador'];
  const africa = ['south africa','nigeria','egypt','kenya','ghana','morocco','algeria','tunisia'];
  const oceania = ['australia','new zealand'];

  const groups = [
    { region: 'Asia', list: asia },
    { region: 'North America', list: northAmerica },
    { region: 'Western Europe', list: westernEurope },
    { region: 'Eastern Europe', list: easternEurope },
    { region: 'Middle East', list: middleEast },
    { region: 'South America', list: southAmerica },
    { region: 'Africa', list: africa },
    { region: 'Australia/Oceania', list: oceania }
  ];

  for (const g of groups) {
    for (const name of g.list) {
      if (c.includes(name)) return g.region;
    }
  }
  return null;
}

// Helper: ensure the current req.user owns the given process row id
async function ensureOwnershipOrFail(res, id, userId) {
  try {
    const q = await pool.query('SELECT userid FROM "process" WHERE id = $1', [id]);
    if (q.rows.length === 0) {
      res.status(404).json({ error: 'Not found' });
      return false;
    }
    const owner = q.rows[0].userid;
    if (String(owner) !== String(userId)) {
      res.status(403).json({ error: 'Forbidden: not owner' });
      return false;
    }
    return true;
  } catch (err) {
    console.error('[AUTHZ] ownership check failed', err);
    res.status(500).json({ error: 'Ownership check failed' });
    return false;
  }
}

// ========================= END HELPERS =========================


// ========== NEW: Ensure process table has necessary columns (idempotent) ==========
async function ensureProcessTable() {
  try {
    // Create if missing with a superset of columns we expect.
    // Note: column names are chosen to match the mapping you provided.
    // ADDED linkedinurl to creation script for completeness, though ADD COLUMN below handles existing
    await pool.query(`
      CREATE TABLE IF NOT EXISTS "process" (
        id SERIAL PRIMARY KEY,
        name TEXT,
        jobtitle TEXT,
        company TEXT,
        sector TEXT,
        jobfamily TEXT,
        role_tag TEXT,
        skillset TEXT,
        geographic TEXT,
        country TEXT,
        email TEXT,
        mobile TEXT,
        office TEXT,
        personal TEXT,
        seniority TEXT,
        sourcingstatus TEXT,
        product TEXT,
        userid TEXT,
        username TEXT,
        cv BYTEA,
        lskillset TEXT,
        linkedinurl TEXT,
        jskillset TEXT,
	rating INTEGER,
        pic BYTEA,
        education TEXT,
        comment TEXT
      )
    `);

    // Add columns if missing (idempotent)
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS name TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS jobtitle TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS company TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS sector TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS jobfamily TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS role_tag TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS skillset TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS geographic TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS country TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS email TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS mobile TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS office TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS personal TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS seniority TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS sourcingstatus TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS product TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS userid TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS username TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS cv BYTEA`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS lskillset TEXT`);
    // Ensure linkedinurl column exists for lookups
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS linkedinurl TEXT`);
    // Ensure jskillset column exists
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS jskillset TEXT`);
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS rating INTEGER`);
    // Ensure pic column exists for candidate images
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS pic BYTEA`);
    // Ensure education column exists
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS education TEXT`);
    // Ensure comment column exists
    await pool.query(`ALTER TABLE "process" ADD COLUMN IF NOT EXISTS comment TEXT`);
  } catch (err) {
    console.error('[INIT] Failed to ensure process table/columns exist:', err);
  }
}
ensureProcessTable();
// ========== END NEW ==========


// ========== NEW: Ensure login table has columns for Google OAuth (idempotent) ==========
async function ensureLoginColumns() {
  try {
    // Add columns to hold Google OAuth refresh token and optional expiry
    await pool.query(`ALTER TABLE "login" ADD COLUMN IF NOT EXISTS google_refresh_token TEXT`);
    await pool.query(`ALTER TABLE "login" ADD COLUMN IF NOT EXISTS google_token_expires TIMESTAMP`);
  } catch (err) {
    console.error('[INIT] Failed to ensure login table columns exist:', err);
  }
}
ensureLoginColumns();
// ========== END NEW ==========

// ========================= AUTHENTICATION HELPERS =========================

// Python's Werkzeug generate_password_hash often uses 'pbkdf2:sha256:iterations$salt$hash'
// This helper attempts to verify such a hash using Node built-ins.
function verifyWerkzeugHash(password, hash) {
  if (!hash) return false;
  if (!password) return false;

  const parts = hash.split('$');
  if (parts.length === 3 && parts[0].startsWith('pbkdf2:sha256')) {
    const methodParts = parts[0].split(':');
    const iterations = parseInt(methodParts[2], 10) || 260000; // default default for recent werkzeug
    const salt = parts[1];
    const originalHash = parts[2];

    const derivedKey = crypto.pbkdf2Sync(password, salt, iterations, 32, 'sha256');
    const derivedHex = derivedKey.toString('hex');
    return derivedHex === originalHash;
  }
  
  // Fallback: simple comparison or bcrypt if your DB uses bcrypt (standard $2b$ prefix)
  // If your DB has plain text (unsafe), this covers it too.
  if (hash === password) return true;
  
  return false;
}

// Authentication Middleware
const requireLogin = async (req, res, next) => {
  // Allow OPTIONS preflight
  if (req.method === 'OPTIONS') return next();

  // Check cookies
  const userid = req.cookies.userid;
  const username = req.cookies.username;

  if (!userid || !username) {
    return res.status(401).json({ error: 'Unauthorized', message: 'Authentication required' });
  }

  // Validate existence in DB (optional security hardening)
  // For performance, we trust the cookie presence + matching pair, or you can query DB.
  // Here we trust the presence to mimic the simpler flask cookie check pattern.
  req.user = { id: userid, username: username };
  next();
};

// ========================= AUTH ROUTES =========================

app.post('/login', async (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).json({ ok: false, error: "Missing credentials" });
  }

  try {
    const result = await pool.query('SELECT * FROM login WHERE username = $1', [username]);
    if (result.rows.length === 0) {
      return res.status(401).json({ ok: false, error: "Invalid username or password" });
    }

    const user = result.rows[0];
    const storedHash = user.password; // Assumes column name is 'password'

    const isValid = verifyWerkzeugHash(password, storedHash);
    
    if (!isValid) {
      return res.status(401).json({ ok: false, error: "Invalid username or password" });
    }

    // Success
    const uid = user.id || user.userid || user.username;
    
    // Set cookies (standard options)
    const cookieOpts = { maxAge: 2592000000, httpOnly: false, path: '/' }; // httpOnly false so client JS can read if needed, or secure true in prod
    res.cookie('username', user.username, cookieOpts);
    res.cookie('userid', uid, cookieOpts);

    res.json({
      ok: true,
      userid: uid,
      username: user.username,
      full_name: user.full_name || user.username
    });

  } catch (err) {
    console.error('Login error:', err);
    res.status(500).json({ ok: false, error: "Internal login error" });
  }
});

app.post('/logout', (req, res) => {
  res.clearCookie('username');
  res.clearCookie('userid');
  res.json({ ok: true, message: "Logged out" });
});

app.get('/user/resolve', async (req, res) => {
  const userid = req.cookies.userid;
  const username = req.cookies.username;
  
  if (userid && username) {
    // UPDATED: query full_name from DB instead of just returning cookies
    try {
      const r = await pool.query('SELECT full_name FROM login WHERE username = $1', [username]);
      const full_name = (r.rows.length > 0 && r.rows[0].full_name) ? r.rows[0].full_name : "";
      return res.json({ ok: true, userid, username, full_name });
    } catch(e) {
      // Fallback if DB fails
      return res.json({ ok: true, userid, username });
    }
  }
  
  // Fallback for query param check if needed similar to Flask
  const qName = req.query.username;
  if (qName) {
     try {
       const result = await pool.query('SELECT id, username, full_name FROM login WHERE username = $1', [qName]);
       if (result.rows.length > 0) {
         const u = result.rows[0];
         return res.json({ ok: true, userid: u.id, username: u.username, full_name: u.full_name });
       }
     } catch(e) {}
  }

  res.status(401).json({ ok: false });
});

// GET /user-tokens - Fetch user token information from login table
// NOTE: Consider adding rate limiting for this endpoint in production
app.get('/user-tokens', requireLogin, async (req, res) => {
  try {
    const username = req.user.username;
    const result = await pool.query('SELECT token FROM login WHERE username = $1', [username]);
    
    if (result.rows.length > 0) {
      const accountTokens = result.rows[0].token || 0;
      // For now, tokensLeft is the same as accountTokens
      // You can add separate logic if needed
      return res.json({ 
        accountTokens: accountTokens,
        tokensLeft: accountTokens 
      });
    }
    
    res.json({ accountTokens: 0, tokensLeft: 0 });
  } catch (err) {
    console.error('Error fetching user tokens:', err);
    res.status(500).json({ error: 'Failed to fetch tokens' });
  }
});

// ========================= END AUTH ROUTES =========================

app.get('/', (req, res) => {
  res.send('Backend API is running!');
});

app.get('/skillset-mapping', (req, res) => {
  try {
    if (!fs.existsSync(mappingPath)) {
      return res.status(404).json({ error: 'skillset-mapping.json not found.' });
    }
    const raw = fs.readFileSync(mappingPath, 'utf8');
    const json = JSON.parse(raw);
    res.json(json);
  } catch (err) {
    console.error('Read skillset-mapping error:', err);
    res.status(500).json({ error: 'Failed to read skillset mapping.' });
  }
});

// === Helpers for ingestion normalization (Project_Title/Project_Date restoration) ===
function firstVal(obj, keys = []) {
  for (const k of keys) {
    if (Object.prototype.hasOwnProperty.call(obj, k) && obj[k] != null && String(obj[k]).trim() !== '') {
      return obj[k];
    }
  }
  return undefined;
}

// Parse to YYYY-MM-DD; supports SG DD/MM/YYYY and Excel serials
function toISODate(value) {
  if (value == null || value === '') return null;

  // Numeric Excel serial
  if (typeof value === 'number' && Number.isFinite(value)) {
    const epoch = new Date(Date.UTC(1899, 11, 30));
    const dt = new Date(epoch.getTime() + value * 86400000);
    if (!isNaN(dt.getTime())) {
      const yyyy = dt.getUTCFullYear();
      const mm = String(dt.getUTCMonth() + 1).padStart(2, '0');
      const dd = String(dt.getUTCDate()).padStart(2, '0');
      return `${yyyy}-${mm}-${dd}`;
    }
  }

  if (value instanceof Date && !isNaN(value.getTime())) {
    const yyyy = value.getUTCFullYear();
    const mm = String(value.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(value.getUTCDate()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd}`;
  }

  if (typeof value === 'string') {
    const v = value.trim();

    // ISO or starts with ISO
    const iso = v.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (iso) return `${iso[1]}-${iso[2]}-${iso[3]}`;

    // DD/MM/YYYY or DD-MM-YYYY
    const sg = v.match(/^(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})$/);
    if (sg) {
      const dd = sg[1].padStart(2, '0');
      const mm = sg[2].padStart(2, '0');
      const yyyy = sg[3];
      return `${yyyy}-${mm}-${dd}`;
    }

    const dt = new Date(v);
    if (!isNaN(dt.getTime())) {
      const yyyy = dt.getUTCFullYear();
      const mm = String(dt.getUTCMonth() + 1).padStart(2, '0');
      const dd = String(dt.getUTCDate()).padStart(2, '0');
      return `${yyyy}-${mm}-${dd}`;
    }
  }
  return null;
}

function normalizeIncomingRow(c) {
  return {
    name: firstVal(c, ['name', 'Name']) || '',
    role: firstVal(c, ['role', 'Role']) || '',
    // Accept multiple job title keys (some inputs use 'jobtitle' already)
    jobtitle: firstVal(c, ['jobtitle', 'job_title', 'Job Title', 'role', 'Role']) || '',
    organisation: firstVal(c, ['organisation', 'Organisation']) || '',
    sector: firstVal(c, ['sector', 'Sector']) || '',
    job_family: firstVal(c, ['job_family', 'Job Family']) || '',
    role_tag: firstVal(c, ['role_tag', 'Role Tag']) || '',
    skillset: firstVal(c, ['skillset', 'Skillset']) || '',
    geographic: firstVal(c, ['geographic', 'Geographic']) || '',
    country: firstVal(c, ['country', 'Country']) || '',
    email: firstVal(c, ['email', 'Email']) || '',
    mobile: firstVal(c, ['mobile', 'Mobile']) || '',
    office: firstVal(c, ['office', 'Office']) || '',
    personal: firstVal(c, ['personal', 'Personal']) || '',
    seniority: firstVal(c, ['seniority', 'Seniority']) || '',
    sourcing_status: firstVal(c, ['sourcing_status', 'Sourcing Status']) || '',
    product: firstVal(c, ['product', 'Product', 'type']) || null,
    linkedinurl: firstVal(c, ['linkedinurl', 'linkedin', 'LinkedIn', 'URL']) || '', // Added for capture
    cv: firstVal(c, ['cv', 'CV', 'resume', 'Resume']) || ''
  };
}

// Mapping from normalized candidate-style keys to process table columns
const processColumnMap = {
  name: 'name',
  role: 'jobtitle',
  jobtitle: 'jobtitle',
  organisation: 'company',
  sector: 'sector',
  job_family: 'jobfamily',
  role_tag: 'role_tag',
  skillset: 'skillset',
  geographic: 'geographic',
  country: 'country',
  email: 'email',
  mobile: 'mobile',
  office: 'office',
  personal: 'personal',
  seniority: 'seniority',
  sourcing_status: 'sourcingstatus',
  product: 'product',
  linkedinurl: 'linkedinurl' // Added
};

// ========== UPDATED: BULK INGESTION supports Project_Title and Project_Date and writes to process table ==========
app.post('/candidates/bulk', requireLogin, async (req, res) => {
  let candidates = req.body.candidates;
  console.log('==== Bulk Upload Candidates ====');
  console.log('Received candidates:', JSON.stringify(candidates, null, 2));
  if (!Array.isArray(candidates) || candidates.length === 0) {
    console.log('No candidates data provided!');
    return res.status(400).json({ error: 'No candidates data provided.' });
  }

  candidates = candidates.filter(
    c => Object.values(c).some(val => val && String(val).trim() !== '')
  );

  if (candidates.length === 0) {
    console.log('No valid candidates found!');
    return res.status(400).json({ error: 'No valid candidates found.' });
  }

  // Normalize each row to include canonical and legacy fields
  const normalized = candidates.map(normalizeIncomingRow);

  // Canonical + legacy insertion keys (normalized)
  const normKeys = [
    'name', 'role', 'jobtitle', 'organisation', 'sector', 'job_family',
    'role_tag', 'skillset', 'geographic', 'country',
    'email', 'mobile', 'office', 'personal',
    'seniority', 'sourcing_status', 'product', 'linkedinurl'
  ];

  try {
    // Fetch user's JD skill from login table using USERNAME (more reliable)
    let userJskillset = null;
    try {
      const ures = await pool.query('SELECT jskillset FROM login WHERE username = $1', [req.user.username]);
      if (ures.rows.length > 0) userJskillset = ures.rows[0].jskillset || null;
    } catch (e) {
      console.warn('[BULK] unable to fetch user jskillset via username', e && e.message);
      userJskillset = null;
    }

    // Map normalized keys to process table column names
    const processCols = normKeys.map(k => processColumnMap[k] || k);
    
    // ADD userid/username from session
    processCols.push('userid');
    processCols.push('username');

    // NEW: push jskillset into columns so it will be stored per inserted row
    processCols.push('jskillset');

    const values = [];
    const placeholders = normalized.map((row, i) => {
      const start = i * processCols.length + 1;
      // existing per-row values (normKeys -> processCols except last three we handled)
      processCols.slice(0, -3).forEach((col, idx) => {
        const k = normKeys[idx];
        let v = Object.prototype.hasOwnProperty.call(row, k) ? row[k] : null;
        if (v === '') v = null;
	if (k === 'seniority' && v != null && String(v).trim() !== '') {
	 const std = standardizeSeniority(v);
	 v = std || null;
	}
        values.push(v);
      });
      // Add user info
      values.push(req.user.id);
      values.push(req.user.username);
      
      // Add jskillset value for this user (same for each row)
      values.push(userJskillset);
      
      return `(${Array.from({ length: processCols.length }, (_, j) => `$${start + j}`).join(',')})`;
    }).join(',');

    const sql = `
      INSERT INTO "process" (${processCols.join(', ')})
      VALUES ${placeholders}
      RETURNING id
    `;

    const result = await pool.query(sql, values);
    console.log('Inserted rows into process:', result.rowCount);

    // Persist canonical company/personal for each inserted row (use returned ids & normalized inputs)
    try {
      const returnedRows = result.rows || [];
      for (let i = 0; i < returnedRows.length; i++) {
        const insertedId = returnedRows[i].id;
        const src = normalized[i];
        const currentCompany = src.organisation || src.company || null;
        const currentJobTitle = src.jobtitle || src.role || '';
        const currentPersonal = src.personal || null;
        await ensureCanonicalFieldsForId(insertedId, currentCompany, currentJobTitle, currentPersonal);
      }
    } catch (e) {
      console.warn('[BULK_CANON] failed to persist canonical fields for inserted rows', e && e.message);
    }

    // Notify clients that new candidates were inserted (clients can choose to refetch)
    try {
      broadcastSSE('candidates_changed', { action: 'bulk_insert', count: result.rowCount });
    } catch (_) { /* ignore emit errors */ }

    res.json({ rowsInserted: result.rowCount });
  } catch (err) {
    console.error('=== Bulk insert error! ===');
    console.error(err);
    res.status(500).json({ error: err.message || 'Bulk insert failed.' });
  }
});

// GET /candidates: return process rows but include candidate-style fallback keys
// UPDATED: Filter by userid to ensure user only sees their own records
app.get('/candidates', requireLogin, async (req, res) => {
  try {
    // Always restrict to the authenticated user's records
    const result = await pool.query('SELECT * FROM "process" WHERE userid = $1 ORDER BY id DESC', [String(req.user.id)]);
    const rows = result.rows.map(r => {
      // ensure both process-style and candidate-style keys are present for frontend compatibility
      const companyCanonical = normalizeCompanyName(r.company || r.organisation || '');
      // if personal empty, generate from jobtitle
      const personalFallback = (r.personal && String(r.personal).trim()) ? String(r.personal).trim() : (r.jobtitle ? canonicalJobTitle(r.jobtitle) : null);
      
      // Convert bytea pic to base64 string for frontend
      let picBase64 = null;
      if (r.pic && Buffer.isBuffer(r.pic)) {
        picBase64 = r.pic.toString('base64');
      }
      
      return {
        ...r,
        // process-style keys explicit (helpful for debugging)
        jobtitle: r.jobtitle ?? null,
        company: companyCanonical ?? (r.company ?? null),
        jobfamily: r.jobfamily ?? null,
        sourcingstatus: r.sourcingstatus ?? null,
        product: r.product ?? null,
        lskillset: r.lskillset ?? null, // ensure lskillset is available
        linkedinurl: r.linkedinurl ?? null,
        jskillset: r.jskillset ?? null, // return jskillset for frontend if needed
        pic: picBase64, // Convert bytea to base64 for frontend

        // candidate-style fields mapped from process columns if missing
        role: r.role ?? r.jobtitle ?? null,
        organisation: companyCanonical ?? (r.organisation ?? r.company ?? null),
        job_family: r.job_family ?? r.jobfamily ?? null,
        sourcing_status: r.sourcing_status ?? r.sourcingstatus ?? null,
        // type maps to product
        type: r.product ?? null,
        // personal: prefer the DB personal if present, otherwise fallback to canonicalized jobtitle
        personal: (r.personal && String(r.personal).trim()) ? String(r.personal).trim() : personalFallback
      };
    });
    res.json(rows);
  } catch (err) {
    console.error('Fetch process rows error:', err);
    res.status(500).json({ error: 'Failed to fetch candidates/process rows.' });
  }
});

// GET /candidates/:id/cv - Secure CV Fetch by ID (Keep existing)
app.get('/candidates/:id/cv', requireLogin, async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) return res.status(400).send('Invalid ID');

  try {
    // Ownership guard
    const q = await pool.query('SELECT userid, cv FROM "process" WHERE id = $1', [id]);
    if (q.rows.length === 0) return res.status(404).send('No CV found');
    if (String(q.rows[0].userid) !== String(req.user.id)) return res.status(403).send('Forbidden');

    const cv = q.rows[0].cv;

    if (!cv) {
      return res.status(404).send('No CV found');
    }

    // Handle Buffer (Postgres BYTEA)
    if (Buffer.isBuffer(cv)) {
        res.setHeader('Content-Type', 'application/pdf');
        // Optional: Check magic bytes for PDF to be sure, otherwise default to pdf
        return res.send(cv);
    }

    // Handle String (Base64 or File Path)
    if (typeof cv === 'string') {
        // If it's a data URI
        if (cv.startsWith('data:')) {
            const matches = cv.match(/^data:([A-Za-z-+\/]+);base64,(.+)$/);
            if (matches && matches.length === 3) {
                const type = matches[1];
                const buf = Buffer.from(matches[2], 'base64');
                res.setHeader('Content-Type', type);
                return res.send(buf);
            }
        }
        try {
           const buf = Buffer.from(cv, 'base64');
           res.setHeader('Content-Type', 'application/pdf');
           return res.send(buf);
        } catch (e) {
           // Not base64
        }
    }

    // Fallback
    res.status(500).send('Unknown CV format');

  } catch (err) {
    console.error('CV fetch error:', err);
    res.status(500).send('Server Error');
  }
});

// ========== NEW: GET /process/download_cv - Secure CV Fetch by LinkedIn URL ==========
app.get('/process/download_cv', requireLogin, async (req, res) => {
  const linkedinUrl = req.query.linkedin;
  if (!linkedinUrl) {
    return res.status(400).send('Missing linkedin parameter');
  }

  try {
    // Fetch process row and ensure ownership
    const result = await pool.query('SELECT cv, userid FROM "process" WHERE linkedinurl = $1', [linkedinUrl]);
    
    // If exact match fails, try relaxed match (without query params or trailing slash)
    if (result.rows.length === 0) {
        const relaxed = linkedinUrl.split('?')[0].replace(/\/+$/, '');
        const retry = await pool.query('SELECT cv, userid FROM "process" WHERE linkedinurl LIKE $1', [relaxed + '%']);
        if (retry.rows.length > 0) {
             if (String(retry.rows[0].userid) !== String(req.user.id)) return res.status(403).send('Forbidden');
             if (!retry.rows[0].cv) return res.status(404).send('No CV found');
             return serveCV(res, retry.rows[0].cv);
        }
        return res.status(404).send('No CV found for this profile');
    }

    if (String(result.rows[0].userid) !== String(req.user.id)) {
      return res.status(403).send('Forbidden');
    }

    if (!result.rows[0].cv) {
      return res.status(404).send('No CV found');
    }

    serveCV(res, result.rows[0].cv);

  } catch (err) {
    console.error('/process/download_cv error:', err);
    res.status(500).send('Server Error');
  }
});

function serveCV(res, cv) {
    // Handle Buffer (Postgres BYTEA)
    if (Buffer.isBuffer(cv)) {
        res.setHeader('Content-Type', 'application/pdf');
        return res.send(cv);
    }

    // Handle String (Base64)
    if (typeof cv === 'string') {
        if (cv.startsWith('data:')) {
            const matches = cv.match(/^data:([A-Za-z-+\/]+);base64,(.+)$/);
            if (matches && matches.length === 3) {
                const type = matches[1];
                const buf = Buffer.from(matches[2], 'base64');
                res.setHeader('Content-Type', type);
                return res.send(buf);
            }
        }
        try {
           const buf = Buffer.from(cv, 'base64');
           res.setHeader('Content-Type', 'application/pdf');
           return res.send(buf);
        } catch (e) { }
    }
    res.status(500).send('Unknown CV format');
}

app.delete('/candidates/:id', requireLogin, async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) {
    return res.status(400).json({ error: 'Invalid candidate id.' });
  }

  // Ownership guard
  const ownerOk = await ensureOwnershipOrFail(res, id, req.user.id);
  if (!ownerOk) return;

  try {
    const result = await pool.query('DELETE FROM "process" WHERE id = $1 RETURNING id', [id]);
    if (result.rowCount === 0) {
      return res.status(404).json({ error: 'Candidate not found.' });
    }

    // Emit deletion event so connected clients can react if they listen
    try {
      broadcastSSE('candidate_deleted', { id });
      broadcastSSE('candidates_changed', { action: 'delete', ids: [id] });
    } catch (_) { /* ignore emit errors */ }

    res.json({ deleted: id });
  } catch (err) {
    console.error('Delete process row error:', err);
    res.status(500).json({ error: 'Failed to delete candidate/process row.' });
  }
});

app.post('/candidates/bulk-delete', requireLogin, async (req, res) => {
  const { ids } = req.body;
  console.log('[API] bulk-delete received ids:', ids);

  if (!Array.isArray(ids) || ids.length === 0) {
    return res.status(400).json({ error: 'No valid candidate ids provided.' });
  }

  const cleanIds = ids
    .map(id => {
      const n = typeof id === 'number' ? id : parseInt(id, 10);
      return Number.isInteger(n) && n > 0 ? n : null;
    })
    .filter(n => n !== null);

  console.log('[API] bulk-delete cleanIds (numeric):', cleanIds);

  if (cleanIds.length === 0) {
    return res.status(400).json({
      error: 'No valid candidate ids provided. Expecting numeric ids only.',
      received: ids
    });
  }

  try {
    // Only delete rows that belong to the requesting user
    const result = await pool.query(
      'DELETE FROM "process" WHERE id = ANY($1::int[]) AND userid = $2 RETURNING id',
      [cleanIds, String(req.user.id)]
    );
    console.log('[API] bulk-delete deletedCount:', result.rowCount);

    // emit event to notify clients
    try {
      broadcastSSE('candidates_changed', { action: 'bulk_delete', ids: result.rows.map(r => r.id) });
    } catch (_) { /* ignore */ }

    res.json({ deletedCount: result.rowCount, attempted: cleanIds.length, ids: result.rows.map(r => r.id) });
  } catch (err) {
    console.error('Bulk delete error:', err);
    res.status(500).json({ error: 'Bulk delete failed.' });
  }
});

app.post('/generate-skillsets', requireLogin, async (req, res) => {
  try {
    if (!fs.existsSync(mappingPath)) {
      return res.status(500).json({ error: 'Skillset mapping file not found.' });
    }
    const raw = fs.readFileSync(mappingPath, 'utf8');
    const skillsetMap = JSON.parse(raw);

    const candidates = (await pool.query('SELECT id, role_tag, skillset FROM "process"')).rows;

    let updatedCount = 0;
    for (const candidate of candidates) {
      const roleTag = candidate.role_tag ? candidate.role_tag.trim() : '';
      const newSkillset = skillsetMap[roleTag] || '';
      if (newSkillset && newSkillset !== candidate.skillset) {
        await pool.query(
          'UPDATE "process" SET skillset = $1 WHERE id = $2',
          [newSkillset, candidate.id]
        );
        updatedCount++;
      }
    }

    // Let clients know skillsets changed (they can refetch)
    try {
      broadcastSSE('candidates_changed', { action: 'skillset_update', count: updatedCount });
    } catch (_) { /* ignore */ }

    res.json({ message: `Skillsets generated for ${updatedCount} process rows.` });
  } catch (err) {
    console.error('Skillset generation error:', err);
    res.status(500).json({ error: 'Failed to generate skillsets.' });
  }
});

app.get('/org-chart', requireLogin, (req, res) => {
  res.json([{ name: 'Sample Org Chart' }]);
});

/**
 * POST /candidates
 * Create a new process row. Accepts candidate-style keys (role, organisation, job_family, sourcing_status, type)
 * or process-style keys (jobtitle, company, jobfamily, sourcingstatus, product). Returns the created row.
 */
app.post('/candidates', requireLogin, async (req, res) => {
  const body = req.body || {};

  // Acceptable mapping for create (candidate-style -> process column)
  const createFieldMap = {
    // candidate -> process
    role: 'jobtitle',
    jobtitle: 'jobtitle',
    organisation: 'company',
    job_family: 'jobfamily',
    sourcing_status: 'sourcingstatus',
    type: 'product',
    product: 'product',

    // process keys (pass-through)
    jobtitle: 'jobtitle',
    company: 'company',
    jobfamily: 'jobfamily',
    sourcingstatus: 'sourcingstatus',

    // same-name fields
    name: 'name',
    sector: 'sector',
    role_tag: 'role_tag',
    skillset: 'skillset',
    geographic: 'geographic',
    country: 'country',
    email: 'email',
    mobile: 'mobile',
    office: 'office',
    personal: 'personal',
    seniority: 'seniority',
    lskillset: 'lskillset',
    linkedinurl: 'linkedinurl',
    comment: 'comment'
  };

  // Build columns and values for insert
  const cols = [];
  const values = [];
  const placeholders = [];
  let idx = 1;

  for (const key of Object.keys(body)) {
  if (!Object.prototype.hasOwnProperty.call(createFieldMap, key)) continue;
  let col = createFieldMap[key];
  let val = body[key];

  // Canonicalize seniority on create
  if (key === 'seniority' && val != null && String(val).trim() !== '') {
    const std = standardizeSeniority(val);
    // persist only canonical value (or null if unrecognized)
    val = std || null;
  }

  // normalize empty string to null
  if (val === '') val = null;

  cols.push(`"${col}"`);
  values.push(val);
  placeholders.push(`$${idx}`);
  idx++;
}
  // Inject User info
  cols.push(`"userid"`);
  values.push(req.user.id);
  placeholders.push(`$${idx++}`);

  cols.push(`"username"`);
  values.push(req.user.username);
  placeholders.push(`$${idx++}`);

  // Fetch user's JD skill from login table (jskillset) using USERNAME (more reliable)
  let userJskillset = null;
  try {
    const ures = await pool.query('SELECT jskillset FROM login WHERE username = $1', [req.user.username]);
    if (ures.rows.length > 0) userJskillset = ures.rows[0].jskillset || null;
  } catch (e) {
    console.warn('[POST /candidates] unable to fetch user jskillset via username', e && e.message);
    userJskillset = null;
  }

  // NEW: include jskillset column + value
  cols.push(`"jskillset"`);
  values.push(userJskillset);
  placeholders.push(`$${idx++}`);

  if (cols.length === 0) {
    return res.status(400).json({ error: 'No valid fields provided for create.' });
  }

  const sql = `INSERT INTO "process" (${cols.join(', ')}) VALUES (${placeholders.join(', ')}) RETURNING *`;

  try {
    const result = await pool.query(sql, values);
    const r = result.rows[0];

    // After insert, ensure canonical company/personal persisted for consistency
    try {
      await ensureCanonicalFieldsForId(r.id, r.company || r.organisation, r.jobtitle || r.role, r.personal);
    } catch (e) {
      console.warn('[POST_CANON] failed to persist canonical fields', e && e.message);
    }

    // Reload latest row to include persisted canonical fields
    const fresh = (await pool.query('SELECT * FROM "process" WHERE id = $1', [r.id])).rows[0];

    const mapped = {
      ...fresh,
      jobtitle: fresh.jobtitle ?? null,
      company: (normalizeCompanyName(fresh.company || fresh.organisation) ?? (fresh.company ?? null)),
      jobfamily: fresh.jobfamily ?? null,
      sourcingstatus: fresh.sourcingstatus ?? null,
      product: fresh.product ?? null,
      lskillset: fresh.lskillset ?? null,
      linkedinurl: fresh.linkedinurl ?? null,
      jskillset: fresh.jskillset ?? null, // include jskillset in response

      // candidate-style fallbacks
      role: fresh.role ?? fresh.jobtitle ?? null,
      organisation: (normalizeCompanyName(fresh.company || fresh.organisation) ?? (fresh.organisation ?? fresh.company ?? null)),
      job_family: fresh.job_family ?? fresh.jobfamily ?? null,
      sourcing_status: fresh.sourcing_status ?? fresh.sourcingstatus ?? null,
      type: fresh.product ?? null,
      personal: (fresh.personal && String(fresh.personal).trim()) ? String(fresh.personal).trim() : (fresh.jobtitle ? canonicalJobTitle(fresh.jobtitle) : null)
    };

    // Emit creation event
    try {
      broadcastSSE('candidate_created', mapped);
      broadcastSSE('candidates_changed', { action: 'create', id: mapped.id });
    } catch (_) { /* ignore */ }

    res.status(201).json(mapped);
  } catch (err) {
    console.error('POST /candidates error', err);
    res.status(500).json({ error: 'Create failed', detail: err.message });
  }
});

/**
 * PUT /candidates/:id
 * Update a process row. Accepts either candidate-style keys (role, organisation, job_family, sourcing_status)
 * or process-style keys (jobtitle, company, jobfamily, sourcingstatus, product). Writes to process table.
 */
app.put('/candidates/:id', requireLogin, async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (Number.isNaN(id)) return res.status(400).json({ error: 'Invalid id' });

  // Ownership guard
  const ownerOk = await ensureOwnershipOrFail(res, id, req.user.id);
  if (!ownerOk) return;

  const body = req.body || {};

  const fieldMap = {
    // candidate -> process
    role: 'jobtitle',
    organisation: 'company',
    job_family: 'jobfamily',
    sourcing_status: 'sourcingstatus',
    product: 'product',
    type: 'product', // MAP frontend "type" to backend "product"

    // process keys (pass-through)
    jobtitle: 'jobtitle',
    company: 'company',
    jobfamily: 'jobfamily',
    sourcingstatus: 'sourcingstatus',

    // same-name fields
    name: 'name',
    sector: 'sector',
    role_tag: 'role_tag',
    skillset: 'skillset',
    geographic: 'geographic',
    country: 'country',
    email: 'email',
    mobile: 'mobile',
    office: 'office',
    personal: 'personal',
    seniority: 'seniority',
    lskillset: 'lskillset',
    linkedinurl: 'linkedinurl',
    comment: 'comment'
  };

  const keys = Object.keys(body).filter(k => Object.prototype.hasOwnProperty.call(fieldMap, k));
  if (keys.length === 0) {
    return res.status(400).json({ error: 'No updatable fields provided.' });
  }

  try {
    // Build unique column -> value map to avoid assigning the same DB column twice
    const colValueMap = new Map();
    for (const k of keys) {
      const col = fieldMap[k];
      let v = body[k];
      if (k === 'seniority' && v != null && String(v).trim() !== '') {
        const std = standardizeSeniority(v);
        v = std || null;
      }
      colValueMap.set(col, v === '' ? null : v);
    }

    const cols = [];
    const values = [];
    let idx = 1;
    for (const [col, val] of colValueMap.entries()) {
      cols.push(`"${col}" = $${idx}`);
      values.push(val);
      idx++;
    }
    values.push(id);

    const sql = `UPDATE "process" SET ${cols.join(', ')} WHERE id = $${idx} RETURNING *`;

    const result = await pool.query(sql, values);
    if (result.rowCount === 0) return res.status(404).json({ error: 'Not found' });

    let r = result.rows[0];

    // Persist canonical company/personal if needed after the update
    try {
      await ensureCanonicalFieldsForId(r.id, r.company || r.organisation, r.jobtitle || r.role, r.personal);
    } catch (e) {
      console.warn('[PUT_CANON] failed to persist canonical fields', e && e.message);
    }

    // Reload to reflect any canonical updates
    r = (await pool.query('SELECT * FROM "process" WHERE id = $1', [r.id])).rows[0];

    // Return row with both process-style and candidate-style fallback keys for frontend convenience
    const mapped = {
      ...r,
      // process-style explicit
      jobtitle: r.jobtitle ?? null,
      company: normalizeCompanyName(r.company || r.organisation) ?? (r.company ?? null),
      jobfamily: r.jobfamily ?? null,
      sourcingstatus: r.sourcingstatus ?? null,
      product: r.product ?? null,
      lskillset: r.lskillset ?? null,
      linkedinurl: r.linkedinurl ?? null,
      jskillset: r.jskillset ?? null,

      // candidate-style fallbacks
      role: r.role ?? r.jobtitle ?? null,
      organisation: normalizeCompanyName(r.company || r.organisation) ?? (r.organisation ?? r.company ?? null),
      job_family: r.job_family ?? r.jobfamily ?? null,
      sourcing_status: r.sourcing_status ?? r.sourcingstatus ?? null,
      type: r.product ?? null, // return product as type for frontend
      personal: (r.personal && String(r.personal).trim()) ? String(r.personal).trim() : (r.jobtitle ? canonicalJobTitle(r.jobtitle) : null)
    };

    // Emit candidate_updated via SSE if connections exist
    try {
      broadcastSSE('candidate_updated', mapped);
    } catch (e) {
      // ignore socket emit errors
    }

    res.json(mapped);
  } catch (err) {
    console.error('PUT /candidates/:id error', err);
    res.status(500).json({ error: 'Update failed', detail: err.message });
  }
});

// ========== NEW: Calculate Unmatched Skillset ==========
app.post('/candidates/:id/calculate-unmatched', requireLogin, async (req, res) => {
    const id = parseInt(req.params.id, 10);
    if (Number.isNaN(id)) return res.status(400).json({ error: 'Invalid candidate id' });

    try {
        let jdSkillsetRaw = '';
        
        // 1. Fetch JD Skillset from Process table (per-profile jskillset)
        try {
            const pRes = await pool.query('SELECT jskillset FROM "process" WHERE id = $1', [id]);
            if (pRes.rows.length > 0 && pRes.rows[0].jskillset) {
                jdSkillsetRaw = pRes.rows[0].jskillset;
            }
        } catch (e) {
             console.warn('[CALC_UNMATCHED] failed to read process.jskillset', e.message);
        }

        // 2. Fallback: Fetch User's JD Skillset from login table if process.jskillset is missing
        if (!jdSkillsetRaw) {
            try {
                // Use username for consistency
                const uRes = await pool.query('SELECT jskillset FROM login WHERE username = $1', [req.user.username]);
                if (uRes.rows.length > 0) {
                    jdSkillsetRaw = uRes.rows[0].jskillset || '';
                }
            } catch (e) {
                console.warn('[CALC_UNMATCHED] fallback login.jskillset read failed', e.message);
            }
        }
        
        // 3. Fetch Candidate's current skillset, sector, and jobfamily from process table
        const candidateRes = await pool.query('SELECT skillset, sector, jobfamily FROM "process" WHERE id = $1', [id]);
        if (candidateRes.rows.length === 0) {
            return res.status(404).json({ error: 'Candidate not found.' });
        }
        const candidateSkillsetRaw = candidateRes.rows[0].skillset || '';
        const sectorRaw = candidateRes.rows[0].sector ? String(candidateRes.rows[0].sector).trim() : 'Unknown';
        const jobFamilyRaw = candidateRes.rows[0].jobfamily ? String(candidateRes.rows[0].jobfamily).trim() : 'Unknown';

        // 4. Use Gemini to Calculate Unmatched Skillset
        if (!GoogleGenerativeAIClass) {
             return res.status(500).json({ error: "Gemini SDK not installed." });
        }
        const apiKey = process.env.GOOGLE_API_KEY;
        if (!apiKey) return res.status(500).json({ error: 'Gemini API key not configured.' });

        const genAI = new GoogleGenerativeAIClass(apiKey);
        const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });

        const prompt = `
            Compare the Job Description (JD) Skillset and the Candidate Skillset below.
            Context:
            - Sector: "${sectorRaw}"
            - Job Family: "${jobFamilyRaw}"

            Identify the skills that are present in the JD Skillset but are MISSING or UNMATCHED in the Candidate Skillset.
            
            JD Skillset: "${jdSkillsetRaw}"
            Candidate Skillset: "${candidateSkillsetRaw}"
            
            Return the result as a simple list. Do NOT include any introductory or explanatory text.
        `;

        const result = await model.generateContent(prompt);
        const rawText = result.response.text();

        // 5. Data Cleansing
        // Strip explanatory text using strict patterns
        let cleaned = rawText.replace(/^(Here are|The following|These are).*?[:\n]/gim, '');
        cleaned = cleaned.replace(/Here are the skills present in the JD Skillset but missing or unmatched in the Candidate Skillset[:\s]*/i, '');
        
        // Remove JSON structural chars
        cleaned = cleaned.replace(/[\[\]"']/g, '');
        
        // Replace newlines and commas with semicolons
        cleaned = cleaned.replace(/[\n\r,]+/g, ';');
        
        // Split, trim, remove leading bullets (hyphens), and filter empty
        const tokens = cleaned
          .split(';')
          .map(s => s.trim().replace(/^[-*•]\s+/, '').replace(/^[-*•]/, '')) // Remove leading bullet/hyphen
          .filter(s => s.length > 0);
        
        const unmatchedStr = tokens.join('; ');

        // 6. Update process table column 'lskillset' ONLY
        const updateRes = await pool.query(
            'UPDATE "process" SET lskillset = $1 WHERE id = $2 RETURNING *',
            [unmatchedStr, id]
        );

        const r = updateRes.rows[0];

        // 7. Return standard updated object
        // Use standard mapping helper logic manually here to ensure consistency
        const companyCanonical = normalizeCompanyName(r.company || r.organisation || '');
        const personalFallback = (r.personal && String(r.personal).trim()) ? String(r.personal).trim() : (r.jobtitle ? canonicalJobTitle(r.jobtitle) : null);
        
        const mapped = {
            ...r,
            jobtitle: r.jobtitle ?? null,
            company: companyCanonical ?? (r.company ?? null),
            lskillset: r.lskillset ?? null,
            linkedinurl: r.linkedinurl ?? null,
            jskillset: r.jskillset ?? null,
            
            // fallbacks
            role: r.role ?? r.jobtitle ?? null,
            organisation: companyCanonical ?? (r.organisation ?? r.company ?? null),
            type: r.product ?? null,
            personal: (r.personal && String(r.personal).trim()) ? String(r.personal).trim() : personalFallback
        };

        // Emit update
        try {
            broadcastSSE('candidate_updated', mapped);
        } catch (_) {}

        res.json({ lskillset: unmatchedStr, fullUpdate: mapped });

    } catch (err) {
        console.error('Calculate unmatched error:', err);
        res.status(500).json({ error: 'Failed to calculate unmatched skillset', detail: err.message });
    }
});

// ========== NEW: Assess Unmatched Skills via Gemini ==========
app.post('/candidates/:id/assess-unmatched', requireLogin, async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!Number.isInteger(id) || id <= 0) return res.status(400).json({ error: 'Invalid id' });

    if (!GoogleGenerativeAIClass) {
      return res.status(500).json({ error: "Gemini SDK not installed." });
    }
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) return res.status(500).json({ error: 'Gemini API key not configured.' });

    const { source = 'candidate', sourceSkills = [], unmatched = [] } = req.body;
    // sourceSkills = Canonical/JD skills
    // unmatched = Raw tokens found in lskillset or provided list
    
    if (!Array.isArray(unmatched) || !unmatched.length) {
      return res.status(400).json({ error: 'No unmatched skills provided.' });
    }

    // Build an instruction telling Gemini to compare the two lists and classify each unmatched token
    const instruction = `
You are a skill matching assistant. Inputs:
- sourceSkills: canonical skillset list (comma-separated): ${JSON.stringify(sourceSkills)}
- unmatched: list of tokens to check (array): ${JSON.stringify(unmatched)}

For each entry in unmatched, return JSON item:
{ "original": "<raw token>", "normalized": "<canonical label or null>", "verdict": "<true-missing|synonym|ignore>", "mappedTo": "<if synonym then canonical skill>" }

Return JSON only:
{ "suggestions": [ ... ] }
    `;

    const genAI = new GoogleGenerativeAIClass(apiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });
    
    const result = await model.generateContent(instruction);
    const text = result.response && typeof result.response.text === 'function' ? result.response.text() : '';

    // Attempt to robustly extract JSON
    const cleaned = text.replace(/```(?:json)?/g, '').trim();
    let parsed;
    try {
      parsed = JSON.parse(cleaned);
    } catch (e) {
      const match = cleaned.match(/\{[\s\S]*\}/);
      if (match) parsed = JSON.parse(match[0]);
    }
    if (!parsed || !Array.isArray(parsed.suggestions)) {
      // Fallback if parsing fails or structure is wrong
      return res.status(500).json({ error: 'AI response parse failed.', raw: text });
    }

    // Normalize result structure
    parsed.suggestions = parsed.suggestions.map(s => ({
      original: s.original || s.o || '',
      normalized: s.normalized || s.normal || null,
      verdict: s.verdict || 'true-missing',
      mappedTo: s.mappedTo || s.mapped || null
    }));

    res.json(parsed);
  } catch (err) {
    console.error('/assess-unmatched error', err);
    res.status(500).json({ error: 'Assessment failed' });
  }
});

/**
 * POST /candidates/bulk-update
 * Accept an array of candidate objects to update in the "process" table.
 * Each item must include a numeric "id" and any updatable fields. Uses the same field mapping as PUT /candidates/:id.
 * Returns the list of updated rows.
 */
app.post('/candidates/bulk-update', requireLogin, async (req, res) => {
  const rows = Array.isArray(req.body?.rows) ? req.body.rows : [];
  if (!rows.length) return res.status(400).json({ error: 'No rows provided.' });

  // Field mapping identical to the single PUT endpoint mapping
  const fieldMap = {
    role: 'jobtitle',
    organisation: 'company',
    job_family: 'jobfamily',
    sourcing_status: 'sourcingstatus',
    product: 'product',
    type: 'product',
    jobtitle: 'jobtitle',
    company: 'company',
    jobfamily: 'jobfamily',
    sourcingstatus: 'sourcingstatus',
    name: 'name',
    sector: 'sector',
    role_tag: 'role_tag',
    skillset: 'skillset',
    geographic: 'geographic',
    country: 'country',
    email: 'email',
    mobile: 'mobile',
    office: 'office',
    personal: 'personal',
    seniority: 'seniority',
    lskillset: 'lskillset',
    linkedinurl: 'linkedinurl'
  };

  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const updatedRows = [];
    for (const item of rows) {
      const id = Number(item?.id);
      if (!Number.isInteger(id) || id <= 0) continue;

      const keys = Object.keys(item).filter(k => k !== 'id' && Object.prototype.hasOwnProperty.call(fieldMap, k));
      if (!keys.length) continue;

      // Ownership check: skip rows not owned by this user
      try {
        const ownerQ = await client.query('SELECT userid FROM "process" WHERE id = $1', [id]);
        if (ownerQ.rows.length === 0) continue; // row doesn't exist
        if (String(ownerQ.rows[0].userid) !== String(req.user.id)) {
          // Skip updating rows not owned by user (optional: collect skipped ids)
          continue;
        }
      } catch (e) {
        console.warn('[BULK_UPDATE_AUTH] failed ownership check for id', id, e && e.message);
        continue;
      }

      // Build unique column -> value map to prevent multiple assignments to same column
      const colValueMap = new Map();
      for (const k of keys) {
        const col = fieldMap[k];
        let v = item[k];
        if (k === 'seniority' && v != null && String(v).trim() !== '') {
          const std = standardizeSeniority(v);
          v = std || null;
        }
        colValueMap.set(col, v === '' ? null : v);
      }

      const cols = [];
      const values = [];
      let idx = 1;
      for (const [col, val] of colValueMap.entries()) {
        cols.push(`"${col}" = $${idx}`);
        values.push(val);
        idx++;
      }
      values.push(id);
      const sql = `UPDATE "process" SET ${cols.join(', ')} WHERE id = $${idx} RETURNING *`;
      // eslint-disable-next-line no-await-in-loop
      const result = await client.query(sql, values);
      if (result.rowCount === 1) {
        let r = result.rows[0];
        // Persist canonical fields for this updated row
        try {
          await ensureCanonicalFieldsForId(r.id, r.company || r.organisation, r.jobtitle || r.role, r.personal);
          // reload to reflect persisted canonicalization
          r = (await client.query('SELECT * FROM "process" WHERE id = $1', [r.id])).rows[0];
        } catch (e) {
          console.warn('[BULK_UPDATE_CANON] failed for id', r.id, e && e.message);
        }

        const mapped = {
          ...r,
          jobtitle: r.jobtitle ?? null,
          company: normalizeCompanyName(r.company || r.organisation) ?? (r.company ?? null),
          jobfamily: r.jobfamily ?? null,
          sourcingstatus: r.sourcingstatus ?? null,
          product: r.product ?? null,
          lskillset: r.lskillset ?? null,
          role: r.role ?? r.jobtitle ?? null,
          organisation: normalizeCompanyName(r.company || r.organisation) ?? (r.organisation ?? r.company ?? null),
          job_family: r.job_family ?? r.jobfamily ?? null,
          sourcing_status: r.sourcing_status ?? r.sourcingstatus ?? null,
          type: r.product ?? null,
          personal: (r.personal && String(r.personal).trim()) ? String(r.personal).trim() : (r.jobtitle ? canonicalJobTitle(r.jobtitle) : null),
          jskillset: r.jskillset ?? null
        };
        updatedRows.push(mapped);
      }
    }
    await client.query('COMMIT');

    // Emit change notification
    try {
      broadcastSSE('candidates_changed', { action: 'bulk_update', count: updatedRows.length });
      for (const u of updatedRows) {
        broadcastSSE('candidate_updated', u);
      }
    } catch (e) { /* ignore */ }

    res.json({ updatedCount: updatedRows.length, rows: updatedRows });
  } catch (err) {
    await client.query('ROLLBACK');
    console.error('Bulk update error:', err);
    res.status(500).json({ error: 'Bulk update failed.' });
  } finally {
    client.release();
  }
});

/**
 * ========== Data Verification (Company & Job Title Standardization via Gemini 2.5 Flash Lite) ==========
 * Endpoint: POST /verify-data
 * Body: { rows: [ { id, organisation, jobtitle?, seniority?, geographic?, country? } ] }
 * Response: { corrected: [ { id, organisation?, company?, jobtitle?, standardized_job_title?, personal?, seniority?, geographic?, country? } ] }
 *
 * This endpoint sends organisation, job title, seniority, geographic and country data to Gemini
 * to standardize them (e.g. normalize company names, categorize job titles, normalize countries).
 */
app.post('/verify-data', requireLogin, async (req, res) => {
  const { rows } = req.body;
  if (!rows || !Array.isArray(rows) || rows.length === 0) {
    return res.status(400).json({ error: 'No rows provided.' });
  }

  // Check for SDK
  if (!GoogleGenerativeAIClass) {
    return res.status(500).json({ error: "Gemini SDK not installed on server." });
  }
  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'Gemini API key not configured.' });
  }

  try {
    const genAI = new GoogleGenerativeAIClass(apiKey);
    // Use gemini-2.5-flash-lite if available, otherwise gemini-1.5-flash
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });

    // Construct prompt for batch
    // We send subset of fields: id, organisation, jobtitle (or role), seniority, geographic, country
    const lines = rows.map(r => {
      const org = r.organisation || r.company || '';
      const title = r.jobtitle || r.role || '';
      const sen = r.seniority || '';
      const geo = r.geographic || '';
      const country = r.country || '';
      return JSON.stringify({ id: r.id, org, title, sen, geo, country });
    });

    const prompt = `
      You are a data standardization assistant.
      I will provide a JSON list of candidate records with fields: id, org (company), title (job title), sen (seniority), geo (geographic region), country.
      
      Your task:
      1. Standardize "org" to the canonical company name (e.g. "Tencent Gaming" -> "Tencent", "Tencent Cloud" -> "Tencent", "Mihoyo Co Ltd" -> "Mihoyo").
      2. Standardize "title" to a standard job title (e.g. "Cloud Specialist" -> "Cloud Engineer", "Cloud Developer" -> "Cloud Engineer", but "Cloud Architect" remains "Cloud Architect").
      3. Infer or standardize "sen" (seniority) to one of: Junior, Mid, Senior, Lead, Manager, Director, Expert, Executive.
      4. Standardize "country" to canonical country names (e.g. "South Korea" -> "Korea", "USA" -> "United States").
      5. Return a JSON list of objects with keys: "id", "organisation" (standardized), "jobtitle" (standardized), "seniority" (standardized), "country" (standardized).
      6. IMPORTANT: Return ONLY the JSON. No markdown formatting.

      Input:
      [${lines.join(',\n')}]
    `;

    const result = await model.generateContent(prompt);
    const text = result.response.text();

    // Clean potential markdown blocks
    const jsonStr = text.replace(/```json|```/g, '').trim();
    let data;
    try {
      data = JSON.parse(jsonStr);
    } catch (e) {
      // If direct array parse fails, try finding array bracket
      const match = text.match(/\[.*\]/s);
      if (match) {
        data = JSON.parse(match[0]);
      } else {
        throw new Error("Failed to parse Gemini response");
      }
    }

    if (!Array.isArray(data)) {
        throw new Error("Gemini response is not an array");
    }

    // Apply our local normalization functions to the Gemini results
    const normalized = data.map(item => {
      const result = { ...item };
      
      // Apply company normalization with special character removal
      if (result.organisation) {
        result.organisation = normalizeCompanyName(result.organisation);
      }
      
      // Apply job title normalization
      if (result.jobtitle) {
        result.jobtitle = canonicalJobTitle(result.jobtitle);
        result.personal = result.jobtitle; // Also set personal field
      }
      
      // Apply country normalization using countrycode.JSON
      if (result.country) {
        result.country = normalizeCountry(result.country);
      }
      
      return result;
    });

    res.json({ corrected: normalized });

  } catch (err) {
    console.error('/verify-data error:', err);
    res.status(500).json({ error: 'Verification failed', detail: err.message });
  }
});

// ========== NEW: Calendar & Google Meet Integration ==========

// Helper to create an OAuth2 client for Google using googleapis and persisted tokens for a username.
// Returns oauth2Client or throws error.
async function getOAuthClientForUser(username) {
  if (!google) throw new Error('googleapis module not available');
  const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
  const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
  const GOOGLE_REDIRECT_URI = process.env.GOOGLE_CALENDAR_REDIRECT || (process.env.GOOGLE_REDIRECT_URI || 'http://localhost:4000/auth/google/calendar/callback');

  if (!GOOGLE_CLIENT_ID || !GOOGLE_CLIENT_SECRET) {
    throw new Error('Google OAuth client not configured in environment.');
  }

  const oauth2Client = new google.auth.OAuth2(
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI
  );

  // Fetch stored refresh token
  try {
    const r = await pool.query('SELECT google_refresh_token FROM login WHERE username = $1', [username]);
    if (r.rows.length > 0 && r.rows[0].google_refresh_token) {
      oauth2Client.setCredentials({ refresh_token: r.rows[0].google_refresh_token });
    }
  } catch (e) {
    console.warn('[OAUTH] failed to load refresh token for user', username, e && e.message);
  }

  // Listen for new tokens and persist refresh token if provided (idempotent)
  oauth2Client.on && oauth2Client.on('tokens', async (tokens) => {
    if (tokens.refresh_token) {
      try {
        await pool.query('UPDATE login SET google_refresh_token = $1 WHERE username = $2', [tokens.refresh_token, username]);
      } catch (e) {
        console.warn('[OAUTH] failed to persist new refresh token', e && e.message);
      }
    }
    // Optionally persist access token expiry if you want
    if (tokens.expiry_date) {
      try {
        const dt = new Date(tokens.expiry_date);
        await pool.query('UPDATE login SET google_token_expires = $1 WHERE username = $2', [dt.toISOString(), username]);
      } catch (e) {
        // ignore
      }
    }
  });

  return oauth2Client;
}

// Route: start OAuth flow to connect Google Calendar for current logged in user
app.get('/auth/google/calendar/connect', requireLogin, async (req, res) => {
  if (!google) return res.status(500).send('Google APIs not available on server.');
  const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
  const GOOGLE_REDIRECT_URI = process.env.GOOGLE_CALENDAR_REDIRECT || (process.env.GOOGLE_REDIRECT_URI || 'http://localhost:4000/auth/google/calendar/callback');
  if (!GOOGLE_CLIENT_ID) return res.status(500).send('GOOGLE_CLIENT_ID not configured.');

  const oauth2Client = new google.auth.OAuth2(
    process.env.GOOGLE_CLIENT_ID,
    process.env.GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI
  );

  // Scopes for creating events and reading freebusy
  const scopes = [
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/calendar'
  ];

  const url = oauth2Client.generateAuthUrl({
    access_type: 'offline',
    scope: scopes,
    prompt: 'consent',
    state: req.user.username // carry username through callback
  });

  res.redirect(url);
});

// Callback: exchange code and persist refresh token to login table
app.get('/auth/google/calendar/callback', requireLogin, async (req, res) => {
  if (!google) return res.status(500).send('Google APIs not available on server.');
  const code = req.query.code;
  const state = req.query.state; // username passed back
  if (!code) return res.status(400).send('Missing code');

  try {
    const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
    const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
    const GOOGLE_REDIRECT_URI = process.env.GOOGLE_CALENDAR_REDIRECT || (process.env.GOOGLE_REDIRECT_URI || 'http://localhost:4000/auth/google/calendar/callback');

    const oauth2Client = new google.auth.OAuth2(
      GOOGLE_CLIENT_ID,
      GOOGLE_CLIENT_SECRET,
      GOOGLE_REDIRECT_URI
    );

    const { tokens } = await oauth2Client.getToken(code);
    oauth2Client.setCredentials(tokens);

    // Persist refresh_token if present; prefer req.user.username but fallback to state
    const username = req.user && req.user.username ? req.user.username : state;
    if (!username) {
      return res.status(400).send('Cannot determine username to persist OAuth tokens.');
    }

    if (tokens.refresh_token) {
      await pool.query('UPDATE login SET google_refresh_token = $1, google_token_expires = $2 WHERE username = $3', [tokens.refresh_token, tokens.expiry_date ? new Date(tokens.expiry_date).toISOString() : null, username]);
    } else {
      // If no refresh token was returned (possible if already granted and offline access not requested), we can still persist expiry info
      if (tokens.expiry_date) {
        await pool.query('UPDATE login SET google_token_expires = $1 WHERE username = $2', [new Date(tokens.expiry_date).toISOString(), username]);
      }
    }

    // Show a friendly success message (frontend typically navigates here in the popup)
    res.send(`<html><body><h3>Google Calendar connected for ${username}</h3><p>You can close this window and return to the app.</p><script>window.close()</script></body></html>`);
  } catch (err) {
    console.error('/auth/google/calendar/callback error', err);
    res.status(500).send('OAuth callback failed: ' + (err.message || 'unknown'));
  }
});

// Utility to build ICS content for event (METHOD:REQUEST recommended)
function buildICS({ uid, startISO, endISO, summary, description = '', organizerEmail, attendees = [], timezone = 'UTC', meetLink = '' }) {
  // Convert ISO date to ICS timestamp (UTC) format: YYYYMMDDTHHMMSSZ
  function toUTCStamp(dtISO) {
    const d = new Date(dtISO);
    if (isNaN(d.getTime())) return '';
    const yyyy = d.getUTCFullYear();
    const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(d.getUTCDate()).padStart(2, '0');
    const hh = String(d.getUTCHours()).padStart(2, '0');
    const mi = String(d.getUTCMinutes()).padStart(2, '0');
    const ss = String(d.getUTCSeconds()).padStart(2, '0');
    return `${yyyy}${mm}${dd}T${hh}${mi}${ss}Z`;
  }

  const dtstamp = toUTCStamp(new Date().toISOString());
  const dtstart = toUTCStamp(startISO);
  const dtend = toUTCStamp(endISO);
  const safeSummary = (summary || '').replace(/\r\n/g, '\\n').replace(/\n/g, '\\n');
  const safeDesc = (description || '').replace(/\r\n/g, '\\n').replace(/\n/g, '\\n');
  const organizer = organizerEmail ? `ORGANIZER;CN="Organizer":mailto:${organizerEmail}` : '';

  const lines = [
    'BEGIN:VCALENDAR',
    'PRODID:-//CandidateManagement//EN',
    'VERSION:2.0',
    'CALSCALE:GREGORIAN',
    'METHOD:REQUEST',
    'BEGIN:VEVENT',
    `UID:${uid}`,
    `DTSTAMP:${dtstamp}`,
    dtstart ? `DTSTART:${dtstart}` : '',
    dtend ? `DTEND:${dtend}` : '',
    `SUMMARY:${safeSummary}`,
    `DESCRIPTION:${safeDesc}`,
    meetLink ? `LOCATION:${meetLink}` : '',
    organizer
  ];

  for (const a of attendees || []) {
    const mail = String(a).trim();
    if (!mail) continue;
    // simple attendee line; no CN available
    lines.push(`ATTENDEE;ROLE=REQ-PARTICIPANT;RSVP=TRUE:mailto:${mail}`);
  }

  // Add Google Meet link as an X- property to help Gmail clients
  if (meetLink) {
    lines.push(`X-ALT-DESC;FMTTYPE=text/html:Join via Google Meet: <a href="${meetLink}">${meetLink}</a>`);
  }

  lines.push('END:VEVENT', 'END:VCALENDAR');
  return lines.filter(Boolean).join('\r\n');
}

// Helper: compute simple free slots between timeMin/timeMax avoiding busy intervals
function computeFreeSlots(busyIntervals = [], timeMinISO, timeMaxISO, durationMinutes = 30, businessHours = { startHour: 9, endHour: 17, timezone: 'UTC' }, maxResults = 6) {
  const start = new Date(timeMinISO).getTime();
  const end = new Date(timeMaxISO).getTime();
  if (isNaN(start) || isNaN(end) || start >= end) return [];

  // Convert busy intervals to numeric ranges
  const busyRanges = (busyIntervals || []).map(b => {
    const s = new Date(b.start).getTime();
    const e = new Date(b.end).getTime();
    if (isNaN(s) || isNaN(e)) return null;
    return { start: s, end: e };
  }).filter(Boolean);

  // Merge busy ranges
  busyRanges.sort((a, b) => a.start - b.start);
  const merged = [];
  busyRanges.forEach(r => {
    if (!merged.length) merged.push({ ...r });
    else {
      const last = merged[merged.length - 1];
      if (r.start <= last.end) {
        last.end = Math.max(last.end, r.end);
      } else merged.push({ ...r });
    }
  });

  const durationMs = durationMinutes * 60 * 1000;
  const slots = [];
  // scan from start to end in step of durationMinutes (but aligned to round minutes)
  let cursor = start;
  // Align cursor to next 15-minute boundary for nicer slots
  const d = new Date(cursor);
  const minutes = d.getUTCMinutes();
  const aligned = Math.ceil(minutes / 15) * 15;
  d.setUTCMinutes(aligned);
  d.setUTCSeconds(0);
  d.setUTCMilliseconds(0);
  cursor = d.getTime();

  while (cursor + durationMs <= end && slots.length < maxResults) {
    const slotStart = cursor;
    const slotEnd = cursor + durationMs;

    // Respect business hours in UTC: check startHour/endHour
    const sDate = new Date(slotStart);
    const hourUTC = sDate.getUTCHours();
    if (hourUTC < businessHours.startHour || hourUTC >= businessHours.endHour) {
      cursor += 15 * 60 * 1000; // advance by 15 minutes
      continue;
    }

    // Check overlap with merged busy ranges
    let overlap = false;
    for (const br of merged) {
      if (!(slotEnd <= br.start || slotStart >= br.end)) {
        overlap = true;
        break;
      }
    }
    if (!overlap) {
      slots.push({ start: new Date(slotStart).toISOString(), end: new Date(slotEnd).toISOString() });
    }
    cursor += 15 * 60 * 1000;
  }

  return slots;
}

// Endpoint: query freebusy and return candidate slots (POST body: { startISO, endISO, durationMinutes })
app.post('/calendar/freebusy', requireLogin, async (req, res) => {
  try {
    if (!google) return res.status(500).json({ error: 'Google APIs module not available.' });

    const { startISO, endISO, durationMinutes = 30, attendees = [] } = req.body;
    if (!startISO || !endISO) return res.status(400).json({ error: 'startISO and endISO required.' });

    const oauth2Client = await getOAuthClientForUser(req.user.username);
    const calendar = google.calendar({ version: 'v3', auth: oauth2Client });

    const fbReq = {
      resource: {
        timeMin: startISO,
        timeMax: endISO,
        items: [{ id: 'primary' }]
      }
    };

    // If attendees emails provided and you have permission to check their calendars, include them
    const attendeeItems = (attendees || []).map(email => ({ id: email }));
    if (attendeeItems.length) fbReq.resource.items.push(...attendeeItems);

    const fb = await calendar.freebusy.query(fbReq);
    const primaryBusy = (fb.data && fb.data.calendars && fb.data.calendars.primary && fb.data.calendars.primary.busy) ? fb.data.calendars.primary.busy : [];
    const slots = computeFreeSlots(primaryBusy, startISO, endISO, durationMinutes, { startHour: 9, endHour: 17 }, 8);

    res.json({ ok: true, slots });
  } catch (err) {
    console.error('/calendar/freebusy error', err);
    res.status(500).json({ error: err.message || 'freebusy failed' });
  }
});

// Endpoint: create calendar event with conferenceData (Meet) and return meet link and ICS representation
// Body: { summary, description, startISO, endISO, attendees: ['a@b.com'], timezone, sendUpdates: 'none'|'all' }
app.post('/calendar/create-event', requireLogin, async (req, res) => {
  try {
    if (!google) return res.status(500).json({ error: 'Google APIs module not available.' });

    const { summary, description = '', startISO, endISO, attendees = [], timezone = 'UTC', sendUpdates = 'none' } = req.body;
    if (!startISO || !endISO || !summary) return res.status(400).json({ error: 'summary, startISO and endISO are required.' });

    const oauth2Client = await getOAuthClientForUser(req.user.username);
    const calendar = google.calendar({ version: 'v3', auth: oauth2Client });

    // Build event
    const event = {
      summary,
      description,
      start: { dateTime: startISO, timeZone: timezone },
      end: { dateTime: endISO, timeZone: timezone },
      attendees: (attendees || []).filter(Boolean).map(email => ({ email })),
      conferenceData: {
        createRequest: {
          requestId: `meet-${Date.now()}-${Math.random().toString(36).slice(2,8)}`,
          conferenceSolutionKey: { type: 'hangoutsMeet' }
        }
      }
    };

    // Insert event
    const resp = await calendar.events.insert({
      calendarId: 'primary',
      conferenceDataVersion: 1,
      sendUpdates: sendUpdates, // 'none' to not let Google send invites, or 'all' to let Google send invites
      resource: event
    });

    const created = resp.data;
    // Extract Meet link (entryPoints)
    let meetLink = null;
    try {
      const entryPoints = created.conferenceData && created.conferenceData.entryPoints ? created.conferenceData.entryPoints : [];
      for (const ep of entryPoints) {
        if (ep.entryPointType === 'video') {
          meetLink = ep.uri;
          break;
        }
      }
    } catch (e) {
      // ignore
    }

    // Build ICS to attach if frontend wants to send via SMTP instead of letting Google send invites.
    // Use organizer as the authenticated user email if available
    let organizerEmail = null;
    try {
      const o = await oauth2Client.getTokenInfo && oauth2Client.getTokenInfo(oauth2Client.credentials.access_token).catch(()=>null);
      if (o && o.email) organizerEmail = o.email;
    } catch (e) { /* ignore */ }

    const uid = created.id || `ev-${Date.now()}-${Math.random().toString(36).slice(2,6)}`;
    const ics = buildICS({
      uid,
      startISO,
      endISO,
      summary,
      description,
      organizerEmail: organizerEmail || req.user.username || 'organizer@example.com',
      attendees: attendees || [],
      timezone,
      meetLink: meetLink || ''
    });

    res.json({ ok: true, event: created, meetLink, ics });
  } catch (err) {
    console.error('/calendar/create-event error', err);
    res.status(500).json({ error: err.message || 'create-event failed' });
  }
});

// ========== END Calendar & Meet Integration ==========


// ========== EMAIL VERIFICATION LOGIC ==========

// Helper: REAL SMTP Handshake
async function smtpVerify(email, mxHost) {
  if (!email || !mxHost) return 'unknown';
  const domain = email.split('@')[1];
  
  return new Promise((resolve, reject) => {
    const socket = net.createConnection(25, mxHost);
    let step = 0;
    
    // Timeout 6s
    socket.setTimeout(6000);
    
    socket.on('connect', () => { /* connected */ });
    socket.on('timeout', () => {
       socket.destroy();
       resolve('timeout');
    });
    socket.on('error', (err) => {
       socket.destroy();
       resolve('connection_error');
    });

    socket.on('data', (data) => {
      const msg = data.toString();
      // 0. Initial greeting 220
      if (step === 0 && msg.startsWith('220')) {
         socket.write(`EHLO ${domain}\r\n`);
         step = 1;
      }
      // 1. EHLO response 250
      else if (step === 1 && msg.startsWith('250')) {
         socket.write(`MAIL FROM:<check@${domain}\r\n`);
         step = 2;
      }
      // 2. MAIL FROM response 250
      else if (step === 2 && msg.startsWith('250')) {
         socket.write(`RCPT TO:<${email}>\r\n`);
         step = 3;
      }
      // 3. RCPT TO response
      else if (step === 3) {
         if (msg.startsWith('250') || msg.startsWith('251')) {
           resolve('valid');
         } else if (msg.startsWith('550')) {
           resolve('invalid');
         } else {
           resolve('unknown_response');
         }
         socket.end();
      }
    });
  });
}

// ========== NEW ENDPOINT: Generate Emails via Gemini (Ranked, No Verification yet) ==========
app.post('/generate-email', requireLogin, async (req, res) => {
  try {
    const { name, company, country } = req.body;
    if (!name || !company) {
      return res.status(400).json({ error: 'Name and Company are required.' });
    }

    if (!GoogleGenerativeAIClass) {
      return res.status(500).json({ error: "Gemini SDK not installed." });
    }
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) return res.status(500).json({ error: 'API key missing.' });

    const genAI = new GoogleGenerativeAIClass(apiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });

    // Request strictly 3 ranked emails
    const genPrompt = `
      Generate a list of exactly 3 most likely business email address permutations for a person named "${name}" working at the company "${company}"${country ? ` (located in ${country})` : ''}.
      Infer the likely domain name based on the company.
      Sort the list strictly by highest probability of being the correct active email to lowest probability.
      Return strictly a JSON object: { "emails": ["email1", "email2", "email3"] }
      Do not include markdown formatting.
    `;

    const genResult = await model.generateContent(genPrompt);
    const genText = genResult.response.text();
    
    // Clean markdown if present
    const jsonStr = genText.replace(/```json|```/g, '').trim();
    let data;
    try {
      data = JSON.parse(jsonStr);
    } catch(e) {
       const match = genText.match(/\[.*\]/s);
       if (match) data = { emails: JSON.parse(match[0]) };
       else throw new Error("Failed to parse Gemini generation response");
    }
    
    const candidates = data.emails || [];
    
    // RETURN IMMEDIATELY, NO VERIFICATION
    res.json({ emails: candidates });

  } catch (err) {
    console.error('/generate-email error:', err);
    res.status(500).json({ error: 'Generation failed' });
  }
});

// ========== NEW ENDPOINT: Verify Email Details via Gemini + SMTP PING ==========
app.post('/verify-email-details', requireLogin, async (req, res) => {
  try {
    const { email } = req.body;
    if (!email) {
      return res.status(400).json({ error: 'Email is required.' });
    }

    if (!GoogleGenerativeAIClass) {
      return res.status(500).json({ error: "Gemini SDK not installed." });
    }
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) return res.status(500).json({ error: 'API key missing.' });

    const genAI = new GoogleGenerativeAIClass(apiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });

    // 1. Perform Technical Checks First (MX + SMTP)
    const domain = email.split('@')[1];
    let mxRecords = [];
    let mxHost = null;
    let smtpStatus = 'unknown'; // valid, invalid, timeout, etc.

    try {
      mxRecords = await dns.resolveMx(domain);
      if (mxRecords && mxRecords.length > 0) {
        // sort by priority
        mxRecords.sort((a,b) => a.priority - b.priority);
        mxHost = mxRecords[0].exchange;
        
        // Real SMTP Handshake
        smtpStatus = await smtpVerify(email, mxHost);
      } else {
        smtpStatus = 'no_mx';
      }
    } catch (e) {
      smtpStatus = 'dns_error';
    }

    // 2. Ask Gemini to enhance metadata AND interpret result based on Enterprise logic
    // We pass the SMTP result to Gemini so it knows the technical reality
    const prompt = `
      Analyze this email address: "${email}".
      
      Technical Check Result:
      - MX Record: ${mxHost || 'None'}
      - SMTP Handshake Response: ${smtpStatus}

      Act as a strict email verification engine. 
      You must combine the technical check result with enterprise logic.

      Rules for Verification:
      1. STATUS: "Capture All" (Mapped from 'valid')
         - Use this status if SMTP Handshake was "valid" (250 OK).
         - OR if SMTP Handshake was "timeout/unknown" BUT the domain is known to be an Enterprise Gateway (Proofpoint/Mimecast/Google) AND you are highly confident the format is correct.
      2. STATUS: "invalid"
         - Use this if SMTP Handshake was "invalid" (550 User unknown).
         - OR if DNS/MX failed.
      3. STATUS: "catch-all"
         - Use this if the server accepts all emails (wildcard) but you cannot definitively confirm existence.

      Required Fields (Return strictly JSON):
      - status (String: "Capture All", "catch-all", or "invalid")
      - sub_status (String: "None" or failure detail)
      - free_email (String: "Yes" or "No")
      - account (String: part before @)
      - smtp_provider (String: inferred from MX e.g. "proofpoint", "google")
      - first_name (String: inferred)
      - last_name (String: inferred)
      - domain (String)
      - mx_found (String: "Yes" or "No")
      - mx_record (String)
      - domain_age_days (Integer: estimate)
      - did_you_mean (String)

      Example of Success:
      {
        "status": "Capture All",
        "sub_status": "None",
        "free_email": "No",
        "account": "john.doe",
        "smtp_provider": "proofpoint",
        "first_name": "John",
        "last_name": "Doe",
        "domain": "company.com",
        "mx_found": "Yes",
        "mx_record": "mxa-001.proofpoint.com",
        "domain_age_days": 4500,
        "did_you_mean": "Unknown"
      }
    `;

    const result = await model.generateContent(prompt);
    const text = result.response.text();

    const jsonStr = text.replace(/```json|```/g, '').trim();
    let data;
    try {
      data = JSON.parse(jsonStr);
    } catch (e) {
       const match = text.match(/\{[\s\S]*\}/);
       if(match) data = JSON.parse(match[0]);
       else throw new Error("Failed to parse Gemini response");
    }

    res.json(data);
  } catch (err) {
    console.error('/verify-email-details error:', err);
    res.status(500).json({ error: 'Verification failed' });
  }
});

// ========== NEW: Draft Email Endpoint (AI) ==========
app.post('/draft-email', requireLogin, async (req, res) => {
    try {
        const { prompt: userPrompt, context } = req.body;
        const candidateName = context?.candidateName || 'Candidate';
        const myEmail = context?.myEmail || 'Me';

        if (!GoogleGenerativeAIClass) {
            return res.status(500).json({ error: "Gemini SDK not installed." });
        }
        const apiKey = process.env.GOOGLE_API_KEY;
        if (!apiKey) return res.status(500).json({ error: 'API key missing.' });

        const genAI = new GoogleGenerativeAIClass(apiKey);
        const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });

        const instruction = `
            Act as a professional recruiter. Write a draft email based on this request: "${userPrompt}".
            Context:
            - Recipient Name: ${candidateName}
            - Sender Email/Name: ${myEmail}
            
            Return strictly a JSON object with two fields:
            {
                "subject": "Email Subject Line",
                "body": "Email Body Text (plain text, use \\n for new lines)"
            }
            Do not wrap in markdown code blocks.
        `;

        const result = await model.generateContent(instruction);
        const text = result.response.text();
        const jsonStr = text.replace(/```json|```/g, '').trim();
        let data;
        try {
            data = JSON.parse(jsonStr);
        } catch (e) {
             // fallback parsing if model output isn't perfect JSON
             const match = text.match(/\{[\s\S]*\}/);
             if(match) data = JSON.parse(match[0]);
             else throw new Error("Failed to parse AI draft response");
        }
        res.json(data);
    } catch (err) {
        console.error('/draft-email error:', err);
        res.status(500).json({ error: 'Drafting failed' });
    }
});

// ========== NEW: Send Email Endpoint (Nodemailer) ==========
app.post('/send-email', requireLogin, async (req, res) => {
    const { to, cc, bcc, subject, body, from, smtpConfig, ics } = req.body;

    let transporterConfig;

    if (smtpConfig && smtpConfig.user && smtpConfig.pass) {
        // Use provided config
        transporterConfig = {
            host: smtpConfig.host || 'smtp.gmail.com',
            port: parseInt(smtpConfig.port || '587'),
            secure: smtpConfig.secure === true || smtpConfig.secure === 'true', // Handle string/bool
            auth: {
                user: smtpConfig.user,
                pass: smtpConfig.pass,
            },
        };
    } else {
        // Fallback to Env
        if (!process.env.SMTP_USER || !process.env.SMTP_PASS) {
            return res.status(500).json({ error: "Server configuration error: SMTP_USER or SMTP_PASS is missing in environment variables, and no custom config provided." });
        }
        transporterConfig = {
            host: process.env.SMTP_HOST || 'smtp.gmail.com',
            port: parseInt(process.env.SMTP_PORT || '587'),
            secure: process.env.SMTP_SECURE === 'true',
            auth: {
                user: process.env.SMTP_USER,
                pass: process.env.SMTP_PASS,
            },
        };
    }

    try {
        // Create transporter with environment variables
        const transporter = nodemailer.createTransport(transporterConfig);

        const mailOptions = {
            from: from || transporterConfig.auth.user, // Prefer user input > smtp user
            to,
            cc,
            bcc,
            subject,
            text: body, // plain text body
            html: body ? body.replace(/\n/g, '<br/>') : '' // simple html conversion
        };

        // If ICS string provided, attach it as a calendar alternative to improve compatibility across clients.
        if (ics && typeof ics === 'string') {
          // Attach as an alternative content type for invites
          mailOptions.alternatives = mailOptions.alternatives || [];
          mailOptions.alternatives.push({
            contentType: 'text/calendar; charset="utf-8"; method=REQUEST',
            content: ics
          });
          // Also include as a downloadable attachment in some clients
          mailOptions.attachments = mailOptions.attachments || [];
          mailOptions.attachments.push({
            filename: 'invite.ics',
            content: ics,
            contentType: 'text/calendar'
          });
        }

        const info = await transporter.sendMail(mailOptions);
        console.log('Message sent: %s', info.messageId);
        res.json({ message: 'Email sent successfully', messageId: info.messageId });

    } catch (error) {
        console.error('Send email error:', error);
        // Return the error message to the client (which shows up in the alert)
        res.status(500).json({ error: "Failed to send email: " + error.message });
    }
});

// ========================= NEW: DASHBOARD API ENDPOINTS =========================

// Config: Fields allowed for filtering/aggregation
const ALLOWED_FIELDS = {
    country: "country",
    company: "company", 
    jobtitle: "jobtitle",
    sector: "sector",
    jobfamily: "jobfamily",
    geographic: "geographic",
    seniority: "seniority",
    skillset: "skillset", 
    sourcingstatus: "sourcingstatus",
    role_tag: "role_tag",
    product: "product",
    rating: "rating",
    pic: "pic",
    education: "education",
    comment: "comment",
    id: "id", // for simple count
    name: "name",
    linkedinurl: "linkedinurl"
};

/**
 * Helper to build WHERE clause from filters object
 * filters: { country: 'USA', seniority: 'Senior' }
 */
function buildWhereClause(filters, paramStartIdx = 1) {
    const conditions = [];
    const values = [];
    let idx = paramStartIdx;

    if (!filters) return { where: '', values, nextIdx: idx };

    for (const [key, val] of Object.entries(filters)) {
        if (ALLOWED_FIELDS[key] && val) {
            // Handle comma-separated values in filter as OR (simple implementation)
            // Or exact match. Let's do partial match or exact based on field type?
            // Dashboard filters usually imply equality or containment.
            // Using ILIKE for flexibility
            conditions.push(`"${ALLOWED_FIELDS[key]}" ILIKE $${idx}`);
            values.push(`%${val}%`); 
            idx++;
        }
    }

    const where = conditions.length ? 'WHERE ' + conditions.join(' AND ') : '';
    return { where, values, nextIdx: idx };
}

/**
 * POST /api/dashboard/query
 * General purpose endpoint for dashboard charts.
 * Body: { dimension: 'country', measure: 'count', filters: {...} }
 */
app.post('/api/dashboard/query', requireLogin, async (req, res) => {
    try {
        const { dimension, measure, filters } = req.body;
        
        if (!dimension || !ALLOWED_FIELDS[dimension]) {
            return res.status(400).json({ ok: false, error: 'Invalid or missing dimension' });
        }

        const col = ALLOWED_FIELDS[dimension];
        const { where, values } = buildWhereClause(filters);

        // Special handling for 'skillset' or multi-value fields if stored as comma-separated strings
        // For simplicity, we assume standard GROUP BY. 
        // If skillset is comma-separated, proper normalization requires unnesting which depends on DB structure.
        // Assuming simple string column for now as per schema.

        let sql = '';
        
        if (dimension === 'skillset') {
             // Attempt to unnest if it's a string with commas
             // PostgreSQL: unnest(string_to_array(skillset, ','))
             // We need to clean whitespace too.
             sql = `
                SELECT TRIM(s.token) as label, COUNT(*) as value
                FROM "process", unnest(string_to_array(skillset, ',')) as s(token)
                ${where}
                GROUP BY 1
                ORDER BY value DESC
                LIMIT 20
             `;
        } else {
             // Standard Group By
             sql = `
                SELECT "${col}" as label, COUNT(*) as value
                FROM "process"
                ${where}
                GROUP BY 1
                ORDER BY value DESC
                LIMIT 20
             `;
        }
        
        // If measuring ID count (KPI total)
        if (dimension === 'id') {
             sql = `SELECT COUNT(*) as total_rows FROM "process" ${where}`;
             const r = await pool.query(sql, values);
             return res.json({ ok: true, total_rows: parseInt(r.rows[0].total_rows) });
        }

        const result = await pool.query(sql, values);
        
        const labels = [];
        const data = [];
        
        result.rows.forEach(r => {
            if (r.label) {
                labels.push(r.label);
                data.push(parseInt(r.value));
            }
        });

        res.json({ ok: true, labels, data });

    } catch (e) {
        console.error('/api/dashboard/query error', e);
        res.status(500).json({ ok: false, error: e.message });
    }
});

/**
 * GET /api/dashboard/filter-options
 * Get distinct values for a filter dropdown
 * Query: ?field=country
 */
app.get('/api/dashboard/filter-options', requireLogin, async (req, res) => {
    try {
        const field = req.query.field;
        if (!field || !ALLOWED_FIELDS[field]) {
             return res.status(400).json({ ok: false, error: 'Invalid field' });
        }
        
        const col = ALLOWED_FIELDS[field];
        let sql = '';

        if (field === 'skillset') {
             sql = `
                SELECT DISTINCT TRIM(s.token) as val
                FROM "process", unnest(string_to_array(skillset, ',')) as s(token)
                ORDER BY 1 ASC
                LIMIT 100
             `;
        } else {
             sql = `SELECT DISTINCT "${col}" as val FROM "process" ORDER BY 1 ASC LIMIT 100`;
        }

        const result = await pool.query(sql);
        const options = result.rows.map(r => r.val).filter(Boolean);
        
        res.json({ ok: true, options });

    } catch (e) {
        console.error('/api/dashboard/filter-options error', e);
        res.status(500).json({ ok: false, error: e.message });
    }
});

/**
 * ========== NEW: Save Report Template Selection ==========
 */
app.post('/save-report-template', requireLogin, (req, res) => {
    try {
        const { reportId, dsAlias } = req.body;
        const username = req.user.username;
        if (!reportId) return res.status(400).json({ error: 'Report ID required' });
        
        // Validate dsAlias if provided: must be like "ds0", "ds1", ...
        let alias = null;
        if (typeof dsAlias !== 'undefined' && dsAlias !== null) {
            if (!/^ds\d+$/.test(String(dsAlias).trim())) {
                return res.status(400).json({ error: 'Invalid dsAlias. Expected format "ds0", "ds1", ...' });
            }
            alias = String(dsAlias).trim();
        }

        const filename = `template_${username}.json`;
        const filepath = path.resolve(__dirname, 'template', filename);
        
        const data = {
            username: username,
            reportId: reportId,
            dsAlias: alias,
            updatedAt: new Date().toISOString()
        };
        
        // Ensure template directory exists
        try { fs.mkdirSync(path.resolve(__dirname, 'template'), { recursive: true }); } catch (e) {}
        
        fs.writeFileSync(filepath, JSON.stringify(data, null, 2));
        
        res.json({ ok: true, message: 'Template saved', file: filename, dsAlias: alias });
    } catch (e) {
        console.error('Error saving template:', e);
        res.status(500).json({ error: 'Failed to save template' });
    }
});

// ========== PORT TO GOOGLE SHEETS / LOOKER STUDIO ==========

// 1. Initial Route: Redirects to Google Login
app.get('/port-to-looker', requireLogin, (req, res) => {
  if (!google) {
    return res.status(500).send("Google APIs not configured (module missing).");
  }
  const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
  const GOOGLE_REDIRECT_URI = process.env.GOOGLE_REDIRECT_URI || 'http://localhost:4000/auth/google/callback';
  
  if (!GOOGLE_CLIENT_ID) {
    return res.status(500).send("Google Client ID not configured in environment.");
  }

  // Scopes needed: Sheets (read/write), Drive (file creation/copying)
  // UPDATED: Added full drive access to fix 403 insufficient scope error on drive.files.copy
  const scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive' // Full drive access needed to copy arbitrary templates
  ];

  const oauth2Client = new google.auth.OAuth2(
    GOOGLE_CLIENT_ID,
    process.env.GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI
  );

  const url = oauth2Client.generateAuthUrl({
    access_type: 'offline', // ensures we get a refresh token if needed, though simple flow works without
    scope: scopes,
    prompt: 'consent', // Force consent screen to ensure new scopes are granted
    state: req.user.username // pass username to callback for tracking context
  });

  res.redirect(url);
});

// 2. Callback Route: Handles Auth Code -> CSV Export -> Sheet Creation -> Template Copy
app.get('/auth/google/callback', requireLogin, async (req, res) => {
  if (!google) return res.status(500).send("Google module missing.");
  
  const code = req.query.code;
  if (!code) return res.status(400).send("Authorization code missing.");

  try {
    const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
    const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
    const GOOGLE_REDIRECT_URI = process.env.GOOGLE_REDIRECT_URI || 'http://localhost:4000/auth/google/callback';
    
    // Check for user-specific template file first
    let LOOKER_TEMPLATE_ID = process.env.LOOKER_TEMPLATE_ID;
    let LOOKER_TEMPLATE_ALIAS = null;
    try {
        const templateFile = path.resolve(__dirname, 'template', `template_${req.user.username}.json`);
        if (fs.existsSync(templateFile)) {
            const tmplData = JSON.parse(fs.readFileSync(templateFile, 'utf8'));
            if (tmplData.reportId) {
                LOOKER_TEMPLATE_ID = tmplData.reportId;
                console.log(`[LOOKER] Using user-selected template: ${LOOKER_TEMPLATE_ID}`);
                if (tmplData.dsAlias && /^ds\d+$/.test(String(tmplData.dsAlias).trim())) {
                  LOOKER_TEMPLATE_ALIAS = String(tmplData.dsAlias).trim();
                  console.log(`[LOOKER] Using saved ds alias for this template: ${LOOKER_TEMPLATE_ALIAS}`);
              }
            }
        }
    } catch (e) {
        console.warn('Error reading user template file, falling back to ENV', e.message);
    }

    const oauth2Client = new google.auth.OAuth2(
      GOOGLE_CLIENT_ID,
      GOOGLE_CLIENT_SECRET,
      GOOGLE_REDIRECT_URI
    );

    const { tokens } = await oauth2Client.getToken(code);
    oauth2Client.setCredentials(tokens);

    // A. Export Data from Postgres
    // Explicitly select columns to exclude 'cv' (binary/base64 data)
    const colsToExport = [
      'id', 'name', 'jobtitle', 'company', 'sector', 'jobfamily', 'role_tag',
      'skillset', 'geographic', 'country', 'email', 'mobile', 'office',
      'personal', 'seniority', 'sourcingstatus', 'product', 'userid', 'username',
      'linkedinurl', 'jskillset', 'lskillset', 'rating'
    ];
    
    // Construct SELECT query with explicit columns
    // UPDATED: Filter export to only rows owned by the current user
    const sqlExport = `SELECT ${colsToExport.map(c => `"${c}"`).join(', ')} FROM "process" WHERE userid = $1`;
    const result = await pool.query(sqlExport, [String(req.user?.id || '')]);
    const rows = result.rows;
    
    if (rows.length === 0) {
      return res.send("No data in database to export.");
    }

    // Determine Headers (use the explicit list)
    const headers = colsToExport;
    
    // Format as 2D array for Sheets API
    const values = [headers];
    rows.forEach(r => {
      const rowVals = headers.map(h => {
        const val = r[h];
        if (val === null || val === undefined) return '';
        // Basic sanitization for Sheets
        return String(val); 
      });
      values.push(rowVals);
    });

    // B. Create Google Sheet
    const sheets = google.sheets({ version: 'v4', auth: oauth2Client });
    const drive = google.drive({ version: 'v3', auth: oauth2Client });
    const dateStr = new Date().toISOString().slice(0,10);
    
    // Create new sheet
    const createRes = await sheets.spreadsheets.create({
      resource: {
        properties: {
          title: `Talent Intel Data - ${dateStr}`
        }
      }
    });
    const spreadsheetId = createRes.data.spreadsheetId;
    const spreadsheetUrl = createRes.data.spreadsheetUrl;

    // Populate Sheet
    await sheets.spreadsheets.values.update({
      spreadsheetId,
      range: 'Sheet1!A1',
      valueInputOption: 'RAW',
      resource: { values }
    });

    // C. Copy Looker Studio Template (If configured)
    let lookerUrl = "https://lookerstudio.google.com/"; // Default fallback

    // Normalize LOOKER_TEMPLATE_ID (accept URL or plain id)
    if (LOOKER_TEMPLATE_ID && LOOKER_TEMPLATE_ID.includes('http')) {
      const m = LOOKER_TEMPLATE_ID.match(/[-_A-Za-z0-9]{20,}/);
      if (m) LOOKER_TEMPLATE_ID = m[0];
    }

    // === NEW CHECK === 
    // If the ID looks like a Looker Studio UUID (contains hyphens), we cannot copy it via Drive API.
    // Instead, use the create URL to instantiate a report and inject the sheet ID.
    if (LOOKER_TEMPLATE_ID && /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(LOOKER_TEMPLATE_ID)) {
        console.log('[LOOKER] Detected Looker Studio reportId. Building create URL to inject the new Sheet as data source.');

        const encodedReportId = encodeURIComponent(LOOKER_TEMPLATE_ID);
        const encodedSheetId = encodeURIComponent(spreadsheetId || '');
        const encodedWorksheetId = encodeURIComponent('0');

        // Check if we have a specific valid alias from user configuration
        const aliasValidated = (LOOKER_TEMPLATE_ALIAS && /^ds\d+$/.test(LOOKER_TEMPLATE_ALIAS)) ? LOOKER_TEMPLATE_ALIAS : null;

        if (aliasValidated) {
            // Use specific alias
            lookerUrl = `https://lookerstudio.google.com/reporting/create?c.reportId=${encodedReportId}` +
                        `&${aliasValidated}.connector=googleSheets` +
                        `&${aliasValidated}.spreadsheetId=${encodedSheetId}` +
                        `&${aliasValidated}.worksheetId=${encodedWorksheetId}`;
            console.log('[LOOKER] create URL (using user alias):', lookerUrl);
        } else {
            // Best-effort: include several ds aliases (ds0..ds3) to catch common default aliases
            // This ensures the sheet is bound instantly even if the user didn't specify the alias manually
            const aliases = ['ds0','ds1','ds2','ds3'];
            const params = [`c.reportId=${encodedReportId}`];
            aliases.forEach(a => {
                params.push(`${a}.connector=googleSheets`);
                params.push(`${a}.spreadsheetId=${encodedSheetId}`);
                params.push(`${a}.worksheetId=${encodedWorksheetId}`);
            });
            lookerUrl = `https://lookerstudio.google.com/reporting/create?${params.join('&')}`;
            console.log('[LOOKER] create URL (best-effort multiple aliases):', lookerUrl.slice(0, 1000));
        }
    
    } else if (LOOKER_TEMPLATE_ID) {
      // Otherwise, assume it is a Drive File ID and try to copy
      try {
        // 1) Try to GET file metadata to determine visibility/permission results
        const fileMeta = await drive.files.get({ fileId: LOOKER_TEMPLATE_ID, fields: 'id,name,owners' });
        console.log('[LOOKER] template visible:', fileMeta.data);

        // 2) Now attempt the copy
        const copyRes = await drive.files.copy({
          fileId: LOOKER_TEMPLATE_ID,
          resource: {
            name: `My Talent Dashboard - ${dateStr}`
          }
        });
        
        console.log('[LOOKER] copy success:', copyRes.data);
        const fileInfo = await drive.files.get({
            fileId: copyRes.data.id,
            fields: 'webViewLink'
        });
        lookerUrl = fileInfo.data.webViewLink;
      } catch (err) {
        console.warn("Failed to copy template (maybe permissions?):", err.response?.data || err.message || err);
      }
    } else {
        console.log('[LOOKER] LOOKER_TEMPLATE_ID not configured; skipping template copy.');
    }

    // D. Success Response (Redirect to simple page or dashboard with params)
    // We'll return a simple HTML page that displays the links and closes itself or lets user navigate
    res.send(`
      <html>
        <body style="font-family: sans-serif; text-align: center; padding: 50px;">
          <h1 style="color: #1a73e8;">Success!</h1>
          <p>Your data has been ported to your Google Drive.</p>
          <div style="margin: 20px 0;">
            <a href="${spreadsheetUrl}" target="_blank" style="display:inline-block; padding: 10px 20px; background: #188038; color: white; text-decoration: none; border-radius: 5px; margin: 5px;">
              Open Google Sheet (Data)
            </a>
            <a href="${lookerUrl}" target="_blank" style="display:inline-block; padding: 10px 20px; background: #4285f4; color: white; text-decoration: none; border-radius: 5px; margin: 5px;">
              Open Looker Studio Report
            </a>
          </div>
          <p style="color: #555; font-size: 14px;">
            <strong>Next Step:</strong> Open the Looker Studio report, click "Edit", select the data source, and "Reconnect" it to your new "Talent Intel Data" sheet.
          </p>
          <button onclick="window.close()" style="margin-top:20px;">Close Window</button>
        </body>
      </html>
    `);

  } catch (error) {
    console.error("Port to Looker Error:", error);
    res.status(500).send(`Export failed: ${error.message}`);
  }
});

// ========================= END DASHBOARD API =========================

// SSE Connection Management
const sseConnections = new Set();

function broadcastSSE(event, data) {
  const message = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  sseConnections.forEach(client => {
    try {
      client.write(message);
    } catch (e) {
      // Log error with full context for debugging
      console.warn(`[SSE] Error broadcasting event '${event}' to client:`, e);
      sseConnections.delete(client);
    }
  });
}

// SSE Endpoint for real-time updates
app.get('/api/events', (req, res) => {
  // Set headers for SSE
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  // Use the same CORS origins as the rest of the app
  const origin = req.headers.origin;
  if (allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Access-Control-Allow-Credentials', 'true');
  }
  res.flushHeaders();

  // Add this connection to the set
  sseConnections.add(res);
  console.log('[SSE] client connected, total:', sseConnections.size);

  // Send initial connection confirmation
  res.write(`event: connected\ndata: ${JSON.stringify({ message: 'Connected to SSE' })}\n\n`);

  // Clean up on client disconnect
  req.on('close', () => {
    sseConnections.delete(res);
    console.log('[SSE] client disconnected, total:', sseConnections.size);
  });
});

// Create HTTP server
const server = http.createServer(app);

// START SERVER
server.listen(port, () => {
  console.log(`Backend running on port ${port}`);
});