"""
Sector and Product/Domain Mapping Configuration
This file contains mappings used for sector inference and validation in the job description analysis system.
"""

# Product/domain keyword mapping for second sector validation
# Maps product keywords to valid domain keywords they should match
PRODUCT_TO_DOMAIN_KEYWORDS = {
    "mobile phone": ["consumer electronics", "electronics"],
    "smartphone": ["consumer electronics", "electronics"],
    "phone": ["consumer electronics", "electronics"],
    "cloud": ["cloud", "infrastructure"],
    "devops": ["cloud", "infrastructure"],
    "kubernetes": ["cloud", "infrastructure"],
    "aws": ["cloud", "infrastructure"],
    "azure": ["cloud", "infrastructure"],
    "gcp": ["cloud", "infrastructure"],
    "ai": ["ai", "data", "artificial intelligence"],
    "machine learning": ["ai", "data"],
    "data science": ["ai", "data"],
    "gaming": ["gaming", "entertainment"],
    "game": ["gaming", "entertainment"],
    "fintech": ["fintech", "financial"],
    "bank": ["banking", "financial"],
    "insurance": ["insurance", "financial"],
    "investment": ["investment", "financial", "asset management"],
    "asset management": ["investment", "financial", "asset management"],
    "wealth": ["investment", "financial", "asset management"],
    "e-commerce": ["e-commerce", "retail"],
    "ecommerce": ["e-commerce", "retail"],
    "retail": ["retail", "e-commerce"],
    "healthcare": ["healthcare", "healthtech"],
    "medical": ["healthcare", "medical"],
    "pharmaceutical": ["pharmaceutical", "biotech"],
    "biotech": ["pharmaceutical", "biotech", "biotechnology"],
    "software": ["software", "it services"],
    "web": ["software", "it services"],
    "hardware": ["hardware", "electronics"],
    "cybersecurity": ["cybersecurity", "security"],
    "security": ["cybersecurity", "security"],
    "manufacturing": ["manufacturing", "machinery", "automotive"],
    "automotive": ["automotive", "manufacturing"],
    "energy": ["energy", "renewable energy", "oil & gas"],
    "renewable": ["renewable energy", "energy"],
    "oil & gas": ["oil & gas", "energy"],
    "natural gas": ["oil & gas", "energy"],
    "web3": ["web3", "blockchain"],
    "blockchain": ["web3", "blockchain"],
    "crypto": ["web3", "blockchain"],
}

# Generic role keywords that can match any sector when no specific product is found
GENERIC_ROLE_KEYWORDS = [
    "manager", 
    "engineer", 
    "developer", 
    "analyst", 
    "consultant", 
    "director", 
    "lead", 
    "specialist", 
    "coordinator", 
    "administrator",
    "associate",
    "executive",
    "officer",
    "advisor",
    "strategist"
]

# BUCKET_COMPANIES extended with financial_services bucket and other existing buckets
BUCKET_COMPANIES = {
    "pharma_biotech": {
        "global": ["Pfizer", "Roche", "Novartis", "Johnson & Johnson", "Merck", "GSK", "Sanofi", "AstraZeneca", "Bayer"],
        "apac": ["Takeda", "CSL", "Sino Biopharm", "Sun Pharma", "Daiichi Sankyo"]
    },
    "medical_devices": {
        "global": ["Johnson & Johnson", "Medtronic", "Abbott", "Baxter", "Stryker", "BD", "Philips Healthcare", "Siemens Healthineers"],
        "apac": ["Terumo", "Nipro", "Wuxi AppTec (Devices)"]
    },
    "diagnostics": {
        "global": ["Roche Diagnostics", "Siemens Healthineers", "Abbott Diagnostics", "BD", "Qiagen", "Bio-Rad"],
        "apac": ["Sysmex", "Mindray"]
    },
    "clinical_research": {
        "global": ["IQVIA", "Labcorp", "ICON", "Parexel", "PPD", "Syneos Health"],
        "apac": ["Novotech", "Tigermed"]
    },
    "healthtech": {
        "global": ["Philips", "Siemens Healthineers", "GE HealthCare", "Cerner (Oracle Health)", "Epic Systems"],
        "apac": ["HealthHub", "IHiS", "Ramsay Sime Darby Health Care"]
    },
    "technology": {
        "global": ["Microsoft", "Amazon Web Services", "Google Cloud", "Snowflake", "Databricks"],
        "apac": ["Tencent Cloud", "Alibaba Cloud"]
    },
    "manufacturing": {
        "global": ["Siemens", "ABB", "Rockwell Automation", "Schneider Electric", "Bosch"],
        "apac": ["Mitsubishi Electric", "FANUC", "Yaskawa"]
    },
    "energy": {
        "global": ["Shell", "BP", "TotalEnergies", "Schneider Electric", "Siemens Energy"],
        "apac": ["PETRONAS", "Sembcorp", "Keppel"]
    },
    "gaming": {
        "global": ["Sony Interactive Entertainment", "Ubisoft", "Electronic Arts", "Nintendo", "Activision Blizzard"],
        "apac": ["Tencent", "NetEase", "Bandai Namco"]
    },
    "web3": {
        "global": ["Coinbase", "Consensys", "Binance", "Circle"],
        "apac": ["OKX", "Bybit"]
    },
    # New: Financial Services bucket to align with sectors.json "Financial Services > ..."
    "financial_services": {
        "global": [
            "J.P. Morgan", "Goldman Sachs", "Morgan Stanley", "BlackRock", "UBS", "Credit Suisse", "HSBC", "Citi", "BNP Paribas", "Deutsche Bank",
            "Standard Chartered", "State Street", "Northern Trust", "Schroders", "Fidelity"
        ],
        "apac": [
            "Samsung Life Insurance", "Hana Financial Investment", "Mirae Asset", "KB Asset Management", "NH Investment & Securities",
            "Korea Investment & Securities", "Shinhan Investment Corp", "Samsung Securities", "Samsung Fire & Marine Insurance", "Hyundai Marine & Fire Insurance",
            "DB Insurance", "Meritz Fire & Marine Insurance", "Tong Yang Securities", "Woori Investment Bank", "Daishin Securities", "Hana Securities",
            "Kiwoom Securities", "KTB Investment & Securities", "Eugene Investment & Securities", "Korea Life Insurance", "LSM Investment",
            "Shinhan BNP Paribas Asset Management", "Samsung SDS", "LG CNS", "SK C&C", "POSCO ICT", "Hyundai Information Technology", "Hanmi Financial",
            "Nonghyup Bank", "Lotte Card"
        ]
    }
}

BUCKET_JOB_TITLES = {
    "pharma_biotech": ["Regulatory Affairs Manager", "Clinical Research Associate", "Pharmacovigilance Specialist", "Medical Affairs Manager", "Quality Assurance Specialist", "CMC Scientist", "Biostatistician", "Clinical Project Manager"],
    "medical_devices": ["Regulatory Affairs Manager", "Quality Engineer", "Clinical Affairs Specialist", "Design Control Engineer", "Risk Management Engineer", "Product Manager (Medical Device)", "Manufacturing Engineer"],
    "diagnostics": ["IVD Regulatory Specialist", "Quality Systems Engineer", "Clinical Application Specialist", "Assay Development Scientist", "Validation Engineer"],
    "clinical_research": ["CRA", "Senior CRA", "Clinical Project Manager", "Clinical Trial Manager", "Study Start-Up Specialist"],
    "healthtech": ["Product Manager", "Clinical Informatics Lead", "Healthcare Data Scientist", "Interoperability Engineer", "Implementation Consultant"],
    "technology": ["Software Engineer", "ML Engineer", "Data Scientist", "Solutions Architect", "Security Engineer", "MLOps Engineer"],
    "manufacturing": ["Manufacturing Engineer", "Quality Engineer", "Process Engineer", "Supply Chain Analyst", "Automation Engineer"],
    "energy": ["Energy Analyst", "Grid Integration Engineer", "Sustainability Manager", "HSE Engineer"],
    "gaming": ["Game Producer", "Gameplay Engineer", "Level Designer", "Technical Artist"],
    "web3": ["Blockchain Engineer", "Smart Contract Developer", "Web3 Product Manager"],
    "other": ["Project Manager", "Operations Manager", "Business Analyst", "Data Analyst"],
    "financial_services": ["Investment Analyst", "Product Manager (Wealth/Investment)", "Portfolio Manager", "Risk Analyst", "Payments Product Manager", "Fintech Product Manager", "Relationship Manager", "Asset Manager"]
}

