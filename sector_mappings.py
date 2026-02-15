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
    "banking": ["banking", "financial"],
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
    "oil": ["oil & gas", "energy"],
    "gas": ["oil & gas", "energy"],
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
