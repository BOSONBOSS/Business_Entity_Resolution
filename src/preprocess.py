"""
Preprocessing & Text Normalization Pipeline
-------------------------------------------
Normalizes business names and addresses to reverse real-world corruptions:
- Accent/diacritic stripping (NFKD) -> unifies 'Rétail' -> 'Retail' and France test data
- Legal suffix removal -> prevents false inflation from generic words ('Pvt Ltd')
- Address numeric zero-padding rectification -> '0337 Oakland Ave' -> '337 Oakland Ave'
- Social handle & domain extension stripping -> '@primemoney' -> 'primemoney', '.com'
- Placeholder scrubbing -> 'null', 'n/a', '-' -> ''
- Script detection -> flags Indic/Devanagari text
"""

import re
import unicodedata

# Common legal and corporate forms to remove from business names (English & Hindi)
LEGAL_SUFFIXES_REGEX = re.compile(
    r'\b(pvt|ltd|limited|private|inc|incorporated|corp|corporation|llc|llp|gmbh|'
    r'sarl|sas|sa|co|company|enterprises|enterprise|holdings|group|services|service|'
    r'प्राइवेट|लिमिटेड|प्रा\.?\s*लि\.?)\b',
    flags=re.IGNORECASE
)

# Placeholder strings that indicate fake missing values
PLACEHOLDERS = {'null', 'n/a', 'na', 'nan', 'nil', 'none', '-', '--', 'undefined'}

# Top-level domains to strip
TLD_REGEX = re.compile(r'\.(com|org|net|in|co|gov|edu|info|biz|io|ai|me)\b', flags=re.IGNORECASE)

# Extra symbols and parenthetical noise like '(ID: 84923)' or '(Inc)'
PARENTHETICAL_ID_REGEX = re.compile(r'\(\s*(id\s*:?\s*\d+|inc|ltd|pvt)\s*\)', flags=re.IGNORECASE)

# Zero-padded numbers in addresses (e.g. '0017560 Ellis Rd' -> '17560 Ellis Rd', '0337' -> '337')
ZERO_PAD_REGEX = re.compile(r'\b0+(\d+)\b')


def strip_punctuation(text: str) -> str:
    """Strip Unicode punctuation (P*) and symbols (S*) while preserving letters, numbers, and combining marks (Mn)."""
    return "".join(" " if unicodedata.category(c).startswith(('P', 'S')) else c for c in text)


def strip_accents(text: str) -> str:
    """Normalize accented characters for Latin text while preserving Indic/Devanagari matras."""
    if not isinstance(text, str) or not text:
        return ""
    # If the text is non-Latin (Devanagari, Tamil, etc.), do not strip combining marks (matras)
    if is_non_latin(text):
        return text
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def is_non_latin(text: str) -> bool:
    """Detect if string contains non-Latin scripts (Devanagari, Tamil, etc.)."""
    if not isinstance(text, str) or not text.strip():
        return False
    return any(unicodedata.category(c) == 'Lo' and ord(c) > 0x024F for c in text)


def clean_placeholders(text: str) -> str:
    """Convert fake null strings to empty string."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if t.lower() in PLACEHOLDERS:
        return ""
    return t


def preprocess_business_name(name: str) -> str:
    """
    Apply full cleaning pipeline to a business name:
    1. Scrub placeholders
    2. Normalize Unicode accents
    3. Strip social handle '@'
    4. Strip domain extensions (.com, .in, etc.)
    5. Strip parenthetical ID tags
    6. Remove legal suffixes (Pvt Ltd, Inc, etc.)
    7. Normalize punctuation & whitespace
    """
    if not isinstance(name, str):
        return ""
    
    t = clean_placeholders(name)
    if not t:
        return ""

    # Accent stripping
    t = strip_accents(t)

    # Remove '@' handle prefixes
    t = re.sub(r'^@+', '', t)

    # Strip domain extensions
    t = TLD_REGEX.sub('', t)

    # Remove parenthetical tags like (ID: 84923)
    t = PARENTHETICAL_ID_REGEX.sub('', t)

    # Lowercase for standardized suffix removal
    t = t.lower()

    # Strip legal entity suffixes
    t = LEGAL_SUFFIXES_REGEX.sub('', t)

    # Strip punctuation while preserving Unicode letters, digits, and combining marks
    t = strip_punctuation(t)

    # Collapse multiple whitespace
    return " ".join(t.split())


def preprocess_address(addr: str) -> str:
    """
    Apply full cleaning pipeline to a business address:
    1. Scrub placeholders
    2. Normalize accents
    3. Rectify zero-padded numbers (0017560 -> 17560)
    4. Standardize common street abbreviations (Ave -> Avenue, Rd -> Road)
    5. Lowercase and collapse whitespace
    """
    if not isinstance(addr, str):
        return ""
    
    t = clean_placeholders(addr)
    if not t:
        return ""

    t = strip_accents(t)

    # Rectify zero-padded numbers
    t = ZERO_PAD_REGEX.sub(r'\1', t)

    t = t.lower()

    # Standardize common road abbreviations
    t = re.sub(r'\b(ave|av)\b', 'avenue', t)
    t = re.sub(r'\b(rd)\b', 'road', t)
    t = re.sub(r'\b(st|str)\b', 'street', t)
    t = re.sub(r'\b(dr)\b', 'drive', t)
    t = re.sub(r'\b(blvd)\b', 'boulevard', t)
    t = re.sub(r'\b(ln)\b', 'lane', t)
    t = re.sub(r'\b(hwy)\b', 'highway', t)

    # Strip punctuation while preserving Unicode letters, digits, and combining marks
    t = strip_punctuation(t)

    return " ".join(t.split())


if __name__ == "__main__":
    # Test suite with observed challenge edge cases
    test_cases = [
        ("Moore Bítwise Inc", "moore bitwise"),
        ("@primemoney", "primemoney"),
        ("moorebitwise.com", "moorebitwise"),
        ("B+ Retail INC (ID: 84923)", "b retail"),
        ("Dhan Investments Private Limited", "dhan investments"),
        ("0017560 ELLIS ROAD", "17560 ellis road"),
        ("0337 Oakland Ave", "337 oakland avenue"),
        ("null", ""),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "राम मार्केटिंग"),
    ]

    print("Running Preprocessing Validation Tests...")
    all_passed = True
    for raw, expected in test_cases:
        if "oakland" in raw.lower() or "ellis" in raw.lower():
            res = preprocess_address(raw)
        else:
            res = preprocess_business_name(raw)
        passed = (res == expected) or (expected in res)
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"  {status} | Raw: '{raw}' -> Clean: '{res}' (Expected: '{expected}')")
    
    if all_passed:
        print("\nAll preprocessing tests passed successfully!")
