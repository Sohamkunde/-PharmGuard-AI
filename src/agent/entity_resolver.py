"""Drug Entity Resolver for PharmGuard.

Resolves misspellings, typos, and brand/generic mismatches against the 40-drug
PharmGuard cohort using rapidfuzz token similarity matching with cutoff=0.75.
Expands canonical drugs to comprehensive brand and generic alias sets for downstream
pre-filtering and entity-constrained reranking.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple, Any

try:
    import rapidfuzz
    from rapidfuzz import fuzz, process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    import difflib
    _RAPIDFUZZ_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("entity_resolver")

# 40 Canonical Drugs in the PharmGuard Cohort
COHORT_40_DRUGS: List[str] = [
    "warfarin",
    "apixaban",
    "atorvastatin",
    "lisinopril",
    "amiodarone",
    "digoxin",
    "clopidogrel",
    "spironolactone",
    "metformin",
    "empagliflozin",
    "semaglutide",
    "glimepiride",
    "levothyroxine",
    "paxlovid",
    "ciprofloxacin",
    "azithromycin",
    "fluconazole",
    "clarithromycin",
    "rifampin",
    "metronidazole",
    "sertraline",
    "fluoxetine",
    "phenelzine",
    "lithium",
    "sumatriptan",
    "alprazolam",
    "bupropion",
    "venlafaxine",
    "ibuprofen",
    "methotrexate",
    "cyclosporine",
    "tacrolimus",
    "prednisone",
    "tramadol",
    "oxycodone",
    "colchicine",
    "allopurinol",
    "azathioprine",
    "celecoxib",
    "aspirin",
]

# Canonical Generic + Brand / Active Ingredient Alias Mapping
DRUG_BRAND_GENERIC_MAP: Dict[str, List[str]] = {
    "paxlovid": ["paxlovid", "nirmatrelvir", "ritonavir"],
    "metformin": ["metformin", "glucophage", "fortamet", "glumetza", "riomet"],
    "atorvastatin": ["atorvastatin", "lipitor"],
    "warfarin": ["warfarin", "coumadin", "jantoven"],
    "apixaban": ["apixaban", "eliquis"],
    "lisinopril": ["lisinopril", "prinivil", "zestril", "qbrelis"],
    "amiodarone": ["amiodarone", "cordarone", "pacerone", "nexterone"],
    "digoxin": ["digoxin", "lanoxin"],
    "clopidogrel": ["clopidogrel", "plavix"],
    "spironolactone": ["spironolactone", "aldactone", "carospir"],
    "empagliflozin": ["empagliflozin", "jardiance"],
    "semaglutide": ["semaglutide", "ozempic", "wegovy", "rybelsus"],
    "glimepiride": ["glimepiride", "amaryl"],
    "levothyroxine": ["levothyroxine", "synthroid", "levoxyl", "tirosint", "unithroid"],
    "ciprofloxacin": ["ciprofloxacin", "cipro"],
    "azithromycin": ["azithromycin", "zithromax", "z-pak", "zpak"],
    "fluconazole": ["fluconazole", "diflucan"],
    "clarithromycin": ["clarithromycin", "biaxin"],
    "rifampin": ["rifampin", "rifadin", "rimactane"],
    "metronidazole": ["metronidazole", "flagyl"],
    "sertraline": ["sertraline", "zoloft"],
    "fluoxetine": ["fluoxetine", "prozac", "sarafem"],
    "phenelzine": ["phenelzine", "nardil"],
    "lithium": ["lithium", "eskalith", "lithobid"],
    "sumatriptan": ["sumatriptan", "imitrex", "onzetra"],
    "alprazolam": ["alprazolam", "xanax", "niravam"],
    "bupropion": ["bupropion", "wellbutrin", "zyban", "forfivo", "aplenzin"],
    "venlafaxine": ["venlafaxine", "effexor"],
    "ibuprofen": ["ibuprofen", "advil", "motrin"],
    "methotrexate": ["methotrexate", "trexall", "rasuvo", "otrexup"],
    "cyclosporine": ["cyclosporine", "neoral", "sandimmune", "gengraf", "restasis"],
    "tacrolimus": ["tacrolimus", "prograf", "envarsus", "astagraf"],
    "prednisone": ["prednisone", "deltasone", "rayos"],
    "tramadol": ["tramadol", "ultram", "conzip"],
    "oxycodone": ["oxycodone", "oxycontin", "percocet", "roxicodone"],
    "colchicine": ["colchicine", "colcrys", "mitigare", "gloperba"],
    "allopurinol": ["allopurinol", "zyloprim", "aloprim"],
    "azathioprine": ["azathioprine", "imuran", "azasan"],
    "celecoxib": ["celecoxib", "celebrex"],
    "aspirin": ["aspirin", "bayer", "ecotrin", "bufferin", "acetylsalicylic acid", "asa"],
}

# Reverse lookup: maps any alias token back to its canonical drug name
ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in DRUG_BRAND_GENERIC_MAP.items():
    ALIAS_TO_CANONICAL[canonical.lower()] = canonical.lower()
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical.lower()

# Words that should never be matched as drug candidates
NON_DRUG_STOPWORDS: Set[str] = {
    "can", "a", "patient", "with", "moderate", "severe", "mild", "renal", "impairment",
    "kidney", "disease", "taking", "take", "prescribed", "between", "interactions",
    "interaction", "what", "is", "the", "how", "to", "for", "in", "and", "or", "of",
    "on", "doctor", "daily", "mg", "dose", "doses", "dosage", "dosing", "stock", "price",
    "apple", "bake", "cake", "who", "when", "why", "google", "test", "check", "levels",
    "stage", "chronic", "failure", "clearance", "function", "blood", "high", "risk",
    "give", "given", "giveable", "safe", "safely", "therapy", "treatment", "protocol",
    "unverified", "citation", "citations", "gate", "evidence", "marked", "guardrail",
    "failure", "diagnostic", "diagnostics", "trace", "status", "unmatched", "verified",
}


class DrugEntityResolver:
    """Robust clinical drug entity resolver with fuzzy matching and alias expansion."""

    def __init__(self, score_cutoff: float = 0.75):
        """Initialize resolver with similarity cutoff (0.75 -> 75% match)."""
        self.score_cutoff = score_cutoff
        self.cutoff_100 = score_cutoff * 100.0 if score_cutoff <= 1.0 else score_cutoff
        self.canonical_drugs = list(COHORT_40_DRUGS)
        self.alias_map = dict(DRUG_BRAND_GENERIC_MAP)
        self.alias_to_canonical = dict(ALIAS_TO_CANONICAL)
        # All known searchable drug surface forms (lowercase)
        self.all_surface_forms = list(self.alias_to_canonical.keys())

    def resolve_drug_tokens(self, query: str) -> List[str]:
        """Extract candidate tokens and fuzzy-match to canonical drugs in the cohort.

        Handles misspellings (e.g. 'Metformn' -> 'metformin', 'Paxlovd' -> 'paxlovid')
        and brand/generic synonyms (e.g. 'Glucophage' -> 'metformin', 'Lipitor' -> 'atorvastatin').

        Args:
            query: Clinical query string.

        Returns:
            Deduplicated list of canonical drug names in the order of occurrence in query.
        """
        if not query or not query.strip():
            return []

        query_clean = query.lower()
        # Extract potential word tokens and hyphens
        raw_tokens = re.findall(r"\b[a-z0-9\-]+\b", query_clean)
        
        resolved_canonical: List[str] = []
        seen: Set[str] = set()

        for token in raw_tokens:
            token = token.strip("-").strip()
            if not token or len(token) < 4 or token in NON_DRUG_STOPWORDS:
                continue

            # 1. Exact match against alias dictionary
            if token in self.alias_to_canonical:
                canonical = self.alias_to_canonical[token]
                if canonical not in seen:
                    resolved_canonical.append(canonical)
                    seen.add(canonical)
                continue

            # 2. Fuzzy match against all known surface forms
            if _RAPIDFUZZ_AVAILABLE:
                match = process.extractOne(
                    token,
                    self.all_surface_forms,
                    scorer=fuzz.ratio,
                    score_cutoff=self.cutoff_100,
                )
                if match:
                    matched_surface, score, _ = match
                    canonical = self.alias_to_canonical[matched_surface]
                    if canonical not in seen:
                        logger.info(
                            "Fuzzy drug resolved: '%s' -> '%s' (surface: '%s', score=%.1f)",
                            token,
                            canonical,
                            matched_surface,
                            score,
                        )
                        resolved_canonical.append(canonical)
                        seen.add(canonical)
            else:
                # Fallback to standard library difflib
                matches = difflib.get_close_matches(
                    token,
                    self.all_surface_forms,
                    n=1,
                    cutoff=self.score_cutoff,
                )
                if matches:
                    matched_surface = matches[0]
                    canonical = self.alias_to_canonical[matched_surface]
                    if canonical not in seen:
                        logger.info(
                            "Difflib drug resolved: '%s' -> '%s' (surface: '%s')",
                            token,
                            canonical,
                            matched_surface,
                        )
                        resolved_canonical.append(canonical)
                        seen.add(canonical)

        return resolved_canonical

    def resolve_token(self, token: str) -> Optional[str]:
        """Resolve a single candidate token (exact or fuzzy) to a canonical drug, or None."""
        if not token or not token.strip():
            return None
        t = token.strip("-").strip().lower()
        if len(t) < 4 or t in NON_DRUG_STOPWORDS:
            return None
        if t in self.alias_to_canonical:
            return self.alias_to_canonical[t]
        if _RAPIDFUZZ_AVAILABLE:
            match = process.extractOne(
                t,
                self.all_surface_forms,
                scorer=fuzz.ratio,
                score_cutoff=self.cutoff_100,
            )
            if match:
                matched_surface, score, _ = match
                return self.alias_to_canonical[matched_surface]
        else:
            matches = difflib.get_close_matches(
                t,
                self.all_surface_forms,
                n=1,
                cutoff=self.score_cutoff,
            )
            if matches:
                return self.alias_to_canonical[matches[0]]
        return None

    def get_aliases(self, canonical_drug: str) -> List[str]:
        """Get canonical generic + brand alias set for a detected drug."""
        canonical_lower = canonical_drug.lower()
        return self.alias_map.get(canonical_lower, [canonical_lower])

    def expand_drugs(self, canonical_drugs: List[str]) -> Dict[str, List[str]]:
        """Expand list of canonical drugs into mapping of canonical -> full alias set."""
        return {cd: self.get_aliases(cd) for cd in canonical_drugs}

    def get_all_target_aliases(self, canonical_drugs: List[str]) -> Set[str]:
        """Return flattened set of all aliases for the given canonical drugs."""
        aliases: Set[str] = set()
        for cd in canonical_drugs:
            for alias in self.get_aliases(cd):
                aliases.add(alias.lower())
        return aliases


# Global singleton instance
_RESOLVER_INSTANCE: Optional[DrugEntityResolver] = None


def get_entity_resolver() -> DrugEntityResolver:
    """Get or create singleton DrugEntityResolver."""
    global _RESOLVER_INSTANCE
    if _RESOLVER_INSTANCE is None:
        _RESOLVER_INSTANCE = DrugEntityResolver(score_cutoff=0.75)
    return _RESOLVER_INSTANCE
