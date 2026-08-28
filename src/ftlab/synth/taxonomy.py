"""Domain vocabulary for federal public health contracting.

Everything the generated world is assembled from: agencies, capabilities,
vehicles, NAICS/PSC codes, socioeconomic categories, and the name pools used to
mint fictional companies.

The capability entries carry two fields that exist purely to make hard negatives
possible. ``adjacent`` names capabilities a careless reader would treat as
interchangeable, which is how tier-2 "transferable" candidates are built. And
``decoy_terms`` holds vocabulary that collides on keywords while meaning
something else entirely -- "surveillance" in epidemiology versus in physical
security -- which is how tier-1 traps are built.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilitySpec:
    id: str
    name: str
    family: str
    adjacent: tuple[str, ...] = ()
    decoy_terms: tuple[str, ...] = ()


CAPABILITIES: tuple[CapabilitySpec, ...] = (
    # --- epidemiology & surveillance ---
    CapabilitySpec(
        "epi_surveillance",
        "Epidemiologic Surveillance",
        "epidemiology",
        adjacent=("outbreak_response", "biostatistics", "registry_dev"),
        decoy_terms=("video surveillance", "perimeter monitoring", "site security"),
    ),
    CapabilitySpec(
        "outbreak_response",
        "Outbreak Investigation & Response",
        "epidemiology",
        adjacent=("epi_surveillance", "emergency_preparedness"),
    ),
    CapabilitySpec(
        "biostatistics",
        "Biostatistics & Statistical Analysis",
        "analytics",
        adjacent=("data_science", "program_evaluation"),
    ),
    CapabilitySpec(
        "data_science",
        "Data Science & Predictive Modeling",
        "analytics",
        adjacent=("biostatistics", "health_informatics"),
    ),
    CapabilitySpec(
        "registry_dev",
        "Disease & Patient Registry Development",
        "health_it",
        adjacent=("health_informatics", "epi_surveillance"),
    ),
    # --- health IT ---
    CapabilitySpec(
        "health_informatics",
        "Public Health Informatics",
        "health_it",
        adjacent=("data_modernization", "registry_dev", "data_science"),
    ),
    CapabilitySpec(
        "fhir_interop",
        "FHIR / HL7 Interoperability Engineering",
        "health_it",
        adjacent=("data_modernization", "ehr_integration"),
        decoy_terms=("systems integration", "enterprise architecture"),
    ),
    CapabilitySpec(
        "data_modernization",
        "Public Health Data Modernization",
        "health_it",
        adjacent=("health_informatics", "fhir_interop", "cloud_migration"),
    ),
    CapabilitySpec(
        "ehr_integration",
        "EHR Integration & Clinical Data Exchange",
        "health_it",
        adjacent=("fhir_interop", "ehr_integration"),
    ),
    CapabilitySpec(
        "cloud_migration",
        "Cloud Migration & FedRAMP Compliance",
        "health_it",
        adjacent=("data_modernization", "cybersecurity"),
        decoy_terms=("cloud storage reseller", "commercial SaaS licensing"),
    ),
    CapabilitySpec(
        "cybersecurity",
        "Health System Cybersecurity",
        "health_it",
        adjacent=("cloud_migration",),
    ),
    # --- program domains ---
    CapabilitySpec(
        "immunization",
        "Immunization Program Support",
        "program",
        adjacent=("vaccine_safety", "registry_dev"),
    ),
    CapabilitySpec(
        "vaccine_safety",
        "Vaccine Safety Monitoring",
        "program",
        adjacent=("immunization", "epi_surveillance"),
    ),
    CapabilitySpec(
        "hiv_sti",
        "HIV / STI / TB Prevention Programs",
        "program",
        adjacent=("health_equity", "epi_surveillance"),
    ),
    CapabilitySpec(
        "maternal_child",
        "Maternal & Child Health Programs",
        "program",
        adjacent=("health_equity", "program_evaluation"),
    ),
    CapabilitySpec(
        "chronic_disease",
        "Chronic Disease Prevention",
        "program",
        adjacent=("health_communications", "program_evaluation"),
    ),
    CapabilitySpec(
        "behavioral_health",
        "Behavioral Health & Substance Use",
        "program",
        adjacent=("health_equity", "program_evaluation"),
    ),
    CapabilitySpec(
        "health_equity",
        "Health Equity & Social Determinants",
        "program",
        adjacent=("maternal_child", "hiv_sti", "community_engagement"),
    ),
    CapabilitySpec(
        "environmental_health",
        "Environmental Health & Toxicology",
        "program",
        adjacent=("epi_surveillance",),
    ),
    CapabilitySpec(
        "global_health",
        "Global Health & PEPFAR Support",
        "program",
        adjacent=("hiv_sti", "epi_surveillance"),
    ),
    # --- preparedness ---
    CapabilitySpec(
        "emergency_preparedness",
        "Public Health Emergency Preparedness",
        "preparedness",
        adjacent=("outbreak_response", "medical_countermeasures"),
        decoy_terms=("facility emergency planning", "physical security readiness"),
    ),
    CapabilitySpec(
        "medical_countermeasures",
        "Medical Countermeasures & SNS Support",
        "preparedness",
        adjacent=("emergency_preparedness", "lab_systems"),
    ),
    CapabilitySpec(
        "lab_systems",
        "Laboratory Systems & Biosafety",
        "preparedness",
        adjacent=("medical_countermeasures", "epi_surveillance"),
    ),
    # --- services ---
    CapabilitySpec(
        "program_evaluation",
        "Program Evaluation & Performance Measurement",
        "services",
        adjacent=("biostatistics", "health_policy"),
    ),
    CapabilitySpec(
        "health_communications",
        "Health Communications & Campaign Design",
        "services",
        adjacent=("community_engagement", "chronic_disease"),
        decoy_terms=("corporate marketing", "advertising media buying"),
    ),
    CapabilitySpec(
        "community_engagement",
        "Community Engagement & Partner Mobilization",
        "services",
        adjacent=("health_equity", "health_communications"),
    ),
    CapabilitySpec(
        "workforce_training",
        "Public Health Workforce Development",
        "services",
        adjacent=("health_communications", "program_evaluation"),
    ),
    CapabilitySpec(
        "health_policy",
        "Health Policy & Economic Analysis",
        "services",
        adjacent=("program_evaluation", "biostatistics"),
    ),
    CapabilitySpec(
        "grants_management",
        "Grants & Cooperative Agreement Management",
        "services",
        adjacent=("program_evaluation",),
        decoy_terms=("commercial contract administration", "accounts payable"),
    ),
    CapabilitySpec(
        "clinical_quality",
        "Clinical Quality Measurement",
        "services",
        adjacent=("ehr_integration", "program_evaluation"),
    ),
)

CAPABILITY_BY_ID = {c.id: c for c in CAPABILITIES}
CAPABILITY_IDS = tuple(c.id for c in CAPABILITIES)
CAPABILITY_FAMILIES = tuple(dict.fromkeys(c.family for c in CAPABILITIES))


# ---------------------------------------------------------------------------
# agencies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgencySpec:
    id: str
    abbrev: str
    name: str
    parent: str
    subunits: tuple[str, ...]
    favors: tuple[str, ...] = field(default=())


AGENCIES: tuple[AgencySpec, ...] = (
    AgencySpec(
        "cdc",
        "CDC",
        "Centers for Disease Control and Prevention",
        "HHS",
        (
            "NCIRD (Immunization & Respiratory Diseases)",
            "NCHHSTP (HIV, Viral Hepatitis, STD, TB)",
            "NCEZID (Emerging & Zoonotic Infectious Diseases)",
            "NCCDPHP (Chronic Disease Prevention)",
            "CSELS (Surveillance, Epidemiology & Laboratory Services)",
            "NCEH (Environmental Health)",
            "OPHDST (Public Health Data, Surveillance & Technology)",
        ),
        favors=("epi_surveillance", "immunization", "data_modernization", "lab_systems"),
    ),
    AgencySpec(
        "hrsa",
        "HRSA",
        "Health Resources and Services Administration",
        "HHS",
        (
            "MCHB (Maternal & Child Health Bureau)",
            "BPHC (Bureau of Primary Health Care)",
            "HAB (HIV/AIDS Bureau)",
            "BHW (Bureau of Health Workforce)",
        ),
        favors=("maternal_child", "health_equity", "workforce_training", "hiv_sti"),
    ),
    AgencySpec(
        "cms",
        "CMS",
        "Centers for Medicare & Medicaid Services",
        "HHS",
        (
            "CCSQ (Clinical Standards & Quality)",
            "CMCS (Medicaid & CHIP Services)",
            "CMMI (Innovation Center)",
        ),
        favors=("clinical_quality", "health_policy", "ehr_integration", "data_science"),
    ),
    AgencySpec(
        "nih",
        "NIH",
        "National Institutes of Health",
        "HHS",
        (
            "NIAID (Allergy & Infectious Diseases)",
            "NCI (Cancer Institute)",
            "NIMH (Mental Health)",
            "NICHD (Child Health & Human Development)",
        ),
        favors=("biostatistics", "registry_dev", "data_science", "program_evaluation"),
    ),
    AgencySpec(
        "aspr",
        "ASPR",
        "Administration for Strategic Preparedness and Response",
        "HHS",
        ("Office of Preparedness", "SNS (Strategic National Stockpile)", "BARDA"),
        favors=("emergency_preparedness", "medical_countermeasures", "lab_systems"),
    ),
    AgencySpec(
        "samhsa",
        "SAMHSA",
        "Substance Abuse and Mental Health Services Administration",
        "HHS",
        ("CBHSQ (Behavioral Health Statistics)", "CSAT (Substance Abuse Treatment)"),
        favors=("behavioral_health", "program_evaluation", "health_equity"),
    ),
    AgencySpec(
        "ahrq",
        "AHRQ",
        "Agency for Healthcare Research and Quality",
        "HHS",
        ("CQuIPS (Quality Improvement & Patient Safety)", "CEPI (Evidence & Practice)"),
        favors=("clinical_quality", "program_evaluation", "health_policy"),
    ),
    AgencySpec(
        "ihs",
        "IHS",
        "Indian Health Service",
        "HHS",
        ("Office of Information Technology", "Office of Public Health Support"),
        favors=("health_equity", "ehr_integration", "community_engagement"),
    ),
    AgencySpec(
        "fda",
        "FDA",
        "Food and Drug Administration",
        "HHS",
        ("CBER (Biologics)", "CDER (Drug Evaluation)", "CFSAN (Food Safety)"),
        favors=("vaccine_safety", "biostatistics", "epi_surveillance"),
    ),
    AgencySpec(
        "va",
        "VA",
        "Department of Veterans Affairs",
        "VA",
        ("VHA Office of Health Informatics", "VHA Office of Research & Development"),
        favors=("ehr_integration", "behavioral_health", "clinical_quality"),
    ),
    AgencySpec(
        "dha",
        "DHA",
        "Defense Health Agency",
        "DoD",
        ("Public Health Directorate", "Health IT Directorate"),
        favors=("epi_surveillance", "ehr_integration", "emergency_preparedness"),
    ),
    AgencySpec(
        "usaid",
        "USAID",
        "U.S. Agency for International Development",
        "USAID",
        ("Bureau for Global Health",),
        favors=("global_health", "hiv_sti", "health_equity"),
    ),
    AgencySpec(
        "state_hd",
        "State DOH",
        "State Departments of Health (multi-state)",
        "State",
        ("Epidemiology Division", "Immunization Program", "Health Informatics Office"),
        favors=("immunization", "epi_surveillance", "health_informatics"),
    ),
    AgencySpec(
        "acf",
        "ACF",
        "Administration for Children and Families",
        "HHS",
        ("Office of Planning, Research & Evaluation", "Office of Head Start"),
        favors=("maternal_child", "program_evaluation", "health_equity"),
    ),
)

AGENCY_BY_ID = {a.id: a for a in AGENCIES}
AGENCY_IDS = tuple(a.id for a in AGENCIES)


# ---------------------------------------------------------------------------
# contract vehicles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VehicleSpec:
    id: str
    name: str
    kind: str
    naics: tuple[str, ...]
    note: str = ""


VEHICLES: tuple[VehicleSpec, ...] = (
    VehicleSpec("ciosp3", "CIO-SP3 Small Business", "GWAC", ("541512", "541511", "541519")),
    VehicleSpec("ciosp4", "CIO-SP4", "GWAC", ("541512", "541511", "541519")),
    VehicleSpec("gsa_mas", "GSA MAS (SIN 541611 / 541690)", "MAS", ("541611", "541690")),
    VehicleSpec("stars3", "8(a) STARS III", "GWAC", ("541512", "541519"), "8(a) only"),
    VehicleSpec("oasis_plus", "OASIS+ Unrestricted", "IDIQ", ("541611", "541618", "541690")),
    VehicleSpec("oasis_sb", "OASIS+ Small Business", "IDIQ", ("541611", "541618")),
    VehicleSpec("sewp6", "NASA SEWP VI", "GWAC", ("541512", "334111")),
    VehicleSpec("alliant2", "Alliant 2", "GWAC", ("541512",)),
    VehicleSpec("phsc_bpa", "HHS Public Health Services BPA", "BPA", ("541611", "923120")),
    VehicleSpec("cdc_iddiq", "CDC Public Health Analytics IDIQ", "IDIQ", ("541690", "541611")),
    VehicleSpec("cms_sparc", "CMS SPARC", "IDIQ", ("541512", "541611")),
    VehicleSpec("nih_cioidiq", "NIH Scientific Support IDIQ", "IDIQ", ("541715", "541690")),
    VehicleSpec("open", "Full & Open / Standalone", "Standalone", ()),
)

VEHICLE_BY_ID = {v.id: v for v in VEHICLES}
VEHICLE_IDS = tuple(v.id for v in VEHICLES)


# ---------------------------------------------------------------------------
# codes and categories
# ---------------------------------------------------------------------------

NAICS: dict[str, str] = {
    "541611": "Administrative Management & General Management Consulting",
    "541618": "Other Management Consulting Services",
    "541690": "Other Scientific & Technical Consulting Services",
    "541714": "R&D in Biotechnology (except Nanobiotechnology)",
    "541715": "R&D in Physical, Engineering & Life Sciences",
    "541511": "Custom Computer Programming Services",
    "541512": "Computer Systems Design Services",
    "541519": "Other Computer Related Services",
    "541720": "R&D in the Social Sciences and Humanities",
    "541910": "Marketing Research & Public Opinion Polling",
    "611430": "Professional & Management Development Training",
    "621999": "All Other Miscellaneous Ambulatory Health Care Services",
    "923120": "Administration of Public Health Programs",
}

PSC: dict[str, str] = {
    "B505": "Special Studies/Analysis - Epidemiology",
    "B506": "Special Studies/Analysis - Public Health",
    "R408": "Program Management/Support Services",
    "R499": "Other Professional Services",
    "D302": "IT and Telecom - Systems Development",
    "D307": "IT and Telecom - IT Strategy and Architecture",
    "D310": "IT and Telecom - Cyber Security",
    "AN12": "R&D - Health Services, Applied Research",
    "U008": "Education/Training - Training/Curriculum Development",
}

SET_ASIDES: tuple[str, ...] = (
    "Full & Open",
    "Small Business",
    "8(a)",
    "SDVOSB",
    "WOSB",
    "HUBZone",
)

SOCIOECONOMIC: tuple[str, ...] = ("8(a)", "SDVOSB", "WOSB", "EDWOSB", "HUBZone", "SDB")

CPARS_RATINGS: tuple[str, ...] = (
    "Exceptional",
    "Very Good",
    "Satisfactory",
    "Marginal",
)

# Weighted so the library looks like a real one: mostly good, occasionally not.
CPARS_WEIGHTS: tuple[int, ...] = (25, 45, 25, 5)


# ---------------------------------------------------------------------------
# company naming
# ---------------------------------------------------------------------------

# Deliberately generic geographic and nature words. Combined with the suffixes
# below they produce names that read like federal health contractors without
# reproducing any actual firm's identity.
NAME_HEADS: tuple[str, ...] = (
    "Allegheny", "Bluestone", "Brightwater", "Cardinal Ridge", "Cedarbrook",
    "Chesapeake", "Clearfield", "Concord Point", "Cypress Hollow", "Dominion Park",
    "Eastport", "Elkhorn", "Fairhaven", "Foxglove", "Granite Bay", "Greywood",
    "Harborview", "Hollybrook", "Ironwood", "Junipero", "Keystone Bluff",
    "Lakeshore", "Larkspur", "Longmeadow", "Marbury", "Merrivale", "Northfield",
    "Oakhurst", "Orchard Point", "Pinehollow", "Quarrystone", "Redbank",
    "Riverbend", "Saltmarsh", "Silverthorn", "Stonegate", "Sumac Hill",
    "Tanglewood", "Thornbury", "Tidewater", "Umbergrove", "Vantage Hill",
    "Wexford", "Whitmore", "Willowmere", "Windham", "Yarrowfield", "Zephyr Grove",
    "Ashcombe", "Belmont Reach", "Coldspring", "Dunmore", "Everglade",
    "Foxmoor", "Glenarden", "Havenridge", "Inglewood Park", "Kestrel",
)

NAME_TAILS: tuple[str, ...] = (
    "Health Analytics", "Public Health Group", "Health Sciences",
    "Informatics", "Research Partners", "Health Solutions", "Scientific",
    "Health Advisory", "Data Systems", "Health Strategies", "Epidemiology Group",
    "Health Consulting", "Applied Sciences", "Health Systems", "Analytics Group",
    "Public Health Partners", "Biostatistics", "Health Technologies",
    "Population Health", "Health Research",
)

# Nothing generated may collide with a real federal health contractor: the
# corpus attaches invented CPARS ratings and performance history to every name
# it produces, and hanging that on a real firm would be defamatory rather than
# synthetic.
REAL_FIRM_BLOCKLIST: frozenset[str] = frozenset(
    {
        "abt", "icf", "rti", "deloitte", "booz", "allen", "hamilton", "leidos",
        "guidehouse", "mitre", "westat", "norc", "mathematica", "peraton",
        "accenture", "ibm", "maximus", "serco", "caci", "saic", "gdit",
        "dynamics", "northrop", "grumman", "lockheed", "martin", "aveshka",
        "karna", "eagle", "chickasaw", "dlh", "csra", "cognosante", "attain",
        "palantir", "oracle", "cerner", "epic", "mckinsey", "kpmg", "pwc",
        "ernst", "young", "bain", "aecom", "battelle", "mitretek", "noblis",
    }
)

US_STATES: tuple[str, ...] = (
    "MD", "VA", "DC", "GA", "NC", "PA", "MA", "NY", "IL", "TX",
    "CA", "WA", "CO", "MN", "MI", "OH", "FL", "TN", "AZ", "NM",
)

PERSON_FIRST: tuple[str, ...] = (
    "Amara", "Benedict", "Camille", "Darius", "Elena", "Faisal", "Grace",
    "Hiroshi", "Imani", "Jonas", "Kavita", "Lucia", "Malik", "Nadia",
    "Oscar", "Priya", "Quentin", "Rosalind", "Samir", "Tomas", "Ursula",
    "Viktor", "Willa", "Xiomara", "Yusuf", "Zara", "Adaeze", "Bartholomew",
    "Clarissa", "Dmitri", "Esperanza", "Farid", "Giselle", "Hannah",
)

PERSON_LAST: tuple[str, ...] = (
    "Achebe", "Bergstrom", "Castellanos", "Duarte", "Eriksen", "Farhadi",
    "Gallagher", "Hollingsworth", "Ibarra", "Jankowski", "Kowalczyk",
    "Lindqvist", "Mbeki", "Novotny", "Okonkwo", "Pereira", "Quintanilla",
    "Rasmussen", "Sandoval", "Thibodeaux", "Ueda", "Villanueva", "Whitfield",
    "Xu", "Yamamoto", "Zielinski", "Adebayo", "Brennan", "Chaudhry",
    "Delacroix", "Eastwood", "Fontaine", "Grimaldi", "Haverford",
)

PERSON_ROLES: tuple[str, ...] = (
    "Program Manager",
    "Principal Investigator",
    "Technical Director",
    "Lead Epidemiologist",
    "Senior Biostatistician",
    "Health IT Architect",
    "Evaluation Lead",
    "Informatics Lead",
    "Deputy Program Manager",
    "Subject Matter Expert",
)
