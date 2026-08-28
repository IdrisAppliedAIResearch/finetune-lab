"""Narrative fragments used to give generated records readable scope text.

Kept apart from the graph itself so the structure stays the source of truth: a
contract's facts are decided first, and this module only puts sentences around
them. Nothing here may introduce a fact the graph does not already hold, or the
golden answers stop being derivable from the graph.
"""

from __future__ import annotations

# One entry per capability id. {n} is filled with a plausible count.
ACTIVITIES: dict[str, tuple[str, ...]] = {
    "epi_surveillance": (
        "designed and operated syndromic surveillance pipelines across {n} jurisdictions",
        "modernized notifiable disease reporting workflows for {n} reporting sites",
        "built and maintained case surveillance dashboards used by {n} epidemiologists",
    ),
    "outbreak_response": (
        "provided surge epidemiology support during {n} multi-state outbreak investigations",
        "stood up rapid case investigation protocols and contact tracing workflows",
        "delivered field deployment support and after-action analysis for {n} responses",
    ),
    "biostatistics": (
        "performed statistical analysis and study design for {n} program evaluations",
        "developed and validated risk-adjustment models for population health measures",
        "led sample design and weighting for a national health survey of {n} respondents",
    ),
    "data_science": (
        "built predictive models forecasting disease burden across {n} counties",
        "developed machine learning pipelines for anomaly detection in reportable conditions",
        "delivered geospatial analytics and small-area estimation products",
    ),
    "registry_dev": (
        "designed and deployed a national disease registry serving {n} contributing sites",
        "migrated legacy registry data into a modern normalized schema",
        "implemented registry data quality monitoring and deduplication logic",
    ),
    "health_informatics": (
        "delivered informatics architecture and data governance for {n} program areas",
        "standardized vocabulary mapping across {n} data sources",
        "built the informatics roadmap guiding a multi-year modernization effort",
    ),
    "fhir_interop": (
        "implemented FHIR R4 APIs connecting {n} clinical data partners",
        "engineered HL7v2-to-FHIR translation services for electronic case reporting",
        "developed and certified interoperability conformance test suites",
    ),
    "data_modernization": (
        "executed a data modernization initiative retiring {n} legacy systems",
        "built cloud-native ELT pipelines replacing manual data calls",
        "delivered enterprise data platform services supporting {n} downstream programs",
    ),
    "ehr_integration": (
        "integrated {n} electronic health record systems for bidirectional data exchange",
        "built clinical decision support hooks into ambulatory EHR workflows",
        "delivered EHR data extraction and normalization for quality reporting",
    ),
    "cloud_migration": (
        "migrated {n} mission applications to a FedRAMP-authorized cloud environment",
        "designed and secured the cloud landing zone and CI/CD tooling",
        "achieved FedRAMP Moderate authorization for a public-facing platform",
    ),
    "cybersecurity": (
        "delivered continuous monitoring and ATO sustainment for {n} systems",
        "performed security control assessments and POA&M remediation",
        "implemented zero-trust network segmentation for health data systems",
    ),
    "immunization": (
        "provided immunization information system support across {n} state programs",
        "delivered vaccine coverage assessment and reminder-recall analytics",
        "supported provider enrollment and vaccine ordering operations for {n} sites",
    ),
    "vaccine_safety": (
        "supported post-licensure vaccine safety signal detection and analysis",
        "operated adverse event data intake and clinical review workflows",
        "conducted near-real-time sequential monitoring across {n} data partners",
    ),
    "hiv_sti": (
        "supported HIV prevention program planning across {n} jurisdictions",
        "delivered STI surveillance modernization and partner services analytics",
        "provided technical assistance to {n} Ryan White grantees",
    ),
    "maternal_child": (
        "supported maternal and child health block grant reporting for {n} states",
        "delivered home visiting program evaluation and performance measurement",
        "built maternal mortality review committee data infrastructure",
    ),
    "chronic_disease": (
        "delivered chronic disease prevention program support across {n} sites",
        "supported diabetes and cardiovascular risk reduction initiatives",
        "conducted evaluation of {n} community-based prevention programs",
    ),
    "behavioral_health": (
        "supported behavioral health data collection and analysis across {n} states",
        "delivered substance use treatment program evaluation and reporting",
        "provided technical assistance on opioid response program implementation",
    ),
    "health_equity": (
        "developed health equity measurement frameworks for {n} program areas",
        "conducted social determinants of health data linkage and analysis",
        "delivered equity-focused program assessments across {n} communities",
    ),
    "environmental_health": (
        "supported environmental public health tracking across {n} states",
        "delivered exposure assessment and toxicological data analysis",
        "built environmental hazard surveillance data products",
    ),
    "global_health": (
        "provided monitoring and evaluation support to PEPFAR programs in {n} countries",
        "delivered global health security capacity-building technical assistance",
        "supported international disease surveillance data systems",
    ),
    "emergency_preparedness": (
        "supported public health emergency preparedness planning for {n} jurisdictions",
        "delivered exercise design, execution, and after-action reporting",
        "provided readiness assessment and capability gap analysis",
    ),
    "medical_countermeasures": (
        "supported medical countermeasure distribution planning across {n} sites",
        "delivered Strategic National Stockpile inventory and logistics analytics",
        "provided countermeasure allocation modeling and decision support",
    ),
    "lab_systems": (
        "supported laboratory information management system deployment at {n} labs",
        "delivered biosafety and biosecurity program assessments",
        "provided laboratory data exchange and result reporting modernization",
    ),
    "program_evaluation": (
        "conducted mixed-methods evaluation of {n} federally funded programs",
        "designed performance measurement frameworks and logic models",
        "delivered annual program performance reporting and improvement analysis",
    ),
    "health_communications": (
        "developed and fielded health communication campaigns reaching {n} audiences",
        "conducted formative research and message testing",
        "delivered risk communication support during active response operations",
    ),
    "community_engagement": (
        "mobilized community partner networks across {n} jurisdictions",
        "facilitated stakeholder engagement and coalition building",
        "delivered community health needs assessments for {n} service areas",
    ),
    "workforce_training": (
        "designed and delivered training curricula to {n} public health professionals",
        "built competency frameworks and certification pathways",
        "operated a national training and technical assistance center",
    ),
    "health_policy": (
        "conducted policy analysis and cost modeling for {n} program options",
        "delivered regulatory impact analysis and stakeholder assessment",
        "produced economic evaluations of proposed coverage changes",
    ),
    "grants_management": (
        "provided grants and cooperative agreement administration for {n} awards",
        "delivered application review logistics and panel management",
        "supported post-award monitoring and compliance reporting",
    ),
    "clinical_quality": (
        "developed and maintained {n} electronic clinical quality measures",
        "supported quality measure testing, validation, and endorsement",
        "delivered provider performance reporting and feedback products",
    ),
}

OUTCOMES: tuple[str, ...] = (
    "Reduced data submission latency from {a} days to {b} days",
    "Increased reporting completeness from {a}% to {b}%",
    "Consolidated {a} legacy processes into {b} standardized workflows",
    "Delivered all {a} contract deliverables on schedule across {b} option years",
    "Expanded program reach from {a} to {b} participating jurisdictions",
    "Cut manual data reconciliation effort by {a}%",
    "Achieved {a}% stakeholder satisfaction on annual program survey",
    "Supported {a} data releases with zero reportable quality incidents",
    "Trained {a} staff across {b} partner organizations",
    "Improved measure validation throughput by {a}%",
)

# Opening clauses for opportunity scope text, keyed loosely by capability family.
OPPORTUNITY_FRAMING: dict[str, tuple[str, ...]] = {
    "epidemiology": (
        "The agency seeks contractor support to strengthen disease surveillance capacity",
        "This requirement covers epidemiologic analysis and outbreak analytic support",
    ),
    "health_it": (
        "The agency requires engineering support to modernize its data and exchange platforms",
        "This effort covers interoperability engineering and platform sustainment",
    ),
    "analytics": (
        "The agency seeks analytic and statistical support for population health measurement",
        "This requirement covers advanced analytics and modeling services",
    ),
    "program": (
        "The agency requires programmatic and technical assistance support",
        "This effort provides subject matter expertise and program implementation support",
    ),
    "preparedness": (
        "The agency seeks preparedness and response readiness support",
        "This requirement covers emergency response capability development",
    ),
    "services": (
        "The agency requires evaluation, communications, and administrative support",
        "This effort covers program support services and performance measurement",
    ),
}

# Titles are assembled as "{agency} {theme} {suffix}".
OPPORTUNITY_THEMES: dict[str, tuple[str, ...]] = {
    "epidemiology": ("Surveillance Systems", "Epidemiologic Analysis", "Disease Detection"),
    "health_it": ("Data Modernization", "Interoperability Services", "Platform Engineering"),
    "analytics": ("Analytic Services", "Population Health Analytics", "Statistical Support"),
    "program": ("Program Support", "Technical Assistance", "Implementation Support"),
    "preparedness": ("Preparedness Support", "Response Readiness", "Countermeasure Support"),
    "services": ("Evaluation Services", "Program Operations", "Communications Support"),
}

OPPORTUNITY_SUFFIX: tuple[str, ...] = (
    "BPA", "IDIQ", "Task Order", "Contract", "Support Services", "Blanket Purchase Agreement"
)

PLACES: tuple[str, ...] = (
    "Atlanta, GA",
    "Rockville, MD",
    "Bethesda, MD",
    "Baltimore, MD",
    "Washington, DC",
    "Research Triangle Park, NC",
    "Remote / Contractor Site",
    "Hybrid - Atlanta, GA",
    "Hybrid - Washington, DC",
    "Multi-site (national)",
)

DEGREES: tuple[str, ...] = ("MPH", "PhD", "MS", "MD MPH", "DrPH", "MSPH", "MBA")


# Activity phrases are written in past tense because most of their uses describe
# completed work. Opportunity scope needs the base form ("the contractor will
# design ..."), and deriving it by stripping suffixes would mangle the irregular
# verbs, so the mapping is explicit.
PAST_TO_BASE: dict[str, str] = {
    "achieved": "achieve",
    "built": "build",
    "conducted": "conduct",
    "delivered": "deliver",
    "designed": "design",
    "developed": "develop",
    "engineered": "engineer",
    "executed": "execute",
    "facilitated": "facilitate",
    "implemented": "implement",
    "integrated": "integrate",
    "led": "lead",
    "migrated": "migrate",
    "mobilized": "mobilize",
    "modernized": "modernize",
    "operated": "operate",
    "performed": "perform",
    "produced": "produce",
    "provided": "provide",
    "standardized": "standardize",
    "stood": "stand",
    "supported": "support",
}


def to_base(activity: str) -> str:
    """Rewrite a past-tense activity phrase into its base form."""
    head, _, rest = activity.partition(" ")
    base = PAST_TO_BASE.get(head.lower())
    return f"{base} {rest}" if base else activity
