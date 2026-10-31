"""Clinical Guideline Generator and Downloader for PharmGuard.

Generates comprehensive, multi-page clinical guideline PDFs with structured
tables and clinical decision thresholds for:
  1. Renal eGFR Drug Dosing & CKD Adjustment Guidelines
  2. CYP3A4 Drug-Drug Interactions & Pharmacokinetic Management
  3. Serotonin Syndrome Diagnosis, Washout Protocols & Triage Guidelines
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("guideline_downloader")

GUIDELINES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "guidelines"


def create_styles():
    """Create custom ReportLab typography styles."""
    styles = getSampleStyleSheet()

    # Title Style
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6,
    )

    # Subtitle Style
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceAfter=12,
    )

    # Heading 1
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=12,
        spaceAfter=6,
    )

    # Heading 2
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
        spaceBefore=8,
        spaceAfter=4,
    )

    # Body Text
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1A202C"),
        spaceAfter=6,
    )

    # Callout / Alert Box Text
    alert_style = ParagraphStyle(
        "AlertText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#9B2C2C"),
    )

    # Table Header Text
    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=0,
    )

    # Table Cell Text
    td_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#2D3748"),
    )

    # Table Cell Bold
    td_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1A202C"),
    )

    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "h1": h1_style,
        "h2": h2_style,
        "body": body_style,
        "alert": alert_style,
        "th": th_style,
        "td": td_style,
        "td_bold": td_bold,
    }


def build_table(header_row: List[str], data_rows: List[List[str]], col_widths: List[float], s: Dict[str, ParagraphStyle]) -> Table:
    """Helper to construct styled ReportLab tables with wrapped Paragraphs."""
    table_data = []
    
    # Process header
    h_cells = [Paragraph(f"<b>{col}</b>", s["th"]) for col in header_row]
    table_data.append(h_cells)

    # Process rows
    for row in data_rows:
        r_cells = []
        for idx, cell in enumerate(row):
            # Bold first column or alert indicators
            if idx == 0 or "CONTRAINDICATED" in cell or "Severe" in cell or "Avoid" in cell:
                r_cells.append(Paragraph(cell, s["td_bold"]))
            else:
                r_cells.append(Paragraph(cell, s["td"]))
        table_data.append(r_cells)

    t = Table(table_data, colWidths=col_widths)
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F7FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ])
    )
    return t


def generate_renal_guideline(file_path: Path):
    """Generate multi-page Renal Impairment & eGFR Dosing Thresholds guideline."""
    doc = SimpleDocTemplate(
        str(file_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    s = create_styles()
    story = []

    # Title & Metadata
    story.append(Paragraph("KDIGO & Clinical Pharmacokinetics Guideline", s["subtitle"]))
    story.append(Paragraph("Renal Impairment & eGFR-Based Drug Dosing Thresholds", s["title"]))
    story.append(Paragraph("<b>Reference Code:</b> CPG-RENAL-2026.01 | <b>Target Audience:</b> Clinical Pharmacists, Intensivists, Prescribers", s["body"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Clinical Scope", s["h1"]))
    summary_text = (
        "Impaired glomerular filtration significantly diminishes the renal clearance of active pharmaceuticals and "
        "their toxic active metabolites. Inappropriate dosing in chronic kidney disease (CKD) stages G3 through G5 "
        "represents a premier contributor to preventable in-hospital adverse drug events (ADEs), including lactic acidosis, "
        "hyperkalemia, neurotoxicity, severe hemorrhagic events, and irreversible myelosuppression. "
        "This clinical guideline establishes evidence-based eGFR and CrCl thresholds for critical narrow-therapeutic-index drugs."
    )
    story.append(Paragraph(summary_text, s["body"]))

    # Table 1: CKD Staging Classification
    story.append(Paragraph("2. Chronic Kidney Disease (CKD) Staging Reference", s["h1"]))
    ckd_headers = ["CKD Stage", "eGFR Range (mL/min/1.73m²)", "Functional Category", "Pharmacokinetic Action Required"]
    ckd_rows = [
        ["Stage G1", "≥ 90", "Normal or high", "Standard adult dosing; monitor for nephrotoxic agents."],
        ["Stage G2", "60 – 89", "Mildly decreased", "Routine monitoring; baseline assessment for renal-eliminated drugs."],
        ["Stage G3a", "45 – 59", "Mild-to-moderate decrease", "Initiate precautionary dosage adjustments; review SGLT2i & Metformin."],
        ["Stage G3b", "30 – 44", "Moderate-to-severe decrease", "Active dose reduction required for Metformin, DOACs, Methotrexate, and Colchicine."],
        ["Stage G4", "15 – 29", "Severely decreased", "Strict contraindication or 50-75% dose reduction; high risk of metabolite accumulation."],
        ["Stage G5", "< 15 (or Dialysis)", "Kidney failure / ESRD", "Specialized dialysis replacement dosing; multiple drug classes strictly contraindicated."],
    ]
    story.append(build_table(ckd_headers, ckd_rows, [70, 110, 140, 220], s))
    story.append(Spacer(1, 10))

    # Table 2: High-Risk Renal Dosing Thresholds
    story.append(Paragraph("3. High-Risk Drug Dosing Thresholds & Adjustment Protocols", s["h1"]))
    drug_headers = ["Drug Name", "Standard Dose", "eGFR / CrCl Threshold", "Required Clinical Action & Risk Rationale"]
    drug_rows = [
        [
            "Metformin",
            "500–1000 mg BID",
            "eGFR 30–44 mL/min<br/>eGFR &lt; 30 mL/min",
            "eGFR 30-44: Max 1000 mg/day total; do NOT initiate in new patients.<br/><b>CONTRAINDICATED if eGFR &lt; 30</b>: High risk of fatal Lactic Acidosis due to accumulation.",
        ],
        [
            "Apixaban",
            "5 mg BID",
            "Serum Cr ≥ 1.5 mg/dL<br/>with Age ≥80 or Wt ≤60kg",
            "Reduce dose to <b>2.5 mg BID</b> if any 2 ABC criteria met (Age ≥80, Body Wt ≤60 kg, Serum Cr ≥1.5 mg/dL). Monitor for major bleeding.",
        ],
        [
            "Spironolactone",
            "25–50 mg daily",
            "eGFR &lt; 30 mL/min or<br/>K+ &gt; 5.0 mEq/L",
            "<b>CONTRAINDICATED if eGFR &lt; 30</b> or Serum K+ &gt; 5.0 mEq/L. Risk of life-threatening hyperkalemia, fatal cardiac dysrhythmias.",
        ],
        [
            "Methotrexate",
            "7.5–25 mg weekly",
            "eGFR 30–50 mL/min<br/>eGFR &lt; 30 mL/min",
            "eGFR 30-50: Reduce weekly dose by 50%.<br/><b>CONTRAINDICATED if eGFR &lt; 30</b>: Delayed elimination causes severe bone marrow suppression and pancytopenia.",
        ],
        [
            "Colchicine",
            "0.6 mg daily-BID",
            "eGFR 30–50 mL/min<br/>eGFR &lt; 30 mL/min",
            "eGFR 30-50: Max 0.3 mg daily. eGFR &lt; 30: 0.3 mg every 2-3 days.<br/><b>CONTRAINDICATED with CYP3A4/P-gp inhibitors in renal impairment</b> (fatal neuromyopathy).",
        ],
        [
            "Empagliflozin",
            "10–25 mg daily",
            "eGFR &lt; 30 mL/min<br/>eGFR &lt; 20 mL/min",
            "Glycemic efficacy severely blunted if eGFR &lt; 30. Discontinue if eGFR &lt; 20 mL/min or if initiating maintenance dialysis.",
        ],
        [
            "Lisinopril",
            "10–40 mg daily",
            "eGFR 10–30 mL/min<br/>eGFR &lt; 10 mL/min",
            "Initial dose 2.5–5 mg daily; titrate cautiously. Monitor serum creatinine and potassium within 7-14 days. Stop if serum Cr jumps &gt; 30%.",
        ],
        [
            "Allopurinol",
            "100–300 mg daily",
            "eGFR 15–29 mL/min<br/>eGFR &lt; 15 mL/min",
            "Initial dose 50-100 mg daily; titrate by 50 mg increments. Toxic metabolite oxypurinol accumulates, provoking Allopurinol Hypersensitivity Syndrome (AHS).",
        ],
    ]
    story.append(build_table(drug_headers, drug_rows, [85, 80, 115, 260], s))
    
    story.append(PageBreak())

    # Page 2: Clinical Monitoring, Equations, and Black Box Warnings
    story.append(Paragraph("4. Pharmacokinetic Monitoring & Clinical Decision Matrix", s["h1"]))
    
    monitoring_text = (
        "<b>Renal Function Equation Discrepancies:</b><br/>"
        "1. <b>Cockcroft-Gault CrCl:</b> The historical benchmark utilized by the FDA for package insert dosing labels. "
        "Use actual body weight unless patient is obese (BMI ≥ 30 kg/m²), in which case Adjusted Body Weight (ABW) is recommended.<br/>"
        "2. <b>CKD-EPI 2021 (eGFR):</b> Preferred for staging renal disease without race variables. "
        "For narrow-therapeutic-index drugs (e.g., Colchicine, Digoxin, Methotrexate), calculate absolute CrCl in mL/min unindexed for BSA.<br/><br/>"
        "<b>High-Risk Dialysis Caveats:</b><br/>"
        "In intermittent hemodialysis (IHD), drugs with low volume of distribution (Vd &lt; 1 L/kg) and low plasma protein binding "
        "are substantially dialyzable and require post-hemodialysis supplemental administration."
    )
    story.append(Paragraph(monitoring_text, s["body"]))
    story.append(Spacer(1, 10))

    # Renal Interaction Warning Table
    story.append(Paragraph("5. Critical Triple Whammy & Nephrotoxic Synergies", s["h1"]))
    synergy_headers = ["Combination", "Mechanisms of Nephrotoxicity", "Clinical Consequence", "Mandatory Action"]
    synergy_rows = [
        [
            "ACE-I / ARB + Diuretic + NSAID ('Triple Whammy')",
            "Diuretic causes hypovolemia; ACE-I blocks angiotensin II efferent arteriolar constriction; NSAID blocks prostaglandin afferent arteriolar dilation.",
            "Precipitous drop in glomerular hydrostatic filtration pressure, resulting in Acute Kidney Injury (AKI) in up to 30% of elderly patients.",
            "Avoid concurrent NSAID therapy. If analgesia required, substitute with acetaminophen or topical formulations; monitor renal labs at 48 hours.",
        ],
        [
            "Calcineurin Inhibitors (Tacrolimus/Cyclosporine) + Ciprofloxacin",
            "Ciprofloxacin impairs CYP3A4-mediated hepatic/gut metabolism of tacrolimus while producing additive tubular crystalluria.",
            "Sharp increase in trough blood levels (Tacrolimus &gt; 15 ng/mL), inducing acute afferent arteriolar vasospasm and severe nephrotoxicity.",
            "Perform daily Therapeutic Drug Monitoring (TDM); proactively decrease calcineurin inhibitor dosage by 25-50% during fluoroquinolone therapy.",
        ],
    ]
    story.append(build_table(synergy_headers, synergy_rows, [110, 140, 150, 140], s))
    
    doc.build(story)
    logger.info("Successfully generated Renal eGFR Guideline PDF: %s", file_path)


def generate_cyp3a4_guideline(file_path: Path):
    """Generate multi-page CYP3A4 Drug Interactions & Pharmacokinetics guideline."""
    doc = SimpleDocTemplate(
        str(file_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    s = create_styles()
    story = []

    # Title & Metadata
    story.append(Paragraph("FDA & CPIC Clinical Practice Guideline", s["subtitle"]))
    story.append(Paragraph("Cytochrome P450 3A4 (CYP3A4) Drug-Drug Interactions & Management", s["title"]))
    story.append(Paragraph("<b>Reference Code:</b> CPG-CYP3A4-2026.02 | <b>Target Audience:</b> Pharmacologists, Prescribers, Oncology Specialists", s["body"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # Executive Summary
    story.append(Paragraph("1. Mechanism & Pharmacologic Principles", s["h1"]))
    mech_text = (
        "Cytochrome P450 3A4 (CYP3A4) is the predominant phase I metabolic enzyme in human enterocytes and hepatocytes, "
        "responsible for biotransforming more than 50% of clinically prescribed medications. "
        "Inhibition of CYP3A4 leads to catastrophic drug accumulation, dose-dependent organ toxicity, and fatal arrhythmias (such as Torsades de Pointes). "
        "Conversely, potent CYP3A4 induction reduces plasma concentrations below the minimum effective therapeutic threshold, leading to treatment failures, "
        "transplant rejections, and thrombotic events."
    )
    story.append(Paragraph(mech_text, s["body"]))

    # Table 1: Classification of Inhibitors and Inducers
    story.append(Paragraph("2. FDA Classification of CYP3A4 Modulators", s["h1"]))
    class_headers = ["Category", "Definition (AUC Change)", "Key Pharmaceutical Agents", "Clinical Significance"]
    class_rows = [
        [
            "Strong Inhibitors",
            "≥ 5-fold increase in substrate AUC<br/>or &gt; 80% clearance decrease",
            "Clarithromycin, Ketoconazole, Itraconazole, Voriconazole, Ritonavir (including Paxlovid), Idelalisib, Nefazodone",
            "Severe toxicity with narrow index substrates. Co-administration often CONTRAINDICATED.",
        ],
        [
            "Moderate Inhibitors",
            "≥ 2 to &lt; 5-fold increase in AUC<br/>or 50–80% clearance decrease",
            "Fluconazole, Diltiazem, Verapamil, Ciprofloxacin, Erythromycin, Grapefruit juice (&gt;200 mL)",
            "Dose reductions and intensive monitoring required for statins, anticoagulants, and sedatives.",
        ],
        [
            "Strong Inducers",
            "≥ 80% decrease in substrate AUC",
            "Rifampin, Carbamazepine, Phenytoin, Enzalutamide, Mitotane, St. John's Wort (Hypericum perforatum)",
            "Causes therapeutic failure. Alternative non-CYP3A4 antimicrobial or anticonvulsant must be chosen.",
        ],
        [
            "Moderate Inducers",
            "50% to &lt; 80% decrease in substrate AUC",
            "Efavirenz, Bosentan, Modafinil, Phenobarbital, Rifabutin",
            "Compromises clinical efficacy of oral contraceptives, DOACs, and immunosuppressants.",
        ],
    ]
    story.append(build_table(class_headers, class_rows, [85, 115, 170, 170], s))
    story.append(Spacer(1, 10))

    # Table 2: Critical Substrate Drug Interactions
    story.append(Paragraph("3. High-Risk Substrate Drug Interactions & Clinical Protocols", s["h1"]))
    sub_headers = ["Substrate Drug", "Interacting Agent", "Pathophysiologic Outcome", "Management & Dosing Action"]
    sub_rows = [
        [
            "Paxlovid<br/>(Nirmatrelvir/Ritonavir)",
            "Amiodarone / Digoxin",
            "Ritonavir is an ultra-potent irreversible inhibitor of CYP3A4 and P-gp. Precipitates fatal cardiac arrhythmias and digoxin toxicity.",
            "<b>CONTRAINDICATED</b>. Amiodarone has a months-long half-life; Paxlovid cannot be safely administered even if amiodarone recently stopped.",
        ],
        [
            "Atorvastatin / Simvastatin",
            "Clarithromycin / Itraconazole / Paxlovid",
            "Plasma statin concentration surges 4-fold to 12-fold, precipitating acute skeletal muscle breakdown (Rhabdomyolysis) and myoglobinuric renal failure.",
            "<b>Simvastatin: CONTRAINDICATED</b>.<br/>Atorvastatin: Suspend statin during short-course therapy or cap dose at 10–20 mg. Switch to Pravastatin or Rosuvastatin.",
        ],
        [
            "Apixaban / Rivaroxaban",
            "Rifampin (Strong Inducer)",
            "Rifampin induces hepatic CYP3A4 and intestinal P-gp, decreasing DOAC AUC by 50-60%, precipitating stroke or systemic embolism.",
            "<b>AVOID COMBINATION</b>. DOAC efficacy is critically compromised. Switch to unfractionated or low-molecular-weight heparin.",
        ],
        [
            "Cyclosporine / Tacrolimus",
            "Fluconazole / Clarithromycin",
            "Marked suppression of first-pass and systemic clearance. Calcineurin inhibitor trough levels multiply 2- to 4-fold.",
            "Proactively reduce immunosuppressant dose by 50–75%. Initiate daily trough TDM; monitor serum creatinine closely.",
        ],
        [
            "Colchicine",
            "Clarithromycin / Ritonavir (in Renal Impairment)",
            "Severe multi-organ toxicity: marrow aplasia, neuromyopathy, rhabdomyolysis, and fatal circulatory collapse.",
            "<b>STRICTLY CONTRAINDICATED in patients with renal or hepatic impairment</b>. In normal renal function, reduce dose by 75%.",
        ],
        [
            "Tramadol / Oxycodone",
            "Fluconazole / Ciprofloxacin",
            "CYP3A4 inhibition shifts metabolism toward accumulation of parent drug or altered active metabolite balance.",
            "Elevated risk of profound central nervous system and respiratory depression. Titrate downward; ensure Naloxone access.",
        ],
    ]
    story.append(build_table(sub_headers, sub_rows, [90, 100, 175, 175], s))

    story.append(PageBreak())

    # Page 2: Clinical Management Algorithm & Washout Windows
    story.append(Paragraph("4. CYP3A4 Induction & Inhibition Time Courses", s["h1"]))
    time_text = (
        "<b>Inhibition Kinetics:</b> Competitive inhibition (e.g., Fluconazole) occurs almost immediately with peak plasma drug levels. "
        "Mechanism-based (irreversible/suicide) inhibition (e.g., Clarithromycin, Ritonavir, Grapefruit juice) destroys the enzyme; "
        "full CYP3A4 recovery requires de novo enzyme synthesis taking <b>3 to 7 days</b> after discontinuing the offending inhibitor.<br/><br/>"
        "<b>Induction Kinetics:</b> Transcriptional activation mediated by PXR (Pregnane X Receptor) takes <b>7 to 14 days</b> to reach maximum "
        "enzymatic expression following initiation of Rifampin or Carbamazepine. Similarly, enzymatic de-induction persists for <b>2 to 3 weeks</b> "
        "after stopping the inducer."
    )
    story.append(Paragraph(time_text, s["body"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("5. Step-by-Step Clinical Decision Flowchart for Prescribers", s["h1"]))
    flow_headers = ["Step", "Clinical Assessment", "Required Verification", "Safe Alternative Strategy"]
    flow_rows = [
        ["Step 1", "Identify CYP3A4 Dependency", "Is the prescribed agent cleared &gt;50% via hepatic or intestinal CYP3A4?", "If yes, review current medication list for moderate/strong inhibitors or inducers."],
        ["Step 2", "Assess Therapeutic Window", "Is the substrate a narrow-therapeutic-index (NTI) drug (e.g., Tacrolimus, Digoxin, Apixaban)?", "If NTI, co-administration with strong inhibitors is generally contraindicated."],
        ["Step 3", "Select Pharmacologic Alternative", "Can an alternative anti-infective or cardiovascular agent be used?", "E.g., Azithromycin instead of Clarithromycin; Rosuvastatin instead of Simvastatin."],
        ["Step 4", "Execute Preemptive Dose Titration", "If co-administration is clinically unavoidable, reduce dose prior to starting interacting agent.", "Schedule laboratory monitoring (trough drug level, renal panel, LFTs, ECG) within 48-72 hours."],
    ]
    story.append(build_table(flow_headers, flow_rows, [50, 120, 190, 180], s))

    doc.build(story)
    logger.info("Successfully generated CYP3A4 Guideline PDF: %s", file_path)


def generate_serotonin_guideline(file_path: Path):
    """Generate multi-page Serotonin Syndrome Diagnosis & Prevention guideline."""
    doc = SimpleDocTemplate(
        str(file_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    s = create_styles()
    story = []

    # Title & Metadata
    story.append(Paragraph("Toxicology & Clinical Pharmacology Consensus", s["subtitle"]))
    story.append(Paragraph("Serotonin Syndrome: Clinical Diagnosis, Washout Protocols & Triage", s["title"]))
    story.append(Paragraph("<b>Reference Code:</b> CPG-SEROTONIN-2026.03 | <b>Target Audience:</b> Psychiatrists, Emergency Physicians, Clinical Pharmacists", s["body"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # Executive Summary
    story.append(Paragraph("1. Pathophysiology & Clinical Triad", s["h1"]))
    patho_text = (
        "Serotonin Syndrome (serotonin toxicity) is a life-threatening, drug-induced toxidrome caused by excessive agonism "
        "at central and peripheral serotonergic receptors (specifically 5-HT1A and 5-HT2A). "
        "The condition is characterized by a classical triad of: "
        "<b>(1) Neuromuscular Hyperactivity</b> (clonus, hyperreflexia, myoclonus, rigidity); "
        "<b>(2) Autonomic Hyperactivity</b> (diaphoresis, hyperthermia, tachycardia, tachypnea, labile BP); and "
        "<b>(3) Mental Status Alterations</b> (agitation, delirium, confusion, coma). "
        "Unlike Neuroleptic Malignant Syndrome (NMS), serotonin toxicity develops acutely within hours of drug ingestion or dose escalation."
    )
    story.append(Paragraph(patho_text, s["body"]))

    # Table 1: Hunter Serotonin Toxicity Criteria
    story.append(Paragraph("2. Hunter Toxicity Criteria (Gold Standard Diagnostic Framework)", s["h1"]))
    hunter_intro = "In the presence of a known serotonergic agent, Serotonin Syndrome is definitively diagnosed if ANY of the following rules are met:"
    story.append(Paragraph(hunter_intro, s["body"]))
    
    hunter_headers = ["Rule #", "Mandatory Clinical Sign", "Associated Symptoms Required", "Diagnostic Sensitivity / Specificity"]
    hunter_rows = [
        ["Criterion 1", "Spontaneous Clonus", "None required (sufficient on its own)", "Sensitivity 84%, Specificity 97%"],
        ["Criterion 2", "Inducible Clonus", "PLUS Agitation OR Diaphoresis", "High diagnostic specificity"],
        ["Criterion 3", "Ocular Clonus (slow horizontal)", "PLUS Agitation OR Diaphoresis", "Early sensitive neurological indicator"],
        ["Criterion 4", "Tremor", "PLUS Hyperreflexia (especially in lower extremities)", "Common presentation in mild-to-moderate toxicity"],
        ["Criterion 5", "Hypertonia", "PLUS Core Temperature &gt; 38.0°C (100.4°F) PLUS Ocular/Inducible Clonus", "Indicator of severe, immediately life-threatening toxicity"],
    ]
    story.append(build_table(hunter_headers, hunter_rows, [65, 125, 200, 150], s))
    story.append(Spacer(1, 10))

    # Table 2: High-Risk Serotonergic Agents & Combinations
    story.append(Paragraph("3. High-Risk Drug Combinations & Absolute Contraindications", s["h1"]))
    ser_headers = ["Drug Class 1", "Drug Class 2", "Mechanism of Synergy", "Clinical Severity & Mandated Action"]
    ser_rows = [
        [
            "MAO Inhibitors<br/>(Phenelzine, Tranylcypromine, Linezolid, Methylene Blue)",
            "SSRIs / SNRIs<br/>(Sertraline, Fluoxetine, Venlafaxine)",
            "Simultaneous blockade of serotonin degradation (MAO-A) and reuptake inhibition causes massive, catastrophic synaptic 5-HT surge.",
            "<b>ABSOLUTELY CONTRAINDICATED</b>.<br/>High risk of fatal hyperthermia (&gt;41°C), rhabdomyolysis, DIC, and death. Strict washout required.",
        ],
        [
            "SSRIs / SNRIs<br/>(Sertraline, Venlafaxine)",
            "Tramadol",
            "Tramadol is both a μ-opioid agonist and a serotonin/norepinephrine reuptake inhibitor (SNRI). Causes severe additive toxicity.",
            "<b>AVOID COMBINATION</b>. Substitute with non-serotonergic analgesics (e.g., Oxycodone, Morphine, Acetaminophen).",
        ],
        [
            "SSRIs / SNRIs / Tricyclics",
            "Triptans<br/>(Sumatriptan, Zolmitriptan)",
            "Triptans act as selective 5-HT1B/1D agonists; co-administration with reuptake inhibitors enhances peripheral & central receptor activation.",
            "FDA Black Box Advisory. Inform patients of clonus/agitation warning signs. Limit triptan frequency.",
        ],
        [
            "SSRIs / SNRIs",
            "Lithium",
            "Lithium enhances presynaptic release and postsynaptic sensitivity of central serotonergic neurons.",
            "Use lowest possible doses. Monitor for tremor, ataxia, confusion, and clonus during any titration.",
        ],
        [
            "Sertraline / Fluoxetine",
            "Bupropion",
            "Bupropion is a potent CYP2D6 inhibitor, raising SSRI/SNRI plasma concentrations up to 4-fold while lowering seizure threshold.",
            "Monitor antidepressant levels; reduce SSRI dose by 50% when initiating bupropion.",
        ],
    ]
    story.append(build_table(ser_headers, ser_rows, [110, 100, 160, 170], s))

    story.append(PageBreak())

    # Page 2: Mandatory Washout Periods & Emergency Management
    story.append(Paragraph("4. Mandatory Antidepressant Washout Periods", s["h1"]))
    washout_text = (
        "Switching between serotonergic antidepressants requires strict adherence to pharmacokinetic washout intervals "
        "calculated based on 5 elimination half-lives (t½) of the parent drug and active metabolites:"
    )
    story.append(Paragraph(washout_text, s["body"]))
    story.append(Spacer(1, 6))

    wash_headers = ["Initial Agent Discontinued", "Elimination Half-Life", "Mandatory Washout Before MAOI", "Clinical Rationale"]
    wash_rows = [
        ["Fluoxetine (Prozac)", "Parent: 2–4 days<br/>Active norfluoxetine: <b>7–15 days</b>", "<b>5 WEEKS (35 Days)</b>", "Norfluoxetine metabolite persists for over a month. Premature MAOI initiation has resulted in fatalities."],
        ["Sertraline (Zoloft)", "Parent: 26 hours<br/>Active N-desmethylsertraline: 66 hrs", "<b>2 WEEKS (14 Days)</b>", "Safe clearance achieved after 14 days."],
        ["Venlafaxine (Effexor)", "Parent: 5 hours<br/>O-desmethylvenlafaxine: 11 hrs", "<b>7 to 14 Days</b>", "Short half-life, but severe discontinuation syndrome requires gradual taper."],
        ["MAOIs (Phenelzine)", "Irreversible enzyme inactivation", "<b>2 WEEKS (14 Days)</b>", "Requires 14 days for regeneration of functional endogenous MAO enzymes before starting SSRI."],
    ]
    story.append(build_table(wash_headers, wash_rows, [110, 110, 120, 200], s))
    story.append(Spacer(1, 10))

    # Table: Triage and Treatment Algorithm
    story.append(Paragraph("5. Emergency Triage & Management Protocol", s["h1"]))
    triage_headers = ["Severity Grade", "Presenting Signs", "Pharmacologic Interventions", "Level of Care"]
    triage_rows = [
        [
            "Mild",
            "Hyperreflexia, mild tremor, diaphoresis, normal temp, alert",
            "Discontinue all serotonergic agents. Oral Diazepam 5-10 mg for restlessness. Reassurance and observation.",
            "Outpatient observation or 24-hr Clinical Decision Unit (CDU).",
        ],
        [
            "Moderate",
            "Inducible clonus, tachycardia, mydriasis, temp 38.0–38.9°C, agitation",
            "IV fluids, IV Lorazepam 2-4 mg. <b>Cyproheptadine: 12 mg oral initial loading dose</b>, then 2 mg every 2 hours if symptoms persist (max 32 mg/day).",
            "Telemetry or Step-Down Inpatient Unit.",
        ],
        [
            "Severe (Life Threatening)",
            "Spontaneous clonus, rigidity, temp &gt; 39.0°C (102.2°F), delirium, metabolic acidosis",
            "Immediate RSI endotracheal intubation. Neuromuscular paralysis with Vecuronium (avoid succinylcholine). Aggressive active external cooling.",
            "Intensive Care Unit (ICU) admission.",
        ],
    ]
    story.append(build_table(triage_headers, triage_rows, [70, 130, 210, 130], s))

    doc.build(story)
    logger.info("Successfully generated Serotonin Syndrome Guideline PDF: %s", file_path)


def download_or_generate_guidelines(output_dir: Path = GUIDELINES_DIR) -> List[Path]:
    """Ensure clinical guideline PDFs exist in data/raw/guidelines/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    guideline_files = [
        output_dir / "renal_egfr_dosing_guidelines.pdf",
        output_dir / "cyp3a4_drug_interactions_guideline.pdf",
        output_dir / "serotonin_syndrome_clinical_guideline.pdf",
    ]

    logger.info("Generating comprehensive clinical guidelines into %s...", output_dir)
    generate_renal_guideline(guideline_files[0])
    generate_cyp3a4_guideline(guideline_files[1])
    generate_serotonin_guideline(guideline_files[2])

    logger.info("Successfully created all %d clinical guideline PDFs.", len(guideline_files))
    return guideline_files


if __name__ == "__main__":
    download_or_generate_guidelines()
