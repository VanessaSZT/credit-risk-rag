from fpdf import FPDF
import os

def create_pdf(filename, title, content):
    class PDF(FPDF):
        def header(self):
            self.set_font('helvetica', 'B', 15)
            self.cell(w=0, h=10, text=title, border=0, new_x="LMARGIN", new_y="NEXT", align='C')
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('helvetica', 'I', 8)
            self.cell(w=0, h=10, text=f'Page {self.page_no()}', border=0, new_x="RMARGIN", new_y="TOP", align='C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font('helvetica', '', 11)
    pdf.multi_cell(w=0, h=6, text=content.strip())
    
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', filename)
    pdf.output(output_path)
    print(f"Generated: {filename}")

# Document 1: Commercial Real Estate
cre_content = """
1. Introduction
This document outlines the risk parameters for Commercial Real Estate (CRE) origination in Hong Kong.
Policy ID: HK-CRE-2024-V2. 

2. Loan-to-Value (LTV) and DSCR Matrix
The following dictates the maximum LTV and minimum Debt Service Coverage Ratio (DSCR) constraints by property type:
- Grade A Office (Central/Admiralty): Max 50% LTV, Min 1.20x DSCR
- Retail (Prime Districts): Max 40% LTV, Min 1.30x DSCR
- Industrial / Warehouse (Kwun Tong / Tsuen Wan): Max 40% LTV, Min 1.25x DSCR

Note: For properties valued over HKD 500 million, the Max LTV is automatically reduced by 10% from the baseline.

3. Stress Testing Requirements
A stress test must be applied to all commercial mortgage borrowers. The stress test assumes an interest rate increase of 200 basis points (2%). Under this stressed scenario, the borrower's stressed DSCR must not fall below 1.0x.

4. Guarantor Requirements
4.1 Non-Recourse Exceptions: Strictly limited to blue-chip developers listed on the HKEX with a market capitalization exceeding HKD 20 Billion.
4.2 Standard Recourse: For all other borrowers, full recourse personal guarantees are required from any director or shareholder holding a 25% or greater stake.
"""

# Document 2: SME Unsecured Lending
sme_content = """
1. Overview
This policy governs all unsecured term loans and revolving credit facilities provided to Small and Medium Enterprises (SMEs) operating in Hong Kong. Policy ID: SME-UNSEC-2024.

2. Eligibility Criteria
- Minimum years in business: 3 years of continuous operation in Hong Kong.
- Financials: 2 years of latest audited financial statements must be provided.
- The business must not be operating in restricted industries (e.g., Cryptocurrency, Unlicensed Money Service Operators).

3. Loan Limits and Tenor
- Maximum Loan Amount: HKD 5,000,000.
- Maximum Tenor: 60 months (5 years).
- Repayment: Standard monthly principal and interest. Interest-only periods are not permitted.

4. Guarantor Requirements
All SME unsecured facilities MUST be supported by a personal guarantee from the primary business owners. Joint and several guarantees are required if multiple directors hold a combined stake of more than 50%.
"""

# Document 3: AML and KYC
aml_content = """
1. Scope and Applicability
This manual outlines the Anti-Money Laundering (AML) and Know Your Customer (KYC) compliance standards for onboarding and maintaining corporate banking relationships. Policy ID: AML-KYC-2024.

2. Ultimate Beneficial Owner (UBO) Identification
- Standard Risk Corporate Clients: We must identify and verify the identity of any individual holding a 25% or greater ownership stake.
- High-Risk Corporate Clients (e.g., offshore jurisdictions, cash-intensive businesses): We must identify and verify the identity of any individual holding a 10% or greater ownership stake.

3. Politically Exposed Persons (PEPs)
A PEP is defined as an individual who is or has been entrusted with a prominent public function. 
- Enhanced Due Diligence (EDD) is strictly required for any corporate entity where a Director, UBO, or Senior Executive is identified as a PEP.
- Onboarding or lending to a PEP-associated entity requires explicit written approval from the Head of Financial Crime Compliance (FCC) and the Chief Risk Officer (CRO).

4. Transaction Monitoring
Any unexpected transaction exceeding HKD 1,000,000 that does not align with the client's declared business profile will trigger an automated Suspicious Activity Report (SAR) flag.
"""

create_pdf('HK_Commercial_Real_Estate_Policy_2024.pdf', 'HK Commercial Lending Policy - 2024', cre_content)
create_pdf('SME_Unsecured_Lending_Guidelines.pdf', 'SME Unsecured Lending Guidelines', sme_content)
create_pdf('AML_KYC_Compliance_Manual.pdf', 'AML and KYC Compliance Manual', aml_content)
