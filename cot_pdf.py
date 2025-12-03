from fpdf import FPDF
import datetime

class ResearchPaperPDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 15)
        self.cell(0, 10, 'ANALYSE HEBDOMADAIRE: COMMITMENTS OF TRADERS (COT)', 0, 1, 'C')
        self.set_font('Times', 'I', 10)
        self.cell(0, 10, 'CME Group Data - Rapport du 21 Octobre 2025', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Times', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Times', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()

    def add_data_table(self, instrument, non_comm_long, non_comm_short, change_long, change_short, interpretation):
        self.set_font('Times', 'B', 11)
        self.cell(0, 8, f"Instrument: {instrument}", 0, 1)
        
        self.set_font('Courier', '', 9)
        # Headers
        self.cell(45, 6, "Category", 1)
        self.cell(35, 6, "Longs", 1)
        self.cell(35, 6, "Shorts", 1)
        self.cell(35, 6, "Net Position", 1)
        self.ln()
        
        # Data
        net = non_comm_long - non_comm_short
        self.cell(45, 6, "Non-Commercial (Specs)", 1)
        self.cell(35, 6, f"{non_comm_long:,}", 1)
        self.cell(35, 6, f"{non_comm_short:,}", 1)
        self.cell(35, 6, f"{net:,}", 1)
        self.ln()
        
        # Changes
        self.cell(45, 6, "Weekly Change", 1)
        self.cell(35, 6, f"{change_long:+}", 1)
        self.cell(35, 6, f"{change_short:+}", 1)
        self.cell(35, 6, "-", 1)
        self.ln(8)
        
        self.set_font('Times', 'I', 11)
        self.multi_cell(0, 5, f"Interpretation: {interpretation}")
        self.ln(8)

# Create PDF
pdf = ResearchPaperPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Abstract
pdf.set_font('Times', 'B', 12)
pdf.cell(0, 10, 'RESUME EXECUTIF', 0, 1, 'C')
pdf.set_font('Times', '', 11)
abstract_text = (
    "Ce rapport présente une analyse quantitative et qualitative des positions des traders "
    "sur les contrats à terme du Chicago Mercantile Exchange (CME) pour la semaine du 21 octobre 2025. "
    "L'accent est mis sur la divergence entre les positions 'Non-Commercial' (Spéculateurs/Fonds) "
    "et 'Commercial' (Hedgers) sur les indices boursiers, les devises majeures et les actifs numériques. "
    "Les données révèlent un positionnement défensif sur les actions américaines et un retournement haussier "
    "significatif sur le Yen japonais."
)
pdf.multi_cell(0, 6, abstract_text)
pdf.ln(5)
pdf.line(10, pdf.get_y(), 200, pdf.get_y())
pdf.ln(10)

# 1. Equities
pdf.chapter_title('1. MARCHES ACTIONS : S&P 500 (E-MINI)')
body_sp500 = (
    "L'analyse du contrat E-Mini S&P 500 montre un positionnement net vendeur (Net Short) de la part des "
    "spéculateurs. Alors que le marché reste à des niveaux élevés, les 'Non-Commercials' maintiennent "
    "plus de positions courtes que longues, suggérant une anticipation de correction ou une couverture "
    "massive de portefeuilles existants."
)
pdf.chapter_body(body_sp500)
pdf.add_data_table(
    "S&P 500 E-MINI (Code-13874A)", 
    222837, 368187, 
    9011, -11322, 
    "Sentiment Bearish (Baissier). Les spéculateurs sont Net Short (-145k contrats). "
    "Cependant, on note une légère fermeture de positions courtes (-11k) cette semaine, "
    "ce qui pourrait indiquer une prise de profit partielle sur ces paris baissiers."
)

# 2. Forex
pdf.chapter_title('2. FOREX : LE RETOUR DU YEN')
body_fx = (
    "Le marché des changes présente la dynamique la plus forte de cette semaine. "
    "Le Yen Japonais (JPY) connait un changement structurel de positionnement."
)
pdf.chapter_body(body_fx)

# JPY Data
pdf.add_data_table(
    "JAPANESE YEN (Code-097741)", 
    175724, 105310, 
    15085, -18163, 
    "Sentiment Bullish (Haussier). Changement drastique. Les 'Large Specs' ont ajouté 15k positions longues "
    "et liquidé 18k positions courtes. C'est un signal clair de 'Short Squeeze' ou de changement de tendance "
    "fondamental anticipé par les fonds."
)

# EUR Data
pdf.add_data_table(
    "EURO FX (Code-099741)", 
    244507, 132755, 
    1497, -1930, 
    "Sentiment Neutre/Haussier. Le positionnement reste Net Long, mais les variations sont faibles. "
    "Le consensus haussier sur l'Euro semble s'essouffler comparé à la dynamique du Yen."
)

# 3. Crypto
pdf.chapter_title('3. ACTIFS NUMERIQUES : BITCOIN')
body_crypto = (
    "Le positionnement sur les contrats à terme Bitcoin standard (5 BTC) montre une indécision totale "
    "de la part des institutionnels."
)
pdf.chapter_body(body_crypto)
pdf.add_data_table(
    "BITCOIN (Code-133741)", 
    23426, 23755, 
    -466, -589, 
    "Sentiment Neutre. Les positions Long et Short sont presque identiques (23k vs 23k). "
    "Les changements hebdomadaires sont marginaux. Le marché attend un catalyseur externe pour choisir une direction."
)

# Conclusion
pdf.line(10, pdf.get_y(), 200, pdf.get_y())
pdf.ln(5)
pdf.chapter_title('CONCLUSION ET PERSPECTIVES')
conclusion = (
    "En conclusion, le rapport du 21 octobre 2025 met en évidence une prudence institutionnelle. "
    "Le positionnement sur le S&P 500 est sceptique quant à la poursuite de la hausse sans correction. "
    "L'opportunité tactique semble se situer sur le Yen Japonais, qui bénéficie de flux acheteurs massifs. "
    "Le Bitcoin reste en phase de consolidation."
)
pdf.chapter_body(conclusion)

# Output
pdf_file_path = "COT_Report_Analysis_2025.pdf"
pdf.output(pdf_file_path)

pdf_file_path