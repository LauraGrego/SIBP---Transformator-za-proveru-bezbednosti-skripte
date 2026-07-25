import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUT_DIR / "izvestaj_o_projektu_bash_script_check.pdf"
METRICS = json.loads((ROOT / "tmp" / "pdfs" / "metrics.json").read_text(encoding="utf-8"))

NAVY = colors.HexColor("#0B1F33")
BLUE = colors.HexColor("#1877F2")
CYAN = colors.HexColor("#27C2D1")
GREEN = colors.HexColor("#20A86B")
ORANGE = colors.HexColor("#F59E42")
RED = colors.HexColor("#D94A4A")
INK = colors.HexColor("#243548")
MUTED = colors.HexColor("#60758A")
LIGHT = colors.HexColor("#EDF3F8")
PALE_BLUE = colors.HexColor("#EAF3FF")
PALE_ORANGE = colors.HexColor("#FFF4E8")
WHITE = colors.white

pdfmetrics.registerFont(TTFont("UI", r"C:\Windows\Fonts\segoeui.ttf"))
pdfmetrics.registerFont(TTFont("UI-Bold", r"C:\Windows\Fonts\segoeuib.ttf"))


class MetricCards(Flowable):
    def __init__(self, items, width=170*mm, height=31*mm):
        super().__init__()
        self.items = items
        self.width = width
        self.height = height

    def draw(self):
        gap = 4 * mm
        box_w = (self.width - gap * (len(self.items) - 1)) / len(self.items)
        for i, (value, label, color) in enumerate(self.items):
            x = i * (box_w + gap)
            self.canv.setFillColor(color)
            self.canv.roundRect(x, 0, box_w, self.height, 3*mm, fill=1, stroke=0)
            self.canv.setFillColor(WHITE)
            self.canv.setFont("UI-Bold", 18)
            self.canv.drawCentredString(x + box_w/2, 17*mm, value)
            self.canv.setFont("UI", 8.5)
            self.canv.drawCentredString(x + box_w/2, 7*mm, label)


class HorizontalBars(Flowable):
    def __init__(self, labels, values, colors_list, width=166*mm, height=62*mm, maximum=1.0):
        super().__init__()
        self.labels, self.values, self.colors_list = labels, values, colors_list
        self.width, self.height, self.maximum = width, height, maximum

    def draw(self):
        left = 34*mm
        bar_w = self.width - left - 15*mm
        top = self.height - 7*mm
        row_h = (self.height - 10*mm) / len(self.labels)
        for i, (label, value, color) in enumerate(zip(self.labels, self.values, self.colors_list)):
            y = top - i*row_h
            self.canv.setFillColor(INK)
            self.canv.setFont("UI", 9)
            self.canv.drawRightString(left - 3*mm, y - 1.5*mm, label)
            self.canv.setFillColor(LIGHT)
            self.canv.roundRect(left, y-3*mm, bar_w, 5*mm, 2.5*mm, fill=1, stroke=0)
            self.canv.setFillColor(color)
            self.canv.roundRect(left, y-3*mm, bar_w*(value/self.maximum), 5*mm, 2.5*mm, fill=1, stroke=0)
            self.canv.setFillColor(INK)
            self.canv.setFont("UI-Bold", 8.5)
            self.canv.drawString(left + bar_w + 2*mm, y-1.5*mm, f"{value*100:.2f}%")


class Pipeline(Flowable):
    def __init__(self, width=170*mm, height=34*mm):
        super().__init__(); self.width=width; self.height=height

    def draw(self):
        steps = [("Ulazni Bash tekst", PALE_BLUE), ("Tokenizacija", LIGHT), ("Transformer\nenkoder", PALE_BLUE), ("3 klase + razlog", LIGHT)]
        box_w = 35*mm
        gap = (self.width - len(steps)*box_w)/(len(steps)-1)
        for i, (label, fill) in enumerate(steps):
            x = i*(box_w+gap)
            self.canv.setFillColor(fill); self.canv.setStrokeColor(BLUE)
            self.canv.roundRect(x, 6*mm, box_w, 20*mm, 3*mm, fill=1, stroke=1)
            self.canv.setFillColor(INK); self.canv.setFont("UI-Bold", 8.5)
            lines = label.split("\n")
            for j, line in enumerate(lines):
                self.canv.drawCentredString(x+box_w/2, 17*mm-j*4*mm, line)
            if i < len(steps)-1:
                ax = x+box_w+1.5*mm
                ay = 16*mm
                self.canv.setStrokeColor(CYAN); self.canv.setLineWidth(1.5)
                self.canv.line(ax, ay, ax+gap-3*mm, ay)
                self.canv.line(ax+gap-5*mm, ay+2*mm, ax+gap-3*mm, ay)
                self.canv.line(ax+gap-5*mm, ay-2*mm, ax+gap-3*mm, ay)


def page_header_footer(canvas, doc):
    page = canvas.getPageNumber()
    if page == 1:
        return
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E3EC")); canvas.line(20*mm, 283*mm, 190*mm, 283*mm)
    canvas.setFont("UI-Bold", 8); canvas.setFillColor(NAVY)
    canvas.drawString(20*mm, 287*mm, "BASH SCRIPT CHECK")
    canvas.setFont("UI", 8); canvas.setFillColor(MUTED)
    canvas.drawRightString(190*mm, 287*mm, "Tehnički izveštaj | 25. jul 2026.")
    canvas.line(20*mm, 14*mm, 190*mm, 14*mm)
    canvas.drawString(20*mm, 8.5*mm, "Eksperimentalni klasifikator - ne koristiti kao jedinu bezbednosnu kontrolu")
    canvas.drawRightString(190*mm, 8.5*mm, str(page))
    canvas.restoreState()


def cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY); canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setFillColor(BLUE); canvas.circle(175*mm, 255*mm, 44*mm, fill=1, stroke=0)
    canvas.setFillColor(CYAN); canvas.circle(186*mm, 243*mm, 22*mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE); canvas.setFont("UI-Bold", 10)
    canvas.drawString(23*mm, 267*mm, "TEHNIČKI IZVEŠTAJ")
    canvas.setFont("UI-Bold", 30); canvas.drawString(23*mm, 226*mm, "Bash Script Check")
    canvas.setFont("UI", 16); canvas.setFillColor(colors.HexColor("#C8D8E8"))
    canvas.drawString(23*mm, 212*mm, "Transformer za procenu bezbednosti skripti")
    canvas.setFillColor(colors.HexColor("#173B5E")); canvas.roundRect(23*mm, 133*mm, 164*mm, 52*mm, 4*mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE); canvas.setFont("UI-Bold", 11); canvas.drawString(32*mm, 169*mm, "CILJ PROJEKTA")
    canvas.setFont("UI", 11)
    lines = ["Automatska klasifikacija Bash skripti kao bezbednih,", "rizičnih ili zlonamernih, uz dodatno objašnjenje", "tipa zlonamernog ponašanja."]
    for i, line in enumerate(lines): canvas.drawString(32*mm, (155-i*7)*mm, line)
    canvas.setFillColor(CYAN); canvas.roundRect(23*mm, 72*mm, 51*mm, 32*mm, 3*mm, fill=1, stroke=0)
    canvas.setFillColor(GREEN); canvas.roundRect(80*mm, 72*mm, 51*mm, 32*mm, 3*mm, fill=1, stroke=0)
    canvas.setFillColor(ORANGE); canvas.roundRect(137*mm, 72*mm, 51*mm, 32*mm, 3*mm, fill=1, stroke=0)
    for x, val, lab in [(48.5, "4.000", "uzoraka"), (105.5, "3", "glavne klase"), (162.5, "98,51%", "tačnost razloga")]:
        canvas.setFillColor(WHITE); canvas.setFont("UI-Bold", 17); canvas.drawCentredString(x*mm, 90*mm, val)
        canvas.setFont("UI", 8.5); canvas.drawCentredString(x*mm, 80*mm, lab)
    canvas.setFont("UI", 9); canvas.setFillColor(colors.HexColor("#9FB5C9")); canvas.drawString(23*mm, 27*mm, "Verzija izveštaja 1.0  |  25. jul 2026.")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1x", fontName="UI-Bold", fontSize=21, leading=25, textColor=NAVY, spaceAfter=7*mm))
styles.add(ParagraphStyle(name="H2x", fontName="UI-Bold", fontSize=13, leading=16, textColor=BLUE, spaceBefore=4*mm, spaceAfter=2.5*mm))
styles.add(ParagraphStyle(name="Bodyx", fontName="UI", fontSize=9.6, leading=14.2, textColor=INK, spaceAfter=3*mm))
styles.add(ParagraphStyle(name="Smallx", fontName="UI", fontSize=8.2, leading=11, textColor=MUTED))
styles.add(ParagraphStyle(name="Calloutx", fontName="UI", fontSize=9.2, leading=13, textColor=INK, leftIndent=4*mm, rightIndent=4*mm, spaceBefore=2*mm, spaceAfter=2*mm))
styles.add(ParagraphStyle(name="Centerx", fontName="UI-Bold", fontSize=10, leading=13, textColor=NAVY, alignment=TA_CENTER))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def table(data, widths, header=True, font_size=8.5, aligns=None):
    wrapped = []
    for row_index, row in enumerate(data):
        wrapped_row = []
        for cell in row:
            if isinstance(cell, Flowable):
                wrapped_row.append(cell)
            else:
                is_header = header and row_index == 0
                wrapped_row.append(Paragraph(
                    str(cell),
                    ParagraphStyle(
                        name=f"cell-{row_index}", fontName="UI-Bold" if is_header else "UI",
                        fontSize=font_size, leading=font_size+2.5,
                        textColor=WHITE if is_header else INK,
                    ),
                ))
        wrapped.append(wrapped_row)
    t = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0,0), (-1,-1), "UI"), ("FONTSIZE", (0,0), (-1,-1), font_size),
        ("TEXTCOLOR", (0,0), (-1,-1), INK), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#D6E0E8")),
        ("ROWBACKGROUNDS", (0,1 if header else 0), (-1,-1), [WHITE, colors.HexColor("#F7FAFC")]),
    ]
    if header:
        commands += [("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE), ("FONTNAME", (0,0), (-1,0), "UI-Bold")]
    if aligns:
        for column, align in enumerate(aligns): commands.append(("ALIGN", (column,0), (column,-1), align))
    t.setStyle(TableStyle(commands)); return t


story = []

# Cover page occupies its own template; a spacer keeps Platypus content off it.
story += [Spacer(1, 250*mm), PageBreak()]

story += [P("1. Sažetak projekta", "H1x")]
story += [P("Bash Script Check je eksperimentalni sistem za statičku analizu teksta shell skripti. Model ne izvršava kod, već iz njegovih tokena predviđa jednu od tri klase: <b>safe</b> (bezbedno), <b>risky</b> (rizično) ili <b>malicious</b> (zlonamerno). Za zlonamerne uzorke drugi klasifikacioni izlaz procenjuje kategoriju ponašanja, na primer reverse shell, eksfiltraciju ili perzistenciju.")]
story += [MetricCards([("100,00%", "tačnost na izdvojenom testu", BLUE), ("100,00%", "macro-F1 za 3 klase", GREEN), ("98,51%", "tačnost razloga napada", ORANGE)]), Spacer(1, 6*mm)]
story += [P("Najvažniji nalaz", "H2x"), P("Sačuvani checkpoint je na reproduktivnom, stratifikovanom test skupu od 801 uzorka tačno klasifikovao svih 801 primera. Ovaj rezultat je tehnički ispravno izmeren, ali je verovatno optimističan: bezbedni i rizični primeri su uglavnom generisani iz ograničenog broja šablona, pa test skup nije potpuno nezavisan od obrasca trening podataka.")]
callout = Table([[P("<b>Praktična interpretacija:</b> model je veoma uspešan na podacima istog porekla, ali pre produkcione upotrebe mora se proveriti na novom, ručno označenom skupu iz drugih repozitorijuma i organizacija.", "Calloutx")]], colWidths=[170*mm])
callout.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),PALE_ORANGE),("BOX",(0,0),(-1,-1),0.8,ORANGE),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
story += [callout, Spacer(1, 4*mm), P("Ključne karakteristike", "H2x")]
story += [table([
    ["Element", "Opis"],
    ["Ulaz", "Bash/shell skripta kao običan tekst; nema izvršavanja"],
    ["Model", "Encoder-only Transformer sa dva klasifikaciona izlaza"],
    ["Primarni izlaz", "safe / risky / malicious"],
    ["Sekundarni izlaz", "7 kategorija zlonamernog ponašanja"],
    ["Interfejs", "CLI tokovi za CPU, GPU i predikciju showcase skripti"],
], [42*mm,128*mm])]

story += [PageBreak(), P("2. Arhitektura i tok obrade", "H1x"), Pipeline(), Spacer(1, 4*mm)]
story += [P("Skripta se tokenizuje WordLevel tokenizatorom sa whitespace pre-tokenizacijom. Sekvenci se dodaje [CLS] token i ograničava se na kontekst od 384 tokena. Četiri Transformer enkoderska bloka formiraju reprezentaciju, a prosečno objedinjavanje validnih tokena prosleđuje se u dve linearne glave.")]
story += [P("Konfiguracija sačuvanog modela", "H2x")]
story += [table([
    ["Parametar", "Vrednost", "Značenje"],
    ["Dimenzija modela", "192", "širina embeddings/skrivenih reprezentacija"],
    ["Attention glave", "6", "paralelni obrasci pažnje"],
    ["Encoder blokovi", "4", "dubina Transformer enkodera"],
    ["Feed-forward sloj", "768", "unutrašnja FFN dimenzija"],
    ["Maks. kontekst", "384 tokena", "duže skripte se skraćuju"],
    ["Dropout", "0,10", "regularizacija tokom treninga"],
], [47*mm,32*mm,91*mm], aligns=["LEFT","CENTER","LEFT"])]
story += [P("Trening", "H2x"), P("Checkpoint je treniran 12 epoha, batch veličinom 32, AdamW-kompatibilnim parametrima learning rate 3×10<super>-4</super> i weight decay 1×10<super>-4</super>. Ukupni gubitak kombinuje klasifikaciju glavne klase i klasifikaciju razloga, sa težinom sekundarnog gubitka 0,5.")]
story += [P("Bezbednosna granica sistema", "H2x"), P("Sistem je statički klasifikator, a ne sandbox. Ne prati tokove podataka kroz izvršavanje, ne razrešava dinamički generisan kod i ne može garantovati da je skripta bezbedna. Predikcija <i>safe</i> zato nikada ne sme automatski da odobri izvršavanje nepoznate skripte.")]

story += [PageBreak(), P("3. Podaci i poreklo uzoraka", "H1x")]
story += [P("Projektni tok podataka obuhvata tri izvora: <b>sintetički generisane</b> bezbedne i rizične skripte, <b>scrapeovane GitHub</b> shell skripte sa heurističkim oznakama i <b>Red Team Operations Shell Script Dataset</b> sa Hugging Face-a za zlonamerne primere. Fajl korišćen za trenutni checkpoint sadrži 4.000 redova; u njemu se jasno potvrđuju sintetički safe/risky identifikatori i 1.000 RTO zlonamernih uzoraka. GitHub scraper postoji kao ulazni kanal, ali kombinovani artefakt ne čuva dokaz da je svaki scrapeovani zapis uključen u ovaj konkretan trening.")]
story += [P("Raspodela glavnih klasa", "H2x")]
story += [table([
    ["Klasa", "Broj", "Udeo", "Tipični sadržaj"],
    ["safe", "1.500", "37,5%", "lokalna automatizacija, provere, build i backup"],
    ["risky", "1.500", "37,5%", "sudo, sistemske izmene, brisanje, firewall"],
    ["malicious", "1.000", "25,0%", "RTO tehnike i ofanzivne komande"],
    ["Ukupno", "4.000", "100,0%", "JSONL skup za trening i evaluaciju"],
], [35*mm,25*mm,25*mm,85*mm], aligns=["LEFT","CENTER","CENTER","LEFT"]), Spacer(1,4*mm)]
story += [HorizontalBars(["safe", "risky", "malicious"], [1500/1500,1500/1500,1000/1500], [GREEN,ORANGE,RED], maximum=1.0), P("Grafik 1. Relativna veličina klasa (najveća klasa = 100%).", "Smallx")]
story += [P("Poreklo i kontrola kvaliteta", "H2x")]
story += [table([
    ["Izvor", "Uloga", "Kontrola / rizik"],
    ["Sintetički generator", "safe i risky varijante iz parametrizovanih šablona", "deduplikacija SHA-1; moguć šablonski bias"],
    ["GitHub scraping", "realne install/deploy/admin skripte", "heurističke oznake; obavezna ručna provera i licenca"],
    ["Hugging Face RTO", "1.000 malicious uzoraka i kategorije ponašanja", "kurirani skup; proveriti format i nezavisnost evaluacije"],
], [37*mm,67*mm,66*mm])]

story += [PageBreak(), P("4. Evaluacija modela", "H1x")]
story += [P("Evaluacija je reprodukovana nad sačuvanim checkpoint-om. Skup je stratifikovano podeljen seed-om 42 i test frakcijom 20%, uz očuvanje glavnih klasa i kategorija zlonamernog ponašanja. Zbog zaokruživanja po podgrupama izdvojeno je 801 test primera: 300 safe, 300 risky i 201 malicious.")]
story += [P("Glavni skorovi", "H2x")]
story += [HorizontalBars(["Accuracy", "Macro-F1", "Reason accuracy"], [1.0,1.0,METRICS["reason_accuracy"]], [BLUE,GREEN,ORANGE], maximum=1.0), P("Grafik 2. Rezultati na reproduktivnom izdvojenom test skupu.", "Smallx")]
story += [Spacer(1,4*mm), table([
    ["Metrika", "Skor", "Broj uzoraka", "Tumačenje"],
    ["Accuracy", "100,00%", "801", "udeo tačnih predikcija glavne klase"],
    ["Macro-F1", "100,00%", "801", "jednaka težina za sve tri klase"],
    ["Reason accuracy", "98,51%", "201 malicious", "tačnost dodatne kategorije napada"],
], [42*mm,30*mm,35*mm,63*mm], aligns=["LEFT","CENTER","CENTER","LEFT"])]
story += [P("Metričke definicije", "H2x"), P("Precision meri koliko je predikcija date klase ispravno; recall meri koliko je stvarnih primera klase pronađeno; F1 je harmonijska sredina precision-a i recall-a. Macro-F1 je aritmetička sredina F1 skorova klasa i zato ne favorizuje većinsku klasu.")]
story += [P("Važna napomena", "H2x")]
warning = Table([[P("Rezultat od 100% nije dovoljan dokaz generalizacije. Najverovatniji uzroci su ograničen broj generator šablona, varijante istih obrazaca u train/test podeli i karakterističan stil RTO korpusa. Potrebna je evaluacija po izvorima (group split), kao i potpuno eksterni test skup.", "Calloutx")]], colWidths=[170*mm])
warning.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),PALE_ORANGE),("BOX",(0,0),(-1,-1),0.8,ORANGE),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
story += [warning]

story += [PageBreak(), P("5. Detaljni rezultati", "H1x")]
story += [P("Rezultati po klasi", "H2x")]
story += [table([
    ["Klasa", "Precision", "Recall", "F1", "Podrška"],
    ["safe", "100,00%", "100,00%", "100,00%", "300"],
    ["risky", "100,00%", "100,00%", "100,00%", "300"],
    ["malicious", "100,00%", "100,00%", "100,00%", "201"],
    ["macro prosek", "100,00%", "100,00%", "100,00%", "801"],
], [40*mm,31*mm,31*mm,31*mm,37*mm], aligns=["LEFT","CENTER","CENTER","CENTER","CENTER"])]
story += [P("Matrica konfuzije", "H2x"), P("Redovi predstavljaju stvarne, a kolone predviđene klase.", "Smallx")]
cm = Table([
    ["stvarno \\ predviđeno", "safe", "risky", "malicious"],
    ["safe", "300", "0", "0"],
    ["risky", "0", "300", "0"],
    ["malicious", "0", "0", "201"],
], colWidths=[58*mm,37*mm,37*mm,38*mm])
cm.setStyle(TableStyle([
    ("FONTNAME",(0,0),(-1,-1),"UI-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("ALIGN",(0,0),(-1,-1),"CENTER"),
    ("BACKGROUND",(0,0),(-1,0),NAVY),("BACKGROUND",(0,1),(0,-1),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),("TEXTCOLOR",(0,1),(0,-1),WHITE),
    ("BACKGROUND",(1,1),(1,1),colors.HexColor("#DDF5E9")),("BACKGROUND",(2,2),(2,2),colors.HexColor("#DDF5E9")),("BACKGROUND",(3,3),(3,3),colors.HexColor("#DDF5E9")),
    ("GRID",(0,0),(-1,-1),0.6,colors.HexColor("#C9D5DF")),("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
]))
story += [cm]
story += [P("Kategorije zlonamernog ponašanja", "H2x")]
story += [table([
    ["Kategorija", "Broj u skupu", "Test podrška", "Napomena"],
    ["recon", "217", "43", "1 primer zamenjen sa defense_evasion"],
    ["reverse_shell", "195", "39", "svi tačni"],
    ["exfiltration", "195", "39", "1 primer zamenjen sa recon"],
    ["defense_evasion", "194", "39", "svi tačni"],
    ["persistence", "193", "39", "svi tačni"],
    ["obfuscation", "4", "1", "premalo za pouzdan zaključak"],
    ["privilege_escalation", "2", "1", "premalo; primer zamenjen sa recon"],
], [49*mm,31*mm,30*mm,60*mm], font_size=7.9, aligns=["LEFT","CENTER","CENTER","LEFT"])]

story += [PageBreak(), P("6. Ograničenja i preporuke", "H1x")]
story += [P("Glavni rizici validnosti", "H2x")]
story += [table([
    ["Rizik", "Posledica", "Preporučena mera"],
    ["Šablonski sintetički podaci", "model može učiti stil generatora", "group split po šablonu i više realnih primera"],
    ["Heurističke scrape oznake", "pogrešno označavanje safe/risky", "dvostruka ručna anotacija i zapis saglasnosti"],
    ["Neravnoteža razloga", "slaba procena retkih kategorija", "dopuniti obfuscation i privilege_escalation"],
    ["Sličnost izvora", "prenaduvani test skorovi", "eksterni test iz novih repozitorijuma"],
    ["Skraćivanje na 384 tokena", "kritičan deo duge skripte može biti odsečen", "segmentna analiza i agregacija rezultata"],
    ["Statička analiza", "dinamički/obfuskovani kod može proći", "kombinovati sa sandbox-om i pravilima"],
], [45*mm,55*mm,70*mm], font_size=8.0)]
story += [P("Preporučeni sledeći koraci", "H2x")]
for text_item in [
    "1. Formirati potpuno nezavisan, ručno označen test skup sa najmanje 200 primera po glavnoj klasi.",
    "2. Uvesti group split po šablonu, repozitorijumu i izvoru kako srodni primeri ne bi završili u oba dela.",
    "3. Izveštavati weighted-F1, macro-F1, ROC/PR po klasi i intervale pouzdanosti, uz matricu konfuzije.",
    "4. Kalibrisati confidence skorove i definisati zonu neodlučnosti koja zahteva ljudski pregled.",
    "5. Čuvati provenance i licencu za svaki scrapeovani zapis; ne distribuirati kod bez provere dozvole.",
]: story.append(P(text_item))
story += [P("Zaključak", "H2x"), P("Projekat predstavlja funkcionalan end-to-end prototip za tekstualnu procenu Bash skripti: ima pripremu podataka, tokenizator, Transformer model, CPU/GPU trening, evaluaciju i komandnu predikciju. Interni rezultati su odlični, posebno glavni klasifikacioni izlaz. Najveći sledeći doprinos nije dodatno optimizovanje modela, već stroža i nezavisnija validacija podataka.")]
story += [P("Izvori", "H2x")]
story += [P("[1] Lokalni repozitorijum: README.md, generate_dataset.py, scrape_dataset.py, bash_classifier/*, data/*.jsonl i artifacts/bash_transformer.pt.<br/>[2] Hugging Face: <link href='https://huggingface.co/datasets/darkknight25/Red_Team_Operations_ShellScript_Dataset' color='#1877F2'>darkknight25/Red_Team_Operations_ShellScript_Dataset</link> (1.000 RTO skripti, MIT licenca prema dataset kartici).<br/>[3] Rezultati u ovom izveštaju reprodukovani su lokalno iz checkpoint-a i stratifikovane podele seed=42.", "Smallx")]


doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm, title="Bash Script Check - tehnički izveštaj", author="Projektni tim")
body_frame = Frame(20*mm,18*mm,170*mm,263*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id="body")
def decorated_page(canvas, document):
    if canvas.getPageNumber() == 1:
        cover(canvas, document)
    else:
        page_header_footer(canvas, document)

doc.addPageTemplates([PageTemplate(id="AllPages", frames=[body_frame], onPage=decorated_page)])
doc.build(story)
print(OUTPUT)
