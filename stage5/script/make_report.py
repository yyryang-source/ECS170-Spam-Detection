#!/usr/bin/env python3
"""Build the five-page course report and editable HTML from measured artifacts."""
from pathlib import Path
import argparse
import html
import json
import shutil
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.pagesizes import letter
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
BLUE = colors.HexColor("#204a6c")
INK = colors.HexColor("#172b3a")
GREY = colors.HexColor("#526372")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body10", fontName="Helvetica", fontSize=10, leading=11.8,
                          textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle(name="Heading10", fontName="Helvetica-Bold", fontSize=11,
                          leading=13, textColor=BLUE, spaceBefore=7, spaceAfter=6))
styles.add(ParagraphStyle(name="Caption10", fontName="Helvetica", fontSize=10, leading=11.5,
                          textColor=GREY, spaceAfter=6))
styles.add(ParagraphStyle(name="Title15", fontName="Helvetica-Bold", fontSize=15, leading=18,
                          alignment=TA_CENTER, textColor=BLUE, spaceAfter=7))
styles.add(ParagraphStyle(name="Center10", parent=styles["Body10"], alignment=TA_CENTER))
styles.add(ParagraphStyle(name="Cell", parent=styles["Body10"], fontSize=10, leading=11.5, spaceAfter=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=REPORT / "identity.json")
    parser.add_argument("--source-url", default="", help="Verified Stage 5 source URL after publication")
    args = parser.parse_args()
    identity = json.loads(args.identity.read_text())
    summary = json.loads((ROOT / "results/summary.json").read_text())
    groups = {(r["dataset"], r["variant"]): r for r in summary}
    datasets = ["cora", "pubmed", "citeseer"]
    names = {"cora": "Cora", "pubmed": "Pubmed", "citeseer": "Citeseer"}
    primary = {d: json.loads((ROOT / f"results/{d}/baseline/seed_42/metrics.json").read_text()) for d in datasets}
    metadata = {d: json.loads((ROOT / f"results/{d}/baseline/seed_42/dataset.json").read_text()) for d in datasets}
    story, html_pages, current_html = [], [], []

    def para(text, style="Body10"):
        story.append(Paragraph(text, styles[style]))
        tag = "h1" if style == "Title15" else "h2" if style == "Heading10" else "p"
        browser_text = re.sub(r'<link href="([^"]+)">', r'<a href="\1">', text).replace("</link>", "</a>")
        current_html.append(f'<{tag} class="{style}">{browser_text}</{tag}>')

    def table(rows, widths):
        cells = [[Paragraph(str(c), styles["Cell"]) for c in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff5")),
            ("LINEBELOW", (0, 0), (-1, 0), .7, BLUE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7f9")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([t, Spacer(1, 7)])
        current_html.append("<table>" + "".join("<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{c}</{'th' if i == 0 else 'td'}>" for c in row) + "</tr>" for i, row in enumerate(rows)) + "</table>")

    def figure(filename, width=510, max_height=None):
        path = REPORT / "figures" / filename
        iw, ih = ImageReader(str(path)).getSize()
        height = width * ih / iw
        if max_height is not None and height > max_height:
            width *= max_height / height
            height = max_height
        story.extend([Image(str(path), width=width, height=height), Spacer(1, 4)])
        current_html.append(f'<img src="figures/{html.escape(filename)}" style="width:{width / 72:.2f}in;max-width:100%" alt="{html.escape(filename)}">')

    def page():
        html_pages.append("\n".join(current_html))
        current_html.clear()
        story.append(PageBreak())

    def percent(d, variant, metric, sd=True):
        row = groups[(d, variant)]
        value = f"{row[metric + '_mean'] * 100:.2f}"
        return value + (f" +/- {row[metric + '_std'] * 100:.2f}" if sd else "")

    para("ECS 170 - Artificial Intelligence", "Title15")
    para("Fall 2026 | Course Project: Stage 5<br/><b>Graph Embedding and Node Classification with a GCN</b>", "Center10")
    para("Team Information", "Heading10")
    table([["Team", "Name", "Student ID", "Email"],
           [html.escape(identity["team"]), html.escape(identity["name"]), html.escape(identity["student_id"]), html.escape(identity["email"])]], [112, 105, 100, 193])
    para("Section 1: Task Description", "Heading10")
    para("This project implements and trains a graph convolutional network (GCN) to predict publication-topic labels in three instructor-provided citation graphs: Cora, Pubmed and Citeseer. Each node has a feature vector and a class label; citation links supply graph structure. The task is transductive node classification: all nodes' features and links are visible, while only the designated training labels enter the optimization objective. The hidden layer also provides a learned embedding for every node.")
    para("Section 2: Model Description", "Heading10")
    para("A custom two-layer GCN is implemented directly with PyTorch matrix operations, following the propagation principle in [1]. Let X be the row-normalized feature matrix, A the symmetrized citation adjacency, and D the degree matrix of A + I. The fixed graph operator is S = D<super>-1/2</super>(A + I)D<super>-1/2</super>. Each layer computes S(HW) + b. At evaluation time:")
    para("<b>H = ReLU(SXW<sub>0</sub> + b<sub>0</sub>), &nbsp; Z = SHW<sub>1</sub> + b<sub>1</sub>.</b>", "Center10")
    figure("architecture.png", max_height=112)
    para("Figure 1. The hidden representation H has 16 dimensions. Training applies dropout (p = 0.5) to the input and hidden activations. Cross-entropy consumes logits Z; argmax produces class predictions. Sparse propagation avoids a dense N x N adjacency.", "Caption10")
    para("Section 3: Experiment Settings", "Heading10")
    para("3.1 Dataset Description", "Heading10")
    table([["Dataset", "Nodes", "Features", "Classes", "Raw links", "Undirected edges"],
           *[[names[d], f"{metadata[d]['nodes']:,}", f"{metadata[d]['features']:,}", len(metadata[d]["classes"]), f"{metadata[d]['raw_link_rows']:,}", f"{metadata[d]['undirected_nonself_edges']:,}"] for d in datasets]], [68, 69, 69, 59, 96, 149])
    para("Raw links count input rows; undirected edges count unique non-self pairs after symmetrization. Each whitespace-delimited node row contains an ID, features, and a label. Cora/Citeseer features are binary; Pubmed features are continuous. IDs and link endpoints were checked. Citeseer contains 3,312 nodes in this supplied version, 124 existing self-links and 48 nodes isolated when self-links are excluded.", "Body10")
    page()

    para("3.1 Dataset Description (continued): data partition", "Heading10")
    para("The archive's ReadMe requires random, class-balanced train/test samples. The loader's contiguous index ranges are examples, so they are replaced by sampling without replacement. For each sorted class, a NumPy generator with split seed 42 selects training nodes, then test nodes, then a disjoint validation set. Ten validation nodes per class fit even Cora's smallest class (180 nodes).")
    table([["Dataset", "Train", "Validation", "Test", "Other nodes"],
           *[[names[d], metadata[d]["split_counts"]["idx_train"], metadata[d]["split_counts"]["idx_val"], metadata[d]["split_counts"]["idx_test"], metadata[d]["nodes"] - sum(metadata[d]["split_counts"].values())] for d in datasets]], [110, 90, 110, 90, 110])
    para("Every class contributes 20 training nodes; test counts are 150/class for Cora and 200/class for Pubmed/Citeseer. Other nodes contribute features and edges only. The same split is used in every run and ablation. These are the assignment's sampled splits, not canonical Planetoid masks. Test labels are used for stratification and final scoring, never the training loss, stopping rule or checkpoint selection.")
    para("3.2 Detailed Experimental Setups", "Heading10")
    table([["Setting", "Value shared by all datasets"],
           ["Architecture", "2 graph-convolution layers; ReLU; hidden width 16; dropout 0.5"],
           ["Layer dimensions", "Cora 1433 -> 16 -> 7; Pubmed 500 -> 16 -> 3; Citeseer 3703 -> 16 -> 6"],
           ["Optimization", "Full-batch Adam; learning rate 0.01; weight decay 0.0005 on all parameters"],
           ["Initialization", "Xavier uniform weights; zero biases; seeds 42, 43, 44"],
           ["Training budget / selection", "At most 300 epochs; patience 50; restore minimum validation cross-entropy"],
           ["Inputs / graph", "Row-L1 feature normalization; max(A, A transpose); add I; symmetric degree normalization"],
           ["Runtime", "CPU, 4 threads; Python 3.13.17; PyTorch 2.15.0; macOS 14.4 arm64"]], [133, 377])
    para("Feature normalization is an explicit extension of the supplied loader. Class names are sorted to stabilize label IDs. The original loop behavior is retained: adding I makes the 124 existing Citeseer diagonal entries equal to 2 before degree normalization. No node is removed. There are 23,063 / 8,067 / 59,366 trainable parameters for Cora / Pubmed / Citeseer, respectively.")
    para("3.3 Evaluation Metrics", "Heading10")
    para("Accuracy is correct predictions divided by test nodes. For class c, precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is 2PR/(P+R). Precision, recall and F1 are macro-averaged across all classes, with zero for undefined divisions. Because the test sets have equal support per class, macro recall equals accuracy. Results report mean +/- sample standard deviation over three initialization seeds; this measures optimization variation on one fixed split, not variation across independent datasets or splits.")
    para("3.4 Source Code", "Heading10")
    if args.source_url:
        para(f'Stage 5 source and reproducibility files: <link href="{html.escape(args.source_url)}">{html.escape(args.source_url)}</link>.')
    else:
        para("Complete Stage 5 source, model checkpoints and experimental evidence are supplied in the accompanying <b>Stage5_Project.zip</b>. An online Stage 5 source link has not yet been published; add the verified link here after upload.")
    para('The Dataset / Method / Setting / Evaluate / Result organization follows the existing <link href="https://github.com/yyryang-source/ECS170-Spam-Detection">Stage 4 repository</link> (reference snapshot 5fc3a78). The adapted instructor loader and original ReadMe are preserved under references/. Entry point: script/stage_5_script/script_gcn.py. Exact dependencies, split indices, file hashes, histories and predictions are included.')
    page()

    para("3.5 Training Convergence Plot", "Heading10")
    figure("learning_curves.png", max_height=430)
    para("Figure 2. Baseline width-16 GCN, preselected initialization seed 42. Left: cross-entropy on training and validation nodes. Right: accuracy on the same partitions. Both curves are measured after each update with dropout disabled, so they are directly comparable. Dashed vertical lines mark the minimum-validation-loss checkpoint. Test metrics are not monitored during training.", "Caption10")
    table([["Dataset", "Epochs run", "Selected epoch", "Train loss: first -> last", "Validation loss: first -> last"],
           ["Cora", "300", "300", "1.943 -> 0.211", "1.944 -> 0.779"],
           ["Pubmed", "201", "151", "1.095 -> 0.144", "1.095 -> 0.657"],
           ["Citeseer", "300", "298", "1.789 -> 0.222", "1.790 -> 1.106"]], [66, 71, 84, 140, 149])
    para("The training loss declines on all three graphs, demonstrating successful optimization. Pubmed validation loss reaches its best value earlier, triggering the patience rule at epoch 201. Cora and Citeseer use the full budget; their late selected epochs mean full convergence is not established. Training/validation gaps indicate limited-label generalization remains harder than fitting the training set. The reported predictions use the selected checkpoint, not automatically the final epoch.")
    page()

    para("3.6 Model Performance", "Heading10")
    para("Table 4 reports held-out test performance of the predeclared width-16 baseline. All values are percentages, mean +/- sample SD over seeds 42, 43 and 44. Each run selects its checkpoint using validation cross-entropy only.")
    table([["Dataset", "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"],
           *[[names[d], *[percent(d, "baseline", m) for m in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]]] for d in datasets]], [70, 105, 120, 105, 110])
    para("Pubmed achieves the highest baseline accuracy (84.50%), followed by Cora (80.67%) and Citeseer (62.22%). The ordering is descriptive: the graphs differ in class count, features, connectivity and split sizes, so it does not isolate the effect of any one property. Citeseer's disconnected structure and weaker same-label connectivity are plausible contributors, not causal conclusions.")
    class_rows = []
    for d in datasets:
        class_scores = [(label, primary[d]["per_class"][label]["f1-score"]) for label in metadata[d]["classes"]]
        lowest, highest = min(class_scores, key=lambda v: v[1]), max(class_scores, key=lambda v: v[1])
        label_text = lambda entry: f"{entry[0].replace('_', ' ')}: {entry[1]*100:.2f}%"
        class_rows.append([names[d], f"{primary[d]['accuracy']*100:.2f}%", label_text(lowest), label_text(highest)])
    table([["Seed-42 example", "Test accuracy", "Lowest class F1", "Highest class F1"], *class_rows], [100, 95, 155, 160])
    para("Class-level results show that aggregate accuracy can conceal uneven performance, especially for Citeseer. Full per-class precision/recall/F1 and confusion matrices are saved in each metrics.json, alongside test_predictions.csv with original node IDs. The width-16 hidden representation for every node is stored in embeddings.npz.")
    figure("embeddings_pca.png", max_height=222)
    para("Figure 3. Two-dimensional PCA projections of the learned 16-dimensional node embeddings (baseline, seed 42). Each panel fits PCA to that graph's embeddings; labels are used only for display colors. PCA is a lossy visualization, so overlap or apparent separation is qualitative and is not an additional test metric. Class IDs follow sorted labels in dataset.json. Axes are not comparable between graphs.", "Caption10")
    para("The result package preserves all 18 runs, rather than only favorable seeds. For each run it contains the restored checkpoint, selected epoch, complete learning history, exact row-index split and original node-ID predictions. The three-seed summary can therefore be recalculated directly from the recorded evidence.")
    page()

    para("3.7 Ablation Studies", "Heading10")
    para("The ablation doubles the hidden width from 16 to 32 while keeping depth, dropout, optimizer, training budget, validation criterion, split and initialization-seed list fixed. Both widths are reported without replacing the baseline based on test scores. Table 6 gives mean percentages over the same three seeds.")
    table([["Dataset / width", "Accuracy", "Precision", "Recall", "F1"],
           *[[f"{names[d]} / {16 if v == 'baseline' else 32}", *[percent(d, v, m, False) for m in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]]] for d in datasets for v in ["baseline", "hidden32"]]], [142, 92, 92, 92, 92])
    figure("ablation.png", max_height=154)
    para("Figure 4. Width comparison with sample-SD error bars across three initialization seeds. Accuracy starts at zero on the vertical axis. Error bars do not represent uncertainty over train/test splits.", "Caption10")
    para("Width 32 changes accuracy by +0.67 percentage points on Cora, -0.33 on Pubmed and +1.14 on Citeseer. Extra capacity therefore does not improve every dataset. With only three seeds on one split, these differences should be treated as exploratory; no statistical-significance claim is made.")
    para("Limitations and optional GPU experiment", "Heading10")
    para("The validation sets contain only 10 nodes per class, so checkpoint selection may be noisy. Only one stratified split, one depth and two widths were studied; there is no causal graph-versus-feature ablation. Late checkpoints on Cora/Citeseer suggest a longer budget could be investigated using validation data. This experiment does not claim state-of-the-art performance or direct comparability with published Planetoid benchmarks.")
    para("An isolated MPS smoke test completed forward propagation but aborted during backward propagation with NSInvalidArgumentException (SIGABRT, exit -6) on this Mac/PyTorch combination. All reported training used CPU. No GPU speedup is claimed; the probe script and captured evidence are included. CUDA was unavailable.")
    para("References", "Heading10")
    para('[1] T. N. Kipf and M. Welling. <i>Semi-Supervised Classification with Graph Convolutional Networks.</i> ICLR 2017. <link href="https://arxiv.org/abs/1609.02907">arxiv.org/abs/1609.02907</link>.<br/>[2] Instructor-provided stage_5_data/ReadMe.pdf, Dataset_Loader_Node_Classification.py and Stage 2-5 Report Template (ECS 170, Fall 2026).<br/>[3] PyTorch documentation: <link href="https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm">torch.sparse.mm</link> and <link href="https://docs.pytorch.org/docs/main/notes/mps.html">MPS backend</link>.', "Caption10")
    html_pages.append("\n".join(current_html))

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#d4dfe7"))
        canvas.line(51, 40, 561, 40)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY)
        canvas.drawString(51, 28, "ECS 170 | Stage 5 | Yourui Yang")
        canvas.drawRightString(561, 28, f"{doc.page} / 5")

    REPORT.mkdir(exist_ok=True)
    target = REPORT / "Stage5_Report.pdf"
    doc = SimpleDocTemplate(str(target), pagesize=letter, rightMargin=51, leftMargin=51,
                            topMargin=38, bottomMargin=51, title="Stage 5: GCN Node Classification",
                            author=identity["name"])
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    count = len(PdfReader(target).pages)
    if count != 5:
        raise RuntimeError(f"Report must have exactly 5 pages; produced {count}. Adjust layout before delivery.")
    css = '''@page{size:letter;margin:0}body{margin:0;background:#e8edf1;color:#172b3a;font:10pt/1.18 Arial,sans-serif}.page{box-sizing:border-box;width:8.5in;min-height:11in;padding:.53in .71in .65in;background:white;margin:24px auto;page-break-after:always}h1{text-align:center;color:#204a6c;font-size:15pt;line-height:1.2;margin:0 0 7pt}h2{font-size:11pt;color:#204a6c;margin:7pt 0 6pt}p{margin:0 0 6pt}.Center10{text-align:center}.Caption10{font-size:10pt;color:#526372}table{width:100%;border-collapse:collapse;margin:0 0 7pt;font-size:10pt}td,th{padding:4pt;text-align:left;vertical-align:top}th{background:#e8eff5;border-bottom:1px solid #204a6c}tr:nth-child(odd){background:#f5f7f9}img{display:block;margin:0 auto 4pt}a{color:#204a6c}@media print{body{background:white}.page{margin:0;min-height:11in;box-shadow:none}}'''
    browser_doc = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Stage 5 GCN report</title><style>' + css + '</style></head><body>'
    browser_doc += "\n".join('<section class="page" contenteditable="true">' + p + '</section>' for p in html_pages)
    browser_doc += '</body></html>'
    (REPORT / "Stage5_Report.html").write_text(browser_doc)
    print(f"Created {target} ({count} pages) and editable HTML")


if __name__ == "__main__":
    main()
