"""Build a three-page ECS170 Stage 5 PDF using ONLY saved full-run results.

Fails closed on missing data, smoke runs, unknown dataset provenance, missing
figures, or a mismatch between requested and recorded dataset sources.
"""
import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas

DATASETS = ("cora", "citeseer", "pubmed")
REPO_URL = "https://github.com/yyryang-source/ECS170-Spam-Detection"
LEFT = 42
RIGHT = 570
FONT = "Helvetica"
BOLD = "Helvetica-Bold"
BODY = 10


def load_evidence(root, confirmation):
    evidence = {}
    for name in DATASETS:
        folder = root / name / "full"
        summary_path = folder / "experiment_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing complete {name} experiment: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("quick_smoke_test_not_for_report", True):
            raise ValueError(f"{name} is a smoke run; cannot create a final report")
        provenance = summary.get("dataset", {})
        source = provenance.get("source")
        if confirmation == "instructor":
            if source != "instructor_local_unverified":
                raise ValueError(f"{name} did not use local instructor-provided files")
            if not provenance.get("sha256"):
                raise ValueError(f"{name} has no source-file SHA256")
        elif confirmation == "reference":
            if source != "public_planetoid_reference":
                raise ValueError(f"{name} is not a public Planetoid reference run")
        else:
            raise ValueError("Source confirmation required")
        if summary.get("test_used_for_selection") is not False:
            raise ValueError(f"{name} lacks held-out test selection provenance")
        if not summary.get("validation") or not summary.get("final_test"):
            raise ValueError(f"{name} missing classification/ablation metrics")
        for required in ("architecture.png", "best_loss_curve.png",
                         "ablation_val_loss.png", "confusion_matrix.png"):
            if not (folder/required).is_file():
                raise FileNotFoundError(f"Missing report figure: {folder/required}")
        evidence[name] = (summary, folder)
    return evidence


def lines(pdf, text, x, y, width=525, leading=13):
    for line in simpleSplit(text, FONT, BODY, width):
        pdf.drawString(x, y, line)
        y -= leading
    return y


def heading(pdf, text, y):
    pdf.setFont(BOLD, 11)
    pdf.drawString(LEFT, y, text)
    pdf.setFont(FONT, BODY)
    return y - 19


def table(pdf, headers, rows, column_widths, y, row_height=19):
    if sum(column_widths) > RIGHT-LEFT+0.1:
        raise ValueError("Table exceeds page width")
    x0 = LEFT
    pdf.setStrokeColor(colors.Color(.70, .74, .80))
    pdf.setFillColor(colors.Color(.92, .94, .97))
    pdf.rect(x0, y-row_height, sum(column_widths), row_height, stroke=1, fill=1)
    pdf.setFillColor(colors.black)
    pdf.setFont(BOLD, 10)
    xx = x0
    for column, width in zip(headers, column_widths):
        pdf.drawString(xx+4, y-13, str(column))
        xx += width
    y -= row_height
    pdf.setFont(FONT, 10)
    for n, row in enumerate(rows):
        if n % 2 == 1:
            pdf.setFillColor(colors.Color(.965, .975, .99))
            pdf.rect(x0, y-row_height, sum(column_widths), row_height, stroke=0, fill=1)
        pdf.setFillColor(colors.black)
        xx = x0
        for value, width in zip(row, column_widths):
            pdf.drawString(xx+4, y-13, str(value)[:max(2, int(width/5.5))])
            xx += width
        pdf.setStrokeColor(colors.Color(.82, .84, .87))
        pdf.line(x0, y-row_height, x0+sum(column_widths), y-row_height)
        y -= row_height
    return y


def image_fit(pdf, path, x, y, max_width, max_height):
    img = ImageReader(str(path))
    iw, ih = img.getSize()
    ratio = min(max_width/iw, max_height/ih)
    width, height = iw*ratio, ih*ratio
    pdf.drawImage(img, x + (max_width-width)/2, y + (max_height-height)/2,
                  width=width, height=height, preserveAspectRatio=True)


def footer(pdf, page):
    pdf.setFont(FONT, 9)
    pdf.setFillColor(colors.grey)
    pdf.drawString(LEFT, 25, "ECS 170 | Stage 5 | Yourui Yang")
    pdf.drawRightString(RIGHT, 25, f"Page {page} of 3")
    pdf.setFillColor(colors.black)


def build_report(evidence, destination, confirmation, student_id=""):
    pdf = canvas.Canvas(str(destination), pagesize=letter, pageCompression=1)
    pdf.setTitle("ECS170 Stage 5 - GCN Node Classification")
    pdf.setAuthor("Yourui Yang")
    pdf.setFont(BODY and FONT, BODY)

    # Page 1: task, architecture, dataset/experimental setup and source code.
    pdf.setFont(BOLD, 14)
    pdf.drawString(LEFT, 757, "ECS 170 - Stage 5: Graph Neural Networks")
    pdf.setFont(FONT, 10)
    pdf.drawString(LEFT, 738,
                   "Individual Project | Yourui Yang" +
                   (f" | Student ID: {student_id}" if student_id else ""))
    source_note = ("Instructor-local (user-confirmed; archive hashes recorded)" if
                   confirmation == "instructor" else
                   "PUBLIC PLANETOID REFERENCE ONLY - NOT INSTRUCTOR DATA")
    pdf.drawString(LEFT, 722, f"Data provenance: {source_note}")

    y = heading(pdf, "Section 1: Task Description", 698)
    lines(pdf, "Perform transductive node classification and learn graph-aware node "
          "embeddings on Cora, Citeseer, and Pubmed citation networks. Nodes are "
          "documents; their features and citation edges propagate evidence through "
          "graph convolution, while held-out node labels remain unseen in training.",
          LEFT, y)
    y = heading(pdf, "Section 2: Model Description", 632)
    image_fit(pdf, evidence["cora"][1]/"architecture.png", LEFT, 475, 528, 135)
    lines(pdf, "Each GCN applies A_norm = D^(-1/2)(A+I)D^(-1/2), Linear, ReLU "
          "and Dropout; the output layer produces class logits. Sparse COO avoids "
          "dense NxN memory growth. Node embeddings are the last hidden activations.",
          LEFT, 461)

    heading(pdf, "Section 3: Experiment Settings", 411)
    heading(pdf, "3.1 Dataset Description", 392)
    rows = []
    for name in DATASETS:
        info = evidence[name][0]["dataset"]
        rows.append([
            name.title(), info["nodes"], info["features"], info["classes"],
            info["train"], info["val"], info["test"]
        ])
    table(pdf, ["Dataset", "Nodes", "Features", "Classes", "Train", "Val", "Test"],
          rows, [76, 68, 88, 75, 72, 72, 77], 369)
    heading(pdf, "3.2 Detailed Experimental Setups", 270)
    first = evidence["cora"][0]
    y = lines(pdf, f"All runs use seed {first['seed']}, Adam optimizer "
              f"(learning rate {first['learning_rate']}, weight decay "
              f"{first['weight_decay']}); up to {first['epochs_max']} epochs with "
              f"early stopping (patience {first['early_stopping_patience']}).",
              LEFT, 251)
    lines(pdf, "Ablations vary depth (2 versus 3), hidden width (64 versus 128), "
          "and dropout (0.5 versus 0.0); architecture selection uses validation "
          "macro F1 and checkpoints minimize validation cross-entropy.", LEFT, y-2)
    heading(pdf, "3.3 Evaluation Metrics", 185)
    lines(pdf, "Accuracy is correct test-node predictions / total test nodes. "
          "Macro Precision, Recall and F1 average class-specific metrics equally; "
          "test labels are never used for checkpoint or architecture selection.",
          LEFT, 166)
    heading(pdf, "3.4 Source Code", 112)
    pdf.drawString(LEFT, 93, REPO_URL)
    footer(pdf, 1)
    pdf.showPage()

    # Page 2: all three actual training convergence curves.
    pdf.setFont(FONT, 10)
    heading(pdf, "3.5 Training Convergence Plots (Train and Validation Loss)", 755)
    for i, name in enumerate(DATASETS):
        top = 719 - i*218
        pdf.setFont(BOLD, 10)
        pdf.drawString(LEFT, top, name.title() + " - selected model: " +
                       evidence[name][0]["selected_model"])
        pdf.setFont(FONT, 10)
        image_fit(pdf, evidence[name][1]/"best_loss_curve.png",
                  LEFT, top-188, 528, 182)
    footer(pdf, 2)
    pdf.showPage()

    # Page 3: held-out test scores and validation ablation tables.
    pdf.setFont(FONT, 10)
    heading(pdf, "3.6 Model Performance (Held-out Test Nodes)", 755)
    perf = []
    for name in DATASETS:
        summary = evidence[name][0]
        scores = summary["final_test"]
        perf.append([
            name.title(), summary["selected_model"],
            f"{scores['Accuracy']:.4f}", f"{scores['Precision']:.4f}",
            f"{scores['Recall']:.4f}", f"{scores['F1']:.4f}"
        ])
    y = table(pdf, ["Dataset", "Best GCN", "Accuracy", "Macro P", "Macro R", "Macro F1"],
              perf, [74, 105, 86, 86, 88, 89], 728)
    heading(pdf, "3.7 Ablation Studies (Validation Split Only)", y-25)
    y -= 55
    for name in DATASETS:
        summary = evidence[name][0]
        pdf.setFont(BOLD, 10)
        pdf.drawString(LEFT, y, name.title())
        pdf.setFont(FONT, 10)
        y -= 9
        rows = []
        for config in summary["validation"]:
            rows.append([
                config["Model"], f"{config['Parameters']:,}",
                f"{config['Accuracy']:.4f}", f"{config['F1']:.4f}"
            ])
        y = table(pdf, ["Architecture", "Parameters", "Validation Acc", "Validation Macro F1"],
                  rows, [145, 115, 132, 136], y, 18) - 30
    if y < 85:
        raise RuntimeError("Report exceeded three pages; reduce ablation variants")
    pdf.setFont(FONT, 10)
    lines(pdf, "Interpret all architecture gains using held-out validation comparisons; "
          "larger/deeper GCNs may overfit small labeled citation splits. "
          "Single-seed results do not establish statistical significance.",
          LEFT, y-5)
    footer(pdf, 3)
    pdf.save()
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/stage5"))
    p.add_argument("--out", type=Path, default=Path("results/stage5/ECS170_Stage5_Yourui_Yang.pdf"))
    p.add_argument("--source-confirmation", choices=("instructor", "reference"), required=True)
    p.add_argument("--student-id", default="", help="Optional student ID; not hard-coded")
    opts = p.parse_args()
    datasets = load_evidence(opts.results, opts.source_confirmation)
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    print("Report created:", build_report(
        datasets, opts.out, opts.source_confirmation, opts.student_id))


if __name__ == "__main__":
    main()
