"""Generate a realistic sample corpus of ML 'papers' as PDFs with real charts.

Used for sandbox/dev testing where arXiv is unreachable. Each paper has:
title, abstract (real, from the actual paper), sections, and 2 matplotlib
figures embedded — so PDF text AND figure extraction get exercised.
Swap in real arXiv PDFs via scripts/download_arxiv.py when running outside.
"""
import pathlib, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUT = pathlib.Path("data/raw"); OUT.mkdir(parents=True, exist_ok=True)
FIGTMP = pathlib.Path("data/figures/_gen"); FIGTMP.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

PAPERS = [
    dict(
        pid="1706.03762", title="Attention Is All You Need",
        abstract=("The dominant sequence transduction models are based on complex recurrent or "
                  "convolutional neural networks in an encoder-decoder configuration. We propose a new "
                  "simple network architecture, the Transformer, based solely on attention mechanisms, "
                  "dispensing with recurrence and convolutions entirely. Our model achieves 28.4 BLEU on "
                  "the WMT 2014 English-to-German translation task. On English-to-French it establishes a "
                  "new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs."),
        sections={
            "Model Architecture": "The Transformer follows an encoder-decoder structure using stacked self-attention and point-wise fully connected layers. The encoder is composed of a stack of N=6 identical layers. Each layer has a multi-head self-attention mechanism and a position-wise feed-forward network, with residual connections and layer normalization. Multi-head attention allows the model to jointly attend to information from different representation subspaces: we employ h=8 parallel attention heads with dimension d_k = d_v = 64.",
            "Scaled Dot-Product Attention": "We compute attention as softmax(QK^T / sqrt(d_k))V. The scaling factor 1/sqrt(d_k) prevents the dot products from growing large in magnitude, which would push the softmax into regions of extremely small gradients. Figure 1 shows the attention weight distribution across heads.",
            "Results": "On WMT 2014 English-to-German translation, the big Transformer model outperforms the best previously reported models including ensembles by more than 2.0 BLEU, establishing a new state-of-the-art score of 28.4. Training took 3.5 days on 8 P100 GPUs. Figure 2 compares BLEU scores against training cost for competing architectures.",
        },
        figures=[("attention weight heatmap across 8 heads", "heatmap"),
                 ("BLEU score vs training cost comparison", "bar")],
    ),
    dict(
        pid="1810.04805", title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        abstract=("We introduce BERT, a new language representation model which stands for Bidirectional "
                  "Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional "
                  "representations from unlabeled text by jointly conditioning on both left and right context. "
                  "BERT obtains new state-of-the-art results on eleven natural language processing tasks, "
                  "including pushing the GLUE score to 80.5% and SQuAD v1.1 question answering Test F1 to 93.2."),
        sections={
            "Masked Language Model": "We mask 15% of input tokens at random and predict the masked tokens through a deep bidirectional Transformer encoder. Of the masked positions, 80% are replaced with [MASK], 10% with a random token, and 10% left unchanged. This mitigates the mismatch between pre-training and fine-tuning since [MASK] never appears during fine-tuning.",
            "Fine-tuning Results": "BERT-Large achieves 80.5 on GLUE, a 7.7 point absolute improvement over prior art. On SQuAD v1.1 it reaches 93.2 F1. Figure 1 shows GLUE scores per task, and Figure 2 shows the effect of pre-training steps on downstream accuracy.",
        },
        figures=[("GLUE benchmark scores per task", "bar"),
                 ("downstream accuracy vs pre-training steps", "line")],
    ),
    dict(
        pid="2005.11401", title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        abstract=("Large pre-trained language models store factual knowledge in their parameters but their "
                  "ability to access and precisely manipulate knowledge is limited. We explore retrieval-augmented "
                  "generation (RAG): models which combine pre-trained parametric memory (a seq2seq model) with "
                  "non-parametric memory (a dense vector index of Wikipedia accessed with a neural retriever). "
                  "RAG models achieve state-of-the-art on three open-domain QA tasks and generate more specific, "
                  "diverse and factual language than a parametric-only seq2seq baseline."),
        sections={
            "RAG Architecture": "RAG uses the query encoder of DPR to embed the input, retrieves the top-K documents from a dense FAISS index of 21M Wikipedia passages via maximum inner product search, and conditions a BART generator on both the input and the retrieved passages. We marginalize over retrieved documents either per output sequence (RAG-Sequence) or per token (RAG-Token).",
            "Open-Domain QA Results": "RAG sets new state-of-the-art results on Natural Questions, WebQuestions and CuratedTrec, outperforming both extractive readers and closed-book T5. Figure 1 shows exact-match accuracy against the number of retrieved documents K, peaking near K=10. Figure 2 illustrates retrieval recall@K on Natural Questions.",
        },
        figures=[("exact match accuracy vs retrieved documents K", "line"),
                 ("retrieval recall at K on Natural Questions", "line")],
    ),
    dict(
        pid="2010.11929", title="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        abstract=("While the Transformer architecture has become the de-facto standard for natural language "
                  "processing tasks, its applications to computer vision remain limited. We show that a pure "
                  "transformer applied directly to sequences of image patches can perform very well on image "
                  "classification tasks. When pre-trained on large amounts of data, Vision Transformer (ViT) "
                  "attains excellent results compared to state-of-the-art convolutional networks while requiring "
                  "substantially fewer computational resources to train."),
        sections={
            "Patch Embeddings": "We split an image into fixed-size 16x16 patches, linearly embed each patch, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. A learnable classification token is prepended, whose representation at the output serves as the image representation.",
            "Scaling Behavior": "ViT overtakes ResNets when pre-training data grows: on ImageNet-21k and JFT-300M, ViT-L/16 outperforms BiT-L while using far less pre-training compute. Figure 1 shows accuracy versus pre-training dataset size. Figure 2 shows the attention distance across layers, indicating that some heads attend globally already in early layers.",
        },
        figures=[("ImageNet accuracy vs pre-training dataset size", "line"),
                 ("mean attention distance across transformer layers", "scatter")],
    ),
    dict(
        pid="2106.09685", title="LoRA: Low-Rank Adaptation of Large Language Models",
        abstract=("We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and "
                  "injects trainable rank decomposition matrices into each layer of the Transformer architecture, "
                  "greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B "
                  "fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the "
                  "GPU memory requirement by 3 times, while performing on-par or better in model quality."),
        sections={
            "Method": "For a pre-trained weight matrix W0, we constrain its update by a low-rank decomposition W0 + BA where B and A contain trainable parameters and the rank r is much smaller than the matrix dimensions. We apply LoRA to the query and value projection matrices in the self-attention module. At deployment, BA can be merged into W0, introducing zero inference latency.",
            "Experiments": "LoRA matches or exceeds full fine-tuning quality on RoBERTa, DeBERTa, GPT-2 and GPT-3 while training only 0.01% of parameters. Figure 1 shows validation accuracy against the adaptation rank r, showing that very small ranks suffice. Figure 2 compares GPU memory usage for full fine-tuning versus LoRA.",
        },
        figures=[("validation accuracy vs LoRA rank r", "line"),
                 ("GPU memory usage comparison for fine-tuning methods", "bar")],
    ),
    dict(
        pid="2203.02155", title="Training Language Models to Follow Instructions with Human Feedback",
        abstract=("Making language models bigger does not inherently make them better at following a user's "
                  "intent. We show an avenue for aligning language models with user intent by fine-tuning with "
                  "human feedback. Starting with labeler-written prompts, we collect a dataset of demonstrations, "
                  "fine-tune GPT-3 with supervised learning, then collect rankings of model outputs and further "
                  "fine-tune using reinforcement learning from human feedback (RLHF). The resulting InstructGPT "
                  "models are preferred by labelers despite having 100x fewer parameters than GPT-3."),
        sections={
            "RLHF Pipeline": "Our pipeline has three steps: supervised fine-tuning on demonstrations, training a reward model on human preference comparisons between model outputs, and optimizing the policy against the reward model using PPO with a KL penalty toward the supervised policy to prevent reward over-optimization.",
            "Human Evaluation": "Outputs from the 1.3B InstructGPT model are preferred over the 175B GPT-3 baseline. Figure 1 shows human preference win-rate versus model size. Figure 2 shows the reward model score during PPO training along with the KL divergence from the initial policy.",
        },
        figures=[("human preference win-rate vs model size", "line"),
                 ("reward score and KL divergence during PPO training", "line")],
    ),
    dict(
        pid="2302.13971", title="LLaMA: Open and Efficient Foundation Language Models",
        abstract=("We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B "
                  "parameters. We train our models on trillions of tokens, and show that it is possible to train "
                  "state-of-the-art models using publicly available datasets exclusively. LLaMA-13B outperforms "
                  "GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, "
                  "Chinchilla-70B and PaLM-540B."),
        sections={
            "Training Data and Scaling": "Our training dataset is a mixture of CommonCrawl, C4, GitHub, Wikipedia, Books, arXiv and StackExchange totaling roughly 1.4T tokens. Following Chinchilla scaling laws, we train smaller models on more tokens than typical, optimizing for inference budget rather than training budget alone.",
            "Benchmark Results": "LLaMA-13B outperforms GPT-3 on most benchmarks despite being 10x smaller, making it feasible to run on a single GPU. Figure 1 shows training loss curves for the four model sizes. Figure 2 compares zero-shot accuracy on common sense reasoning benchmarks.",
        },
        figures=[("training loss curves for LLaMA model sizes", "line"),
                 ("zero-shot accuracy on reasoning benchmarks", "bar")],
    ),
    dict(
        pid="2407.01449", title="ColPali: Efficient Document Retrieval with Vision Language Models",
        abstract=("Documents are visually rich structures that convey information through text, but also figures, "
                  "page layouts, tables, and fonts. Standard retrieval pipelines struggle to exploit these visual "
                  "cues. We introduce ColPali, a retrieval model that leverages Vision Language Models to produce "
                  "high-quality multi-vector embeddings directly from document page images. Combined with a late "
                  "interaction matching mechanism, ColPali largely outperforms modern document retrieval pipelines "
                  "while being drastically faster at indexing time. We release the ViDoRe benchmark for visual "
                  "document retrieval evaluation."),
        sections={
            "Late Interaction over Page Images": "ColPali encodes each document page image into a grid of patch embeddings using a vision-language backbone. At query time, relevance is computed with a ColBERT-style MaxSim operator: each query token embedding is matched against all page patch embeddings and the maxima are summed. This preserves fine-grained spatial and visual information lost by single-vector approaches.",
            "ViDoRe Benchmark Results": "On the ViDoRe benchmark spanning tables, figures, infographics and multilingual documents, ColPali outperforms captioning-augmented text pipelines by a wide margin while removing the need for OCR and layout parsing. Figure 1 shows nDCG@5 across ViDoRe task categories. Figure 2 compares indexing latency per page against standard OCR-based pipelines.",
        },
        figures=[("nDCG@5 across ViDoRe task categories", "bar"),
                 ("indexing latency per page comparison", "bar")],
    ),
]

STYLES = getSampleStyleSheet()
TSTY = ParagraphStyle("T", parent=STYLES["Title"], fontSize=16, spaceAfter=8)
ASTY = ParagraphStyle("A", parent=STYLES["Normal"], fontSize=9, leading=12.5,
                      leftIndent=24, rightIndent=24, spaceAfter=10)
HSTY = ParagraphStyle("H", parent=STYLES["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
BSTY = ParagraphStyle("B", parent=STYLES["Normal"], fontSize=9.5, leading=13, spaceAfter=8)
CSTY = ParagraphStyle("C", parent=STYLES["Normal"], fontSize=8, leading=10,
                      alignment=1, spaceBefore=2, spaceAfter=12)

def make_fig(kind: str, caption: str, path: pathlib.Path):
    plt.figure(figsize=(4.6, 2.9), dpi=110)
    if kind == "heatmap":
        plt.imshow(rng.random((8, 8)), cmap="Blues"); plt.colorbar()
        plt.xlabel("key position"); plt.ylabel("query position")
    elif kind == "bar":
        labels = ["A", "B", "C", "D", "E"]
        plt.bar(labels, rng.uniform(40, 95, 5), color="#2563eb")
        plt.ylabel("score")
    elif kind == "line":
        x = np.arange(1, 21)
        for i in range(2):
            plt.plot(x, 60 + 30 * (1 - np.exp(-x / (4 + 4 * i))) + rng.normal(0, 1, 20),
                     marker="o", ms=2.5, label=f"variant {i+1}")
        plt.legend(fontsize=7); plt.xlabel("x"); plt.ylabel("metric")
    else:  # scatter
        plt.scatter(rng.uniform(0, 24, 60), rng.uniform(0, 120, 60), s=10, c="#2563eb", alpha=0.7)
        plt.xlabel("layer"); plt.ylabel("attention distance (px)")
    plt.title(caption, fontsize=8)
    plt.tight_layout(); plt.savefig(path); plt.close()

def build_pdf(p: dict):
    pdf_path = OUT / f"{p['pid']}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    story = [Paragraph(p["title"], TSTY),
             Paragraph("<b>Abstract.</b> " + p["abstract"], ASTY)]
    fig_iter = iter(enumerate(p["figures"], start=1))
    for i, (sec, body) in enumerate(p["sections"].items()):
        story.append(Paragraph(f"{i+1}. {sec}", HSTY))
        story.append(Paragraph(body, BSTY))
        try:
            fi, (caption, kind) = next(fig_iter)
            fpath = FIGTMP / f"{p['pid']}_fig{fi}.png"
            make_fig(kind, caption, fpath)
            story.append(RLImage(str(fpath), width=4.2 * inch, height=2.65 * inch))
            story.append(Paragraph(f"Figure {fi}: {caption}.", CSTY))
        except StopIteration:
            pass
    doc.build(story)
    print(f"built {pdf_path}")

if __name__ == "__main__":
    for p in PAPERS:
        build_pdf(p)
    print(f"\n{len(PAPERS)} sample papers in {OUT}/")
