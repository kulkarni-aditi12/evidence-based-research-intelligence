# Evidence-Based Research Intelligence System

An evidence-based research intelligence system that uses Gemini, RAG, and a multi-agent LangGraph pipeline to analyze research documents and generate structured, evidence-grounded research reports.

## Overview

Traditional LLM-based systems can generate answers without clearly showing where the information came from.

This project addresses that problem by building a research pipeline where the system:

- Understands the research question
- Breaks it into smaller subquestions
- Retrieves relevant evidence from research documents
- Analyzes whether claims are actually supported
- Detects agreement and conflicts between evidence
- Generates a research draft
- Critiques the draft for unsupported claims and hallucination risk
- Produces a final evidence-grounded research report
- Stores previous research results in research memory

The goal is to make research generation more **evidence-based, traceable, and reliable**.

---

## System Architecture

```text
                    Research Question
                           |
                           v
                  +-------------------+
                  | Research Planner  |
                  +-------------------+
                           |
                           v
                  +-------------------+
                  | Evidence Retrieval|
                  +-------------------+
                           |
                           v
                  +-------------------+
                  | Evidence Analyzer |
                  +-------------------+
                           |
                           v
                  +-------------------+
                  | Conflict Detector |
                  +-------------------+
                           |
                           v
                  +-------------------+
                  |  Draft Generator  |
                  +-------------------+
                           |
                           v
                  +-------------------+
                  |   Critic Agent    |
                  +-------------------+
                           |
                           v
                  +-------------------+
                  | Final Synthesis   |
                  +-------------------+
                           |
                           v
                  +-------------------+
                  | Research Memory  |
                  +-------------------+
                           |
                           v
                 Final Research Report
Key Features
1. Research Question Decomposition

The Research Planner Agent breaks a complex research question into smaller subquestions.

This allows the system to perform more focused evidence retrieval instead of searching for the entire question at once.

2. PDF Processing

Research papers are processed using PyMuPDF.

The system extracts text from individual pages while preserving page numbers for later citation.

3. Intelligent Chunking

Extracted document text is divided into smaller overlapping chunks using a recursive text splitter.

This improves semantic retrieval by allowing the system to retrieve specific sections of a document rather than the entire paper.

4. Gemini Embeddings

Document chunks are converted into vector embeddings using Gemini Embeddings.

These embeddings represent the semantic meaning of the text.

5. ChromaDB Vector Database

The generated embeddings are stored in ChromaDB.

When a research question is submitted, the system performs semantic similarity search to retrieve the most relevant evidence.

6. Retrieval-Augmented Generation

The retrieved evidence is passed through the research pipeline instead of allowing the LLM to answer purely from its general knowledge.

This helps keep the generated report grounded in the supplied research documents.

7. Evidence Analysis

The Evidence Analyzer identifies important claims and determines whether they are:

Directly Supported
Partially Supported
Not Supported

It also assigns an evidence strength:

Strong
Moderate
Weak
8. Conflict Detection

The Conflict Detection Agent compares retrieved evidence and classifies relationships as:

Agreement
Conflict
Uncertain

The system does not automatically treat different values or statements as contradictions. It considers whether differences could result from different datasets, populations, experiments, conditions, or stages.

9. Critic Agent

Before the final report is generated, the Critic Agent evaluates the draft for:

Unsupported claims
Missing evidence
Citation issues
Contradictions
Completeness
Hallucination risk

It can mark the draft as:

PASS

or

NEEDS REVISION
10. Final Evidence-Based Synthesis

The Final Synthesis Agent combines the research plan, evidence analysis, conflict analysis, draft, and critic feedback into a structured research report.

11. Research Memory

Previous research questions, research plans, and final answers are stored in research memory.

This allows the system to identify potentially relevant previous research.

12. Page-Level Sources

Important claims can be associated with the page number from which the evidence was retrieved.

This improves traceability and makes the generated report easier to verify.

Multi-Agent Pipeline

The system contains multiple specialized agents.

Agent 1 — Research Planner

Input: Research question

Output: Research subquestions

Breaks the original research question into smaller questions that can be investigated independently.

Agent 2 — Evidence Retrieval

Input: Research subquestions

Output: Relevant document chunks

Uses Gemini embeddings and ChromaDB semantic similarity search to retrieve relevant evidence.

Agent 3 — Evidence Analyzer

Input: Research question + retrieved evidence

Output: Evidence-supported claims

Determines how strongly the retrieved evidence supports important claims.

Agent 4 — Conflict Detector

Input: Retrieved evidence

Output: Agreement / Conflict / Uncertain relationships

Identifies meaningful relationships between pieces of evidence.

Agent 5 — Draft Generator

Input: Research question + retrieved evidence

Output: Initial research answer

Creates an initial evidence-grounded draft.

Agent 6 — Critic

Input: Draft + evidence

Output: Critical evaluation

Checks the draft for unsupported claims, missing evidence, citation problems, contradictions, and hallucination risk.

Agent 7 — Final Synthesis

Input: All validated research information

Output: Final research report

Produces the final structured answer while following the evidence and critic feedback.

Agent 8 — Research Memory

Stores the completed research question, research plan, and final answer for future reference.

RAG Pipeline
Research PDF
     |
     v
PyMuPDF
     |
     v
Text Extraction
     |
     v
Text Chunking
     |
     v
Gemini Embeddings
     |
     v
ChromaDB
     |
     v
Semantic Retrieval
     |
     v
Retrieved Evidence
     |
     v
Multi-Agent Research Pipeline
     |
     v
Final Research Report
Tech Stack
Technology	Purpose
Python	Core development
Gemini	Large Language Model
LangChain	LLM and retrieval integration
LangGraph	Multi-agent workflow orchestration
Gemini Embeddings	Semantic text embeddings
ChromaDB	Vector database
RAG	Evidence-grounded generation
PyMuPDF	PDF text extraction
Recursive Character Text Splitter	Document chunking
Project Structure
research_agent/
│
├── data/
│   └── research.pdf
│
├── src/
│   ├── rag.py
│   ├── evidence_analyzer.py
│   ├── conflict_detector.py
│   ├── critic_agent.py
│   ├── synthesis_agent.py
│   ├── research_memory.py
│   ├── create_vector_db.py
│   ├── test_retrieval.py
│   ├── load_pdf.py
│   └── chunk_pdf.py
│
├── requirements.txt
├── .gitignore
├── README.md
└── .env

.env, chroma_db/, research_memory.json, and Python cache files are excluded from version control using .gitignore.
Preparing the Vector Database

Place a research PDF inside the data/ directory.

Then run:

python src/create_vector_db.py

This will:

Load the PDF
Extract page text
Split the text into chunks
Generate Gemini embeddings
Store the embeddings in ChromaDB

The generated ChromaDB directory is intentionally ignored by Git.

Testing Retrieval

To test semantic retrieval independently:

python src/test_retrieval.py

Enter a research question when prompted.

The system will display the most relevant document chunks along with their page numbers.

Running the Complete Research System

After creating the vector database:

python src/rag.py

Enter a research question.

The LangGraph pipeline will execute the complete workflow:

Planner
   ↓
Retrieval
   ↓
Evidence Analysis
   ↓
Conflict Detection
   ↓
Draft Generation
   ↓
Critic
   ↓
Final Synthesis
   ↓
Research Memory
