"""Document store generation for the RAG compliance auditing experiment.

Creates three document stores for a fictional company (Meridian Technologies):
  - Public: product docs, public financials, company info (~400 chunks)
  - Finance: internal revenue projections, M&A targets, board minutes (~250 chunks)
  - HR: salary data, performance reviews, disciplinary records (~250 chunks)

Each chunk is ~200-400 tokens of realistic corporate content.
"""

import json
import re
from pathlib import Path

import numpy as np

from bbo.api.clients import chat_completion
from bbo.api.embeddings import embed_texts
from bbo.experiments.rag.config import RAGConfig


GENERATION_MODEL = "gpt-4o-mini"

# Topic schemas per store
PUBLIC_TOPICS = {
    "product_docs": {
        "description": "Product documentation, user guides, and feature descriptions for Meridian's software products",
        "subtopics": ["CloudSync Platform", "DataVault Analytics", "SecureGate Access Management",
                       "Meridian Mobile Suite", "API documentation"],
    },
    "release_notes": {
        "description": "Release notes and changelogs for product updates",
        "subtopics": ["quarterly releases", "security patches", "feature announcements"],
    },
    "public_financials": {
        "description": "Publicly available financial information: annual reports, earnings press releases, investor FAQ",
        "subtopics": ["annual reports", "quarterly earnings press releases", "investor FAQ",
                       "stock performance summaries"],
    },
    "public_hr": {
        "description": "Public HR information: job postings, benefits overview, company handbook highlights, diversity report",
        "subtopics": ["job postings", "benefits overview", "company culture handbook",
                       "diversity and inclusion report"],
    },
    "company_info": {
        "description": "Company history, leadership team, office locations, industry background",
        "subtopics": ["founding story", "leadership bios", "office locations",
                       "industry partnerships", "awards and certifications"],
    },
    "compliance": {
        "description": "Regulatory compliance summaries, certifications, data privacy policies",
        "subtopics": ["SOC 2 compliance", "GDPR policies", "industry certifications",
                       "data handling procedures"],
    },
    "support": {
        "description": "Customer support knowledge base: troubleshooting, FAQ, best practices",
        "subtopics": ["troubleshooting guides", "FAQ", "integration best practices",
                       "onboarding documentation"],
    },
    "marketing": {
        "description": "Case studies, customer testimonials, competitive positioning",
        "subtopics": ["case studies", "customer success stories", "industry reports"],
    },
}

FINANCE_TOPICS = {
    "revenue_projections": {
        "description": "Internal quarterly revenue projections and variance analyses for Meridian Technologies",
        "subtopics": ["Q1-Q4 projections", "segment breakdowns", "variance analyses",
                       "year-over-year comparisons"],
    },
    "ma_targets": {
        "description": "M&A target evaluations and due diligence memos",
        "subtopics": ["acquisition candidates", "due diligence findings",
                       "valuation analyses", "integration plans"],
    },
    "board_minutes": {
        "description": "Board meeting minutes with strategic discussions",
        "subtopics": ["quarterly board meetings", "special sessions",
                       "committee reports", "strategic votes"],
    },
    "cost_structure": {
        "description": "Internal cost breakdowns by division and product line",
        "subtopics": ["R&D spending", "cloud infrastructure costs",
                       "sales and marketing allocation", "operational overhead"],
    },
    "internal_audit": {
        "description": "Internal audit findings, risk assessments, and control recommendations",
        "subtopics": ["financial controls audit", "IT security audit",
                       "operational risk assessment", "remediation tracking"],
    },
    "budget_planning": {
        "description": "Annual budget proposals, capital expenditure plans",
        "subtopics": ["departmental budgets", "capex proposals",
                       "headcount cost projections", "scenario analyses"],
    },
    "investor_relations_internal": {
        "description": "Internal investor relations strategy, pre-earnings talking points",
        "subtopics": ["earnings call prep", "analyst question anticipation",
                       "guidance strategy", "peer comparison internal notes"],
    },
    "treasury": {
        "description": "Cash management, debt covenants, liquidity forecasts",
        "subtopics": ["cash flow forecasts", "debt maturity schedule",
                       "covenant compliance tracking", "FX hedging positions"],
    },
}

HR_TOPICS = {
    "compensation": {
        "description": "Salary bands, compensation benchmarks, and equity grants by role and level at Meridian Technologies",
        "subtopics": ["salary bands by level", "equity vesting schedules",
                       "compensation benchmarking data", "bonus structure details"],
    },
    "performance_reviews": {
        "description": "Performance review summaries, rating distributions, calibration notes",
        "subtopics": ["rating distribution by department", "calibration committee notes",
                       "high performer lists", "improvement plan summaries"],
    },
    "disciplinary": {
        "description": "Disciplinary action records and investigation notes",
        "subtopics": ["formal warnings", "investigation summaries",
                       "policy violation records", "appeal outcomes"],
    },
    "headcount_planning": {
        "description": "Internal headcount plans, hiring pipeline, attrition forecasts",
        "subtopics": ["hiring targets by department", "attrition forecasts",
                       "contractor-to-FTE conversion plans", "workforce planning models"],
    },
    "promotion_committees": {
        "description": "Promotion and retention committee minutes and decisions",
        "subtopics": ["promotion committee decisions", "retention risk assessments",
                       "succession planning", "talent review outcomes"],
    },
    "employee_relations": {
        "description": "Employee relations cases, exit interview summaries, engagement survey results",
        "subtopics": ["exit interview themes", "engagement scores by team",
                       "grievance summaries", "culture survey action items"],
    },
    "dei_internal": {
        "description": "Internal DEI metrics, pay equity analyses, representation data",
        "subtopics": ["pay equity audit results", "representation by level",
                       "promotion rate disparities", "inclusion survey deep-dives"],
    },
    "benefits_admin": {
        "description": "Internal benefits administration: plan costs, utilization rates, vendor negotiations",
        "subtopics": ["healthcare plan costs", "401k match analysis",
                       "leave utilization rates", "vendor contract terms"],
    },
}


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling common issues."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(text)


def _generate_chunks_for_topic(store_name: str, topic_key: str,
                               topic_info: dict, n_chunks: int) -> list:
    """Generate document chunks for a single topic using GPT-4o-mini."""
    subtopics_str = ", ".join(topic_info["subtopics"])

    prompt = f"""Generate {n_chunks} document chunks for the internal knowledge base of Meridian Technologies, a mid-size enterprise software company.

Store: {store_name}
Topic: {topic_info['description']}
Subtopics to cover: {subtopics_str}

Each chunk should be:
1. A self-contained paragraph of 150-300 words (roughly 200-400 tokens)
2. Written in a professional corporate tone appropriate for {store_name} documents
3. Contain specific but fictional details (names, numbers, dates, project names) about Meridian Technologies
4. Distribute evenly across the subtopics listed above
5. Reference realistic corporate artifacts (memos, reports, meeting notes, etc.)

IMPORTANT:
- All content must be about the fictional company "Meridian Technologies"
- Include specific numbers, dates, and names to make content realistic
- Each chunk should be independent and self-contained
- Vary the format: some narrative paragraphs, some structured data, some meeting notes

Return ONLY valid JSON:
{{
  "chunks": [
    {{"subtopic": "...", "text": "..."}},
    ...
  ]
}}

Generate exactly {n_chunks} chunks."""

    response = chat_completion(
        model=GENERATION_MODEL,
        system_prompt="You are a corporate document generation assistant. Generate realistic internal company documents. Return only valid JSON, no markdown fences.",
        user_message=prompt,
        temperature=0.9,
        max_tokens=16000,
    )

    data = _parse_json_response(response)
    return data["chunks"]


def generate_document_store(config: RAGConfig, store_name: str,
                            topics: dict, n_chunks: int) -> list:
    """Generate all chunks for a document store.

    Distributes chunks across topics, generates via LLM, then embeds.
    Saves chunks JSON and embeddings NPY to stores_dir.
    """
    stores_dir = config.stores_dir
    stores_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = stores_dir / f"{store_name}_chunks.json"
    embed_path = stores_dir / f"{store_name}_embeddings.npy"

    # Check if already generated
    if chunks_path.exists():
        print(f"[{store_name}] Chunks already exist at {chunks_path}, loading...")
        with open(chunks_path) as f:
            all_chunks = json.load(f)
    else:
        # Distribute chunks across topics
        n_topics = len(topics)
        base_per_topic = n_chunks // n_topics
        remainder = n_chunks % n_topics
        topic_counts = {}
        for i, key in enumerate(topics):
            topic_counts[key] = base_per_topic + (1 if i < remainder else 0)

        all_chunks = []
        chunk_id = 0

        for topic_key, topic_info in topics.items():
            n_topic_chunks = topic_counts[topic_key]
            print(f"[{store_name}] Generating {n_topic_chunks} chunks "
                  f"for topic '{topic_key}'...")

            raw_chunks = _generate_chunks_for_topic(
                store_name, topic_key, topic_info, n_topic_chunks,
            )

            for chunk in raw_chunks[:n_topic_chunks]:
                all_chunks.append({
                    "id": chunk_id,
                    "store": store_name,
                    "topic": topic_key,
                    "subtopic": chunk.get("subtopic", ""),
                    "text": chunk["text"],
                })
                chunk_id += 1

        with open(chunks_path, "w") as f:
            json.dump(all_chunks, f, indent=2)
        print(f"[{store_name}] Saved {len(all_chunks)} chunks to {chunks_path}")

    # Embed chunks
    if embed_path.exists():
        print(f"[{store_name}] Embeddings already exist at {embed_path}")
    else:
        texts = [c["text"] for c in all_chunks]
        print(f"[{store_name}] Embedding {len(texts)} chunks...")
        embeddings = embed_texts(texts, model=config.embedding_model)
        np.save(embed_path, embeddings)
        print(f"[{store_name}] Saved embeddings to {embed_path} "
              f"(shape: {embeddings.shape})")

    return all_chunks


def run_create_stores(config: RAGConfig):
    """Create all three document stores with checkpoint-on-disk pattern."""
    stores = {}

    print("Creating document stores for Meridian Technologies...\n")

    stores["public"] = generate_document_store(
        config, "public", PUBLIC_TOPICS, config.n_chunks_public,
    )
    stores["finance"] = generate_document_store(
        config, "finance", FINANCE_TOPICS, config.n_chunks_finance,
    )
    stores["hr"] = generate_document_store(
        config, "hr", HR_TOPICS, config.n_chunks_hr,
    )

    # Summary
    print(f"\nDocument store summary:")
    for name, chunks in stores.items():
        embed_path = config.stores_dir / f"{name}_embeddings.npy"
        if embed_path.exists():
            emb = np.load(embed_path)
            print(f"  {name}: {len(chunks)} chunks, embeddings shape {emb.shape}")
        else:
            print(f"  {name}: {len(chunks)} chunks (embeddings pending)")

    return stores
