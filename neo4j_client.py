"""
Neo4j HTTP client.

Communicates with Neo4j via the transactional HTTP endpoint so we never need
the Bolt driver (avoids auth/port-mapping issues in containerised setups).

Priority rules
--------------
• 85% of seed documents come from documents whose ID contains "BTC"
  (Bộ Tài Chính – Ministry of Finance).
• Prefer documents with a recent promulgationDate.
• Graph expansion follows AMENDS → MENTIONS → BASED_ON → REPLACES
  for up to KG_EXPANSION_HOPS hops; then adds sibling articles from
  the same document to fill MAX_LEGAL_UNITS slots.
"""

import base64
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
from loguru import logger

from config import (
    NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD,
    BTC_PRIORITY_RATIO, MAX_LEGAL_UNITS, KG_EXPANSION_HOPS,
)


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class LegalUnit:
    unit_id: str
    doc_id: str
    official_number: str
    doc_title: str
    heading: str
    content: str
    unit_type: str = ""

    @property
    def article_label(self) -> str:
        """Extract article/clause label from unit_id, e.g. 'Điều 5'.

        unit_id patterns: doc_Dieu5  doc_Khoan2  doc_Diem3  doc_Chuong1
        """
        patterns = [
            (r'_Dieu(\d+)',   "Điều"),
            (r'_Khoan(\d+)',  "Khoản"),
            (r'_Diem(\d+)',   "Điểm"),
            (r'_Chuong(\d+)', "Chương"),
            (r'_Muc(\d+)',    "Mục"),
        ]
        uid = self.unit_id
        for pat, label in patterns:
            m = re.search(pat, uid, re.IGNORECASE)
            if m:
                return f"{label} {m.group(1)}"
        # Fallback: last part after the last underscore if it looks like a number
        m = re.search(r'_(\d+)$', uid)
        if m:
            return f"Điều {m.group(1)}"
        return ""

    @property
    def law_reference(self) -> str:
        """Human-readable reference matching benchmark format: 'OfficialNum - Điều X'."""
        article = self.article_label
        if article:
            return f"{self.official_number} - {article}"
        return self.official_number

    @property
    def is_btc(self) -> bool:
        return "BTC" in self.doc_id.upper()


@dataclass
class LegalBlock:
    seed_unit: LegalUnit
    units: List[LegalUnit] = field(default_factory=list)

    @property
    def doc_ids(self) -> List[str]:
        return list({u.doc_id for u in self.units})

    @property
    def law_references(self) -> List[str]:
        return list({u.law_reference for u in self.units})

    @property
    def combined_text(self) -> str:
        parts = []
        for u in self.units:
            ref = f"{u.official_number} - {u.article_label}" if u.article_label else u.official_number
            header = f"[{ref}]"
            parts.append(f"{header}\n{u.content}")
        return "\n\n---\n\n".join(parts)

    @property
    def is_btc_focused(self) -> bool:
        btc_count = sum(1 for u in self.units if u.is_btc)
        return btc_count > len(self.units) / 2


# ── Client ───────────────────────────────────────────────────────────────────

class Neo4jClient:
    def __init__(self):
        creds = base64.b64encode(f"{NEO4J_USER}:{NEO4J_PASSWORD}".encode()).decode()
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {creds}",
        }
        self._tx_url = f"{NEO4J_URL}/db/neo4j/tx/commit"

    # ── low-level ────────────────────────────────────────────────────────────

    async def _run(self, query: str, params: dict | None = None) -> list:
        body = {"statements": [{"statement": query, "parameters": params or {}}]}
        async with aiohttp.ClientSession() as sess:
            async with sess.post(self._tx_url, headers=self._headers, json=body) as resp:
                data = await resp.json(content_type=None)
        errors = data.get("errors", [])
        if errors:
            raise RuntimeError(f"Neo4j error: {errors}")
        return data["results"][0]["data"]

    def _row(self, record) -> list:
        return record["row"]

    # ── seed sampling ────────────────────────────────────────────────────────

    async def _sample_doc_ids(self, btc: bool, n: int = 50) -> List[str]:
        """Return a pool of doc IDs to sample from (lightweight query)."""
        btc_filter = 'AND d.id CONTAINS "BTC"' if btc else ""
        q = f"""
        MATCH (d:LegalDocument)-[:CONTAINS]->(u:LegalUnit)
        WHERE u.content IS NOT NULL {btc_filter}
        WITH d, count(u) AS cnt
        WHERE cnt >= 3
        RETURN d.id, d.promulgationDate
        ORDER BY rand()
        LIMIT {n}
        """
        rows = await self._run(q)
        return [self._row(r)[0] for r in rows]

    async def _fetch_random_unit(self, doc_id: str) -> Optional[LegalUnit]:
        q = """
        MATCH (d:LegalDocument {id: $doc_id})-[:CONTAINS]->(u:LegalUnit)
        WHERE u.content IS NOT NULL
        WITH d, u ORDER BY rand() LIMIT 1
        RETURN d.id, d.officialNumber, d.title, u.id, u.heading, u.content, u.unitType
        """
        rows = await self._run(q, {"doc_id": doc_id})
        if not rows:
            return None
        r = self._row(rows[0])
        return LegalUnit(
            doc_id=r[0], official_number=r[1] or r[0],
            doc_title=r[2] or "", unit_id=r[3],
            heading=r[4] or "", content=r[5] or "", unit_type=r[6] or "",
        )

    async def sample_seed_unit(self) -> Optional[LegalUnit]:
        """Sample one seed LegalUnit with BTC-priority."""
        use_btc = random.random() < BTC_PRIORITY_RATIO
        doc_ids = await self._sample_doc_ids(btc=use_btc)
        if not doc_ids:
            doc_ids = await self._sample_doc_ids(btc=False)
        if not doc_ids:
            return None
        for _ in range(5):
            doc_id = random.choice(doc_ids)
            unit = await self._fetch_random_unit(doc_id)
            if unit:
                return unit
        return None

    # ── graph expansion ───────────────────────────────────────────────────────

    async def _expand_from_doc(self, doc_id: str, exclude_ids: set, limit: int) -> List[LegalUnit]:
        """
        Follow document-level relationships to nearby documents, then pull
        one representative article from each neighbour.
        """
        q = """
        MATCH (d:LegalDocument {id: $doc_id})-[r:AMENDS|MENTIONS|BASED_ON|REPLACES|ABOLISHED_BY]->(d2:LegalDocument)
        WHERE d2.id <> $doc_id
        WITH d2 ORDER BY rand() LIMIT $limit
        MATCH (d2)-[:CONTAINS]->(u:LegalUnit)
        WHERE u.content IS NOT NULL
        WITH d2, u ORDER BY rand() LIMIT 1
        RETURN d2.id, d2.officialNumber, d2.title, u.id, u.heading, u.content, u.unitType
        """
        rows = await self._run(q, {"doc_id": doc_id, "limit": limit * 3})
        units = []
        for r in rows:
            rv = self._row(r)
            if rv[3] in exclude_ids:
                continue
            units.append(LegalUnit(
                doc_id=rv[0], official_number=rv[1] or rv[0],
                doc_title=rv[2] or "", unit_id=rv[3],
                heading=rv[4] or "", content=rv[5] or "", unit_type=rv[6] or "",
            ))
            exclude_ids.add(rv[3])
            if len(units) >= limit:
                break
        return units

    async def _sibling_units(self, doc_id: str, exclude_ids: set, limit: int) -> List[LegalUnit]:
        """Pull sibling articles from the same document."""
        q = """
        MATCH (d:LegalDocument {id: $doc_id})-[:CONTAINS]->(u:LegalUnit)
        WHERE u.content IS NOT NULL
        WITH d, u ORDER BY u.order LIMIT $limit
        RETURN d.id, d.officialNumber, d.title, u.id, u.heading, u.content, u.unitType
        """
        rows = await self._run(q, {"doc_id": doc_id, "limit": limit * 4})
        units = []
        for r in rows:
            rv = self._row(r)
            if rv[3] in exclude_ids:
                continue
            units.append(LegalUnit(
                doc_id=rv[0], official_number=rv[1] or rv[0],
                doc_title=rv[2] or "", unit_id=rv[3],
                heading=rv[4] or "", content=rv[5] or "", unit_type=rv[6] or "",
            ))
            exclude_ids.add(rv[3])
            if len(units) >= limit:
                break
        return units

    async def build_legal_block(self, seed: LegalUnit) -> LegalBlock:
        """
        Expand the seed into a LegalBlock of up to MAX_LEGAL_UNITS articles.

        Strategy:
        1. Seed unit is always included.
        2. Add sibling articles from the same document (fill ~half the slots).
        3. Follow document relationships outward (1-2 hops) to fill remaining slots.
        """
        seen_ids: set = {seed.unit_id}
        block_units: List[LegalUnit] = [seed]

        sibling_quota = max(1, MAX_LEGAL_UNITS // 2)
        expand_quota  = MAX_LEGAL_UNITS - 1 - sibling_quota

        siblings = await self._sibling_units(seed.doc_id, seen_ids, sibling_quota)
        block_units.extend(siblings)

        # Graph expansion (hops)
        frontier_docs = {seed.doc_id}
        for _ in range(min(KG_EXPANSION_HOPS, 2)):
            if len(block_units) >= MAX_LEGAL_UNITS:
                break
            needed = MAX_LEGAL_UNITS - len(block_units)
            for doc_id in list(frontier_docs):
                extra = await self._expand_from_doc(doc_id, seen_ids, needed)
                block_units.extend(extra)
                frontier_docs.update(u.doc_id for u in extra)
                if len(block_units) >= MAX_LEGAL_UNITS:
                    break

        return LegalBlock(seed_unit=seed, units=block_units[:MAX_LEGAL_UNITS])

    # ── document-schema queries (for tasks 1.5, 2.3) ─────────────────────────

    async def get_doc_relationships(self, doc_id: str) -> List[dict]:
        """Return AMENDS / REPLACES / BASED_ON relationships for a doc."""
        q = """
        MATCH (d:LegalDocument {id: $doc_id})-[r]->(d2:LegalDocument)
        RETURN type(r) AS rel, d.officialNumber AS from_num,
               d2.id AS to_id, d2.officialNumber AS to_num, d2.title AS to_title
        LIMIT 10
        """
        rows = await self._run(q, {"doc_id": doc_id})
        return [
            {
                "relation": self._row(r)[0],
                "from": self._row(r)[1],
                "to_id": self._row(r)[2],
                "to_num": self._row(r)[3],
                "to_title": self._row(r)[4],
            }
            for r in rows
        ]
