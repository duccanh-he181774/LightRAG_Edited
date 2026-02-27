#!/usr/bin/env python3
"""
KG Post-Processing — Duplicate Entity Merger for LightRAG (PostgreSQL backend)
Variant: Phase 2 uses NAME-ONLY embedding (re-embeds entity_name only, not content_vector).

This avoids noise from description-dominated content_vector at low thresholds.
bge-m3 is multilingual so "Question Content" ↔ "Nội dung câu hỏi" are caught naturally.

Detects entity nodes that represent the same concept but differ in casing,
punctuation, or spelling, then merges them into a single canonical node.

Usage:
    # Step 1 — see what would be merged (safe, no DB changes):
    python kg_dedup.py --workspace "" --env .env

    # Step 2 — apply the merges:
    python kg_dedup.py --workspace "" --env .env --apply

    # Target old KG (default workspace = empty string):
    python kg_dedup.py --workspace ""

    # Target new KG:
    python kg_dedup.py --workspace "v2_fixed_prompt"
"""

import asyncio
import argparse
import json
import os
import re
import sys
from collections import defaultdict

# Force UTF-8 output on Windows (prevents UnicodeEncodeError for Vietnamese text)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class _Tee:
    """Write to both the original stream and a log file simultaneously."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

try:
    import asyncpg
except ImportError:
    print("asyncpg not installed. Run: pip install asyncpg")
    sys.exit(1)

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False


def _strip_code_separator(s: str) -> str:
    """
    Normalise the separator between a code prefix and its name.
    'BR 2 - Foo' / 'BR 2 – Foo' / 'BR 2: Foo' / 'BR 2 Foo'
    → all become 'BR 2 Foo' for comparison.
    """
    return re.sub(
        r'^([A-Za-z]{1,5}\s?\d+[\w-]*)\s*[-–:]\s*',
        r'\1 ',
        s,
    )


def to_lookup_key(name: str) -> str:
    """
    Collapse a name to a normalised key used only for duplicate detection.
    Two names with the same key are considered duplicates.

    Key transformations:
    - Lowercase + collapse whitespace
    - Strip trailing parenthetical translation: "System (Hệ thống)" → "system"
      so bilingual "X (Y)" variants merge with plain "X"
    - Normalise code separators: "BR 2 - Foo" / "BR 2: Foo" → same key
    - Hyphen = space
    """
    s = name.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    # Strip trailing parenthetical ONLY if it contains at least one non-ASCII char
    # (= Vietnamese translation annotation), e.g.:
    #   "System (Hệ thống)" → "system"   (Vietnamese inside parens → strip)
    #   "Audit Trail (CBR1)" → unchanged  (all-ASCII inside parens → keep)
    s = re.sub(r'\s*\((?=[^)]*[^\x00-\x7F])[^)]*\)\s*$', '', s).strip()
    s = _strip_code_separator(s)        # normalise code separators
    s = s.replace('-', ' ')             # hyphen = space for comparison
    s = re.sub(r'\s+', ' ', s).strip()
    # Plural normalisation: strip trailing 's' unless preceded by a vowel or 's'
    # Catches: "records"→"record", "screens"→"screen", "groups"→"group"
    # Safe against: "status" (u+s), "class" (s+s), "process" (s+s)
    if len(s) > 2 and s[-1] == 's' and s[-2] not in 'aeiou s':
        s = s[:-1]
    return s


def _code_format(name: str) -> str:
    """
    Reformat 'CODE separator Name' → 'CODE: Name'.
    Leaves non-code names unchanged.
    """
    m = re.match(
        r'^([A-Z]{1,5}\s?\d+[\w-]*)\s*[-–:\s]\s*(.+)$',
        name.strip(),
    )
    if m:
        code, rest = m.group(1).strip(), m.group(2).strip()
        return f"{code}: {rest}"
    return name


def _title_score(name: str) -> int:
    """Count capitalised significant words (rough Title Case score)."""
    minor = {'a', 'an', 'the', 'of', 'in', 'at', 'for', 'on', 'to', 'and', 'or', 'but'}
    words = name.split()
    return sum(
        1 for i, w in enumerate(words)
        if w and w[0].isupper() and (i == 0 or w.lower() not in minor)
    )


def _code_quality_score(name: str) -> int:
    """
    Count ALL-CAPS code tokens (2–5 uppercase letters, optionally followed by digits).
    Prefers "CFD" over "Cfd", "UI" over "Ui", etc.
    Mixed-case tokens (e.g. 'Cfd') do NOT match, so they score 0.
    """
    return len(re.findall(r'\b[A-Z]{2,5}\d*\b', name))


def pick_canonical(names: list[str]) -> str:
    """
    Given a list of duplicate entity names, return the best canonical form.

    Priority:
    1. 'CODE: Name' format (e.g. 'BR 1: Rule Name')
    2. Most ALL-CAPS code tokens  (CFD > Cfd)
    3. Highest Title Case score
    4. Longest name (most information)
    """
    code_re = re.compile(r'^[A-Z]{1,5}\s?\d+[\w-]*:\s+.+$')
    code_names = [n for n in names if code_re.match(n)]
    if code_names:
        best = max(
            code_names,
            key=lambda n: (_code_quality_score(n), _title_score(n), len(n)),
        )
        return _code_format(best)

    return max(names, key=lambda n: (_code_quality_score(n), _title_score(n), len(n)))


def find_duplicate_groups(entities: list[dict]) -> list[dict]:
    """
    Group entities by normalised lookup key.
    Catches rule-based duplicates: casing, punctuation, hyphen/space, code separators.
    Returns only groups with 2+ members.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for ent in entities:
        key = to_lookup_key(ent["entity_name"])
        buckets[key].append(ent)

    groups = []
    for key, group in buckets.items():
        if len(group) > 1:
            canonical = pick_canonical([e["entity_name"] for e in group])
            groups.append({"key": key, "canonical": canonical, "entities": group})

    return sorted(groups, key=lambda g: g["key"])


# ============================================================================
# PostgreSQL helpers
# ============================================================================

# ---- AGE graph helpers -----------------------------------------------------

async def discover_age_graphs(conn: asyncpg.Connection, workspace: str) -> list[str]:
    """
    Return AGE graph names that belong to this workspace.
    AGE stores graph metadata in ag_catalog.ag_graph.
    Graph names follow the pattern: '{workspace}_{NAMESPACE}' or just '{NAMESPACE}'.
    """
    try:
        rows = await conn.fetch("SELECT name FROM ag_catalog.ag_graph")
        all_graphs = [r["name"] for r in rows]
    except Exception:
        return []   # AGE not installed or ag_catalog not accessible

    if workspace:
        prefix = f"{workspace}_"
        return [g for g in all_graphs if g.startswith(prefix)]
    else:
        # Default workspace: graph names have no workspace prefix
        # (they're just namespace names like "CHUNK_DESCRIPTION")
        return all_graphs


async def age_setup(conn: asyncpg.Connection) -> bool:
    """Load AGE extension on this connection. Returns True if successful."""
    try:
        await conn.execute("LOAD 'age'")
        await conn.execute('SET search_path = ag_catalog, "$user", public')
        return True
    except Exception:
        return False


async def age_merge_nodes(
    conn: asyncpg.Connection,
    graph_name: str,
    keep_name: str,
    drop_name: str,
) -> None:
    """
    In the AGE graph:
      1. Find all edges connected to drop_name node.
      2. Re-create them on keep_name node.
      3. DETACH DELETE drop_name node.
    """
    # Escape for Cypher string literals
    keep_esc = keep_name.replace("\\", "\\\\").replace('"', '\\"')
    drop_esc = drop_name.replace("\\", "\\\\").replace('"', '\\"')

    # Re-attach outgoing edges: (drop)-[r]->(other)  →  (keep)-[r]->(other)
    redirect_out = f"""
        SELECT * FROM cypher('{graph_name}', $$
            MATCH (d:base {{entity_id: "{drop_esc}"}})-[r:DIRECTED]->(other:base)
            MATCH (k:base {{entity_id: "{keep_esc}"}})
            MERGE (k)-[:DIRECTED {{
                weight: r.weight,
                description: r.description,
                keywords: r.keywords,
                source_id: r.source_id,
                file_path: r.file_path
            }}]->(other)
        $$) AS (result agtype)
    """
    # Re-attach incoming edges: (other)-[r]->(drop)  →  (other)-[r]->(keep)
    redirect_in = f"""
        SELECT * FROM cypher('{graph_name}', $$
            MATCH (other:base)-[r:DIRECTED]->(d:base {{entity_id: "{drop_esc}"}})
            MATCH (k:base {{entity_id: "{keep_esc}"}})
            MERGE (other)-[:DIRECTED {{
                weight: r.weight,
                description: r.description,
                keywords: r.keywords,
                source_id: r.source_id,
                file_path: r.file_path
            }}]->(k)
        $$) AS (result agtype)
    """
    # Delete drop node (DETACH removes its remaining edges)
    detach_delete = f"""
        SELECT * FROM cypher('{graph_name}', $$
            MATCH (d:base {{entity_id: "{drop_esc}"}})
            DETACH DELETE d
        $$) AS (result agtype)
    """
    try:
        await conn.execute(redirect_out)
        await conn.execute(redirect_in)
        await conn.execute(detach_delete)
    except Exception as e:
        print(f"         [!] AGE update failed for graph '{graph_name}': {e}")


# ---- NetworkX GraphML helpers -----------------------------------------------

def discover_graphml_files(env_path: str, workspace: str) -> list[str]:
    """
    Find GraphML files for this workspace.
    LightRAG stores them at {WORKING_DIR}/{workspace}/graph_*.graphml
    or {WORKING_DIR}/graph_*.graphml for the default workspace.
    """
    load_dotenv(env_path, override=True)
    working_dir = os.getenv("WORKING_DIR", "./rag_storage")

    candidates = []
    if workspace:
        ws_dir = os.path.join(working_dir, workspace)
        if os.path.isdir(ws_dir):
            for f in os.listdir(ws_dir):
                if f.endswith(".graphml"):
                    candidates.append(os.path.join(ws_dir, f))
    else:
        if os.path.isdir(working_dir):
            for f in os.listdir(working_dir):
                if f.endswith(".graphml"):
                    candidates.append(os.path.join(working_dir, f))
    return candidates


def graphml_merge_nodes(graphml_path: str, keep_name: str, drop_name: str) -> bool:
    """
    Update the GraphML file: redirect all edges from drop_name → keep_name,
    then remove drop_name node.
    Returns True if any change was made.
    """
    try:
        import networkx as nx
    except ImportError:
        print("         [!] networkx not installed -- GraphML update skipped.")
        return False

    G = nx.read_graphml(graphml_path)
    if drop_name not in G:
        return False

    changed = False
    # Redirect edges (undirected graph in LightRAG's NetworkX)
    for neighbor in list(G.neighbors(drop_name)):
        if neighbor == keep_name:
            continue
        edge_data = dict(G.edges[drop_name, neighbor])
        if not G.has_edge(keep_name, neighbor):
            G.add_edge(keep_name, neighbor, **edge_data)

    G.remove_node(drop_name)
    nx.write_graphml(G, graphml_path)
    return True


async def make_connection(env_path: str) -> asyncpg.Connection:
    load_dotenv(env_path, override=True)
    return await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "pass"),
        database=os.getenv("POSTGRES_DATABASE", "lightrag"),
    )


async def discover_tables(conn: asyncpg.Connection) -> tuple[str, str]:
    """
    Auto-discover the actual LIGHTRAG_VDB_ENTITY and LIGHTRAG_VDB_RELATION table names.
    LightRAG appends an embedding model suffix, e.g.:
      LIGHTRAG_VDB_ENTITY_openai_text_embedding_3_small_1536d
    Returns (entity_table, relation_table).
    """
    rows = await conn.fetch(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name ILIKE 'lightrag_vdb_entity%'
        ORDER BY table_name
        """
    )
    entity_candidates = [r["table_name"] for r in rows]

    rows2 = await conn.fetch(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name ILIKE 'lightrag_vdb_relation%'
        ORDER BY table_name
        """
    )
    relation_candidates = [r["table_name"] for r in rows2]

    if not entity_candidates or not relation_candidates:
        all_tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name"
        )
        names = [r["table_name"] for r in all_tables]
        raise RuntimeError(
            f"Could not find LIGHTRAG_VDB_ENTITY/RELATION tables.\n"
            f"Tables in public schema: {names}"
        )

    # Prefer the suffixed version (longer name) over the base name
    entity_table = max(entity_candidates, key=len)
    relation_table = max(relation_candidates, key=len)
    return entity_table, relation_table


def _workspace_clause(workspace: str, param_offset: int = 1) -> tuple[str, list]:
    """
    Return (SQL clause, params) for filtering by workspace.
    Default workspace (empty string OR 'default') matches '', 'default', and NULL rows.
    LightRAG stores 'default' as the workspace value when no WORKSPACE env var is set.
    """
    if workspace and workspace.lower() != "default":
        return f"workspace = ${param_offset}", [workspace]
    else:
        # Default workspace: match '', 'default', or NULL
        return "( workspace = '' OR workspace = 'default' OR workspace IS NULL )", []


async def fetch_entities(
    conn: asyncpg.Connection, workspace: str, entity_table: str
) -> list[dict]:
    ws_clause, ws_params = _workspace_clause(workspace)
    rows = await conn.fetch(
        f"""
        SELECT id, entity_name, content, chunk_ids, file_path
        FROM "{entity_table}"
        WHERE {ws_clause}
        ORDER BY entity_name
        """,
        *ws_params,
    )
    return [dict(r) for r in rows]


# ---- Relation key-type detection ------------------------------------------------
# LightRAG versions differ: some store entity UUIDs in source_id/target_id,
# others store entity_name strings.  Detect once and cache for the process lifetime.

_RELATION_KEY_TYPE: str | None = None   # 'uuid' | 'name'


async def _get_relation_key_type(conn: asyncpg.Connection, relation_table: str) -> str:
    """
    Detect whether source_id / target_id in the relation table hold UUIDs or entity names.
    Result is cached globally for the lifetime of the process.
    """
    global _RELATION_KEY_TYPE
    if _RELATION_KEY_TYPE is not None:
        return _RELATION_KEY_TYPE
    try:
        sample = await conn.fetch(
            f'SELECT source_id FROM "{relation_table}" LIMIT 1'
        )
        if sample:
            sid = str(sample[0]["source_id"] or "")
            if re.match(
                r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                sid, re.I,
            ):
                _RELATION_KEY_TYPE = "uuid"
            else:
                _RELATION_KEY_TYPE = "name"
        else:
            _RELATION_KEY_TYPE = "uuid"   # empty table — assume uuid (safe default)
    except Exception:
        _RELATION_KEY_TYPE = "uuid"
    return _RELATION_KEY_TYPE


async def count_relation_refs(
    conn: asyncpg.Connection,
    workspace: str,
    entity_id: str,
    relation_table: str,
    entity_name: str = "",
) -> tuple[int, int]:
    """
    Return (source_refs, target_refs) count for the given entity.

    Automatically uses entity_name or entity_id as the lookup value depending on
    whether the relation table stores UUIDs or name strings in source_id / target_id.
    Pass entity_name for correct results when the schema uses name-based keys.
    """
    key_type = await _get_relation_key_type(conn, relation_table)
    lookup_val = entity_id if key_type == "uuid" else (entity_name or entity_id)
    ws_clause, ws_params = _workspace_clause(workspace)
    src = await conn.fetchval(
        f'SELECT COUNT(*) FROM "{relation_table}" WHERE {ws_clause} AND source_id=${len(ws_params)+1}',
        *ws_params, lookup_val,
    )
    tgt = await conn.fetchval(
        f'SELECT COUNT(*) FROM "{relation_table}" WHERE {ws_clause} AND target_id=${len(ws_params)+1}',
        *ws_params, lookup_val,
    )
    return int(src), int(tgt)


async def apply_merge(
    conn: asyncpg.Connection,
    workspace: str,
    keep: dict,
    drop: dict,
    canonical_name: str,
    entity_table: str,
    relation_table: str,
    age_graphs: list[str],
    graphml_files: list[str],
) -> int:
    """
    Merge `drop` into `keep` across all storage layers:
      1. Relational table: rename keep, merge content/chunks, redirect relations, delete drop.
      2. AGE graph: redirect edges, DETACH DELETE drop node.
      3. NetworkX GraphML: redirect edges, remove drop node.
    Returns total number of relation rows updated.
    """
    ws_clause, ws_params = _workspace_clause(workspace)
    p = len(ws_params)  # number of workspace params already used

    async with conn.transaction():
        # 1. Rename keep to canonical if needed
        if keep["entity_name"] != canonical_name:
            await conn.execute(
                f'UPDATE "{entity_table}" SET entity_name=${p+1}, update_time=NOW() '
                f'WHERE {ws_clause} AND id=${p+2}',
                *ws_params, canonical_name, keep["id"],
            )

        # 2. Merge content
        if drop["content"] and drop["content"].strip() not in (keep["content"] or ""):
            merged_content = (keep["content"] or "") + "\n" + drop["content"]
            await conn.execute(
                f'UPDATE "{entity_table}" SET content=${p+1}, update_time=NOW() '
                f'WHERE {ws_clause} AND id=${p+2}',
                *ws_params, merged_content.strip(), keep["id"],
            )

        # 3. Merge chunk_ids
        if drop.get("chunk_ids"):
            await conn.execute(
                f"""
                UPDATE "{entity_table}"
                SET chunk_ids = ARRAY(
                    SELECT DISTINCT unnest(
                        COALESCE(chunk_ids, ARRAY[]::varchar[]) ||
                        ${p+1}::varchar[]
                    )
                ),
                update_time = NOW()
                WHERE {ws_clause} AND id=${p+2}
                """,
                *ws_params, drop["chunk_ids"], keep["id"],
            )

        # 4. Redirect relation references
        updated = 0
        key_type = await _get_relation_key_type(conn, relation_table)
        keep_key = keep["id"] if key_type == "uuid" else keep["entity_name"]
        drop_key = drop["id"] if key_type == "uuid" else drop["entity_name"]

        src_count, tgt_count = await count_relation_refs(
            conn, workspace, drop["id"], relation_table, drop["entity_name"]
        )

        if src_count:
            await conn.execute(
                f'UPDATE "{relation_table}" SET source_id=${p+1}, update_time=NOW() '
                f'WHERE {ws_clause} AND source_id=${p+2}',
                *ws_params, keep_key, drop_key,
            )
            updated += src_count

        if tgt_count:
            await conn.execute(
                f'UPDATE "{relation_table}" SET target_id=${p+1}, update_time=NOW() '
                f'WHERE {ws_clause} AND target_id=${p+2}',
                *ws_params, keep_key, drop_key,
            )
            updated += tgt_count

        # 5. Delete duplicate from relational table
        await conn.execute(
            f'DELETE FROM "{entity_table}" WHERE {ws_clause} AND id=${p+1}',
            *ws_params, drop["id"],
        )

    # 6. Update AGE graph (outside transaction — AGE has its own MVCC)
    if age_graphs:
        age_ok = await age_setup(conn)
        if age_ok:
            for graph_name in age_graphs:
                await age_merge_nodes(conn, graph_name, canonical_name, drop["entity_name"])
                # Also rename keep node to canonical if its name differed
                if keep["entity_name"] != canonical_name:
                    keep_esc = keep["entity_name"].replace('"', '\\"')
                    canon_esc = canonical_name.replace('"', '\\"')
                    try:
                        await conn.execute(f"""
                            SELECT * FROM cypher('{graph_name}', $$
                                MATCH (n:base {{entity_id: "{keep_esc}"}})
                                SET n.entity_id = "{canon_esc}"
                                RETURN n
                            $$) AS (r agtype)
                        """)
                    except Exception as e:
                        print(f"         [!] AGE rename failed for '{graph_name}': {e}")

    # 7. Update NetworkX GraphML files
    for gml_path in graphml_files:
        changed = graphml_merge_nodes(gml_path, canonical_name, drop["entity_name"])
        if keep["entity_name"] != canonical_name and not changed:
            # Also rename keep node in GraphML if name changed
            graphml_merge_nodes(gml_path, canonical_name, keep["entity_name"])
        if changed:
            print(f"         [+] GraphML updated: {os.path.basename(gml_path)}")

    return updated


# ============================================================================
# Main
# ============================================================================

# ============================================================================
# Embedding-based similarity detection
# ============================================================================


async def embed_entity_names(
    names: list[str],
    env_path: str,
    batch_size: int = 50,
) -> dict[str, "np.ndarray"]:
    """
    Embed a list of entity names using the configured embedding model.
    Calls the embedding API with only the entity_name string (no description).
    Returns {entity_name: unit_normalised_vector}.

    Supports ollama and openai-compatible bindings via EMBEDDING_BINDING env var.
    For ollama, uses /v1/embeddings (OpenAI-compatible endpoint).
    """
    if not _NUMPY_OK:
        print("  [!] numpy required for name embedding.")
        return {}

    load_dotenv(env_path, override=True)
    binding = os.getenv("EMBEDDING_BINDING", "ollama").lower()
    model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
    host = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434").rstrip("/")
    api_key = (
        os.getenv("EMBEDDING_BINDING_API_KEY")
        or os.getenv("LLM_BINDING_API_KEY")
        or "ollama"
    )

    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("  [!] openai package not installed. Run: pip install openai")
        return {}

    # Ollama exposes /v1/embeddings; openai-compatible host may already include /v1
    if binding == "ollama":
        base_url = f"{host}/v1"
    else:
        base_url = host if host.rstrip("/").endswith("/v1") else f"{host}/v1"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    result: dict[str, "np.ndarray"] = {}
    total = len(names)
    print(f"  Re-embedding {total} entity names via {binding} ({model}) ...")

    for i in range(0, total, batch_size):
        batch = names[i : i + batch_size]
        try:
            resp = await client.embeddings.create(model=model, input=batch)
            for j, emb_obj in enumerate(resp.data):
                vec = np.array(emb_obj.embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                result[batch[j]] = vec
        except Exception as e:
            print(f"  [!] Embedding batch {i // batch_size + 1} failed: {e}")

    print(f"  Embedded {len(result)}/{total} names successfully.")
    return result


async def fetch_entities_with_name_vectors(
    conn: asyncpg.Connection, workspace: str, entity_table: str, env_path: str
) -> list[dict]:
    """
    Fetch entity names from DB, then re-embed only the entity_name string
    (not the stored content_vector which includes description).
    Returns same format as fetch_entities_with_vectors.
    """
    entities = await fetch_entities(conn, workspace, entity_table)
    if not entities:
        return []

    names = [e["entity_name"] for e in entities]
    name_to_vec = await embed_entity_names(names, env_path)

    result = []
    for e in entities:
        vec = name_to_vec.get(e["entity_name"])
        if vec is not None:
            result.append({"id": e["id"], "entity_name": e["entity_name"], "vector": vec})
    return result


def _find_similar_clusters(
    entities: list[dict], threshold: float, max_cluster_size: int = 3
) -> list[list[str]]:
    """
    Find groups of entities whose embedding cosine similarity >= threshold.

    Uses a SIZE-CAPPED union-find to prevent "mega-clusters" caused by transitive
    chaining (A~B, B~C → {A,B,C} even when A and C are unrelated).
    Pairs are merged in descending similarity order; a union is skipped when
    it would push the cluster above max_cluster_size.

    Returns list of groups (each group is a list of entity names), only groups with 2+.
    """
    names = [e["entity_name"] for e in entities]
    vecs = np.stack([e["vector"] for e in entities])  # (N, D) — already normalised

    # Cosine similarity matrix via dot product (vectors are unit-normed)
    sim_matrix = vecs @ vecs.T   # (N, N)

    n = len(names)
    parent = list(range(n))
    size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        # Skip merge if the resulting cluster would exceed the size cap
        if size[px] + size[py] > max_cluster_size:
            return
        if size[px] < size[py]:
            px, py = py, px
        parent[py] = px
        size[px] += size[py]

    # Collect all pairs above threshold; merge most-similar first for best quality
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if s >= threshold:
                pairs.append((i, j, s))
    pairs.sort(key=lambda x: -x[2])

    for i, j, _ in pairs:
        union(i, j)

    # Collect groups
    clusters: dict[int, list[str]] = defaultdict(list)
    for i, name in enumerate(names):
        clusters[find(i)].append(name)

    return [group for group in clusters.values() if len(group) >= 2]






async def run(workspace: str, env_path: str, apply: bool) -> None:
    dry_run = not apply
    mode_label = "DRY RUN -- no changes" if dry_run else "[!] APPLY -- modifying database"

    print(f"\n{'=' * 62}")
    print(f"  KG Duplicate Entity Merger")
    print(f"  Workspace : '{workspace}' (empty string = default workspace)")
    print(f"  Mode      : {mode_label}")
    print(f"  Env file  : {env_path}")
    print(f"{'=' * 62}\n")

    conn = await make_connection(env_path)
    try:
        entity_table, relation_table = await discover_tables(conn)
        print(f"  Entity table  : {entity_table}")
        print(f"  Relation table: {relation_table}")

        # Discover graph storage backends
        age_graphs = await discover_age_graphs(conn, workspace)
        graphml_files = discover_graphml_files(env_path, workspace)

        if age_graphs:
            print(f"  AGE graphs    : {age_graphs}")
        if graphml_files:
            print(f"  GraphML files : {[os.path.basename(f) for f in graphml_files]}")
        if not age_graphs and not graphml_files:
            print(f"  Graph storage : not found (only relational tables will be updated)")
        print()

        entities = await fetch_entities(conn, workspace, entity_table)
        print(f"Total entities found : {len(entities)}\n")

        groups = find_duplicate_groups(entities)

        if not groups:
            print("[OK] No duplicate entities detected.")
            return

        print(f"Duplicate groups found: {len(groups)}\n")

        total_removed = 0
        total_relations_updated = 0

        for i, group in enumerate(groups, 1):
            canonical = group["canonical"]
            members = group["entities"]

            # Pick the "keeper": prefer entity whose name already matches canonical,
            # else the one with highest title score (= best existing casing).
            keeper = next(
                (e for e in members if e["entity_name"] == canonical),
                max(members, key=lambda e: (_title_score(e["entity_name"]), len(e["entity_name"]))),
            )
            drops = [e for e in members if e["id"] != keeper["id"]]

            print(f"  [{i:03d}] Canonical: \"{canonical}\"")
            src_r, tgt_r = await count_relation_refs(
                conn, workspace, keeper["id"], relation_table, keeper["entity_name"]
            )
            print(f"         [K] KEEP  : \"{keeper['entity_name']}\"  "
                  f"(id: ...{keeper['id'][-8:]}  refs: {src_r}-> {tgt_r}<-)")

            group_relations = 0
            for drop in drops:
                src_d, tgt_d = await count_relation_refs(
                    conn, workspace, drop["id"], relation_table, drop["entity_name"]
                )
                print(f"         [M] MERGE : \"{drop['entity_name']}\"  "
                      f"(id: ...{drop['id'][-8:]}  refs: {src_d}-> {tgt_d}<-)")
                group_relations += src_d + tgt_d

                if not dry_run:
                    updated = await apply_merge(
                        conn, workspace, keeper, drop, canonical,
                        entity_table, relation_table,
                        age_graphs, graphml_files,
                    )
                    print(f"                   → merged. {updated} relation(s) redirected.")
                    total_relations_updated += updated
                    total_removed += 1
                else:
                    total_removed += 1
                    total_relations_updated += group_relations

            print()

        print(f"{'=' * 62}")
        if dry_run:
            print(f"  DRY RUN SUMMARY")
            print(f"  Duplicate groups  : {len(groups)}")
            print(f"  Entities to remove: {total_removed}")
            print(f"  Relations affected: {total_relations_updated} (estimated)")
            print(f"\n  → Run with --apply to execute.")
        else:
            print(f"  APPLY SUMMARY")
            print(f"  Duplicate groups merged : {len(groups)}")
            print(f"  Entities removed        : {total_removed}")
            print(f"  Relation rows updated   : {total_relations_updated}")
            print(f"\n  [!] Restart the LightRAG server to reload the graph.")
        print(f"{'=' * 62}\n")

    finally:
        await conn.close()


# ============================================================================
# LLM client factory (supports OpenAI-compatible + Gemini)
# ============================================================================

def _make_llm_client(env_path: str):
    """Load env and return (LLM client, model name) - supports both OpenAI and Gemini."""
    load_dotenv(env_path, override=True)

    binding = os.getenv("LLM_BINDING", "openai").lower()
    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("LLM_BINDING_HOST") or None
    model = os.getenv("LLM_MODEL", "google/gemini-2.5-flash").strip()

    if binding == "gemini":
        try:
            import httpx  # noqa: F401
            return _GeminiClient(api_key=api_key, base_url=base_url), model
        except ImportError:
            raise RuntimeError("httpx package not installed. Run: pip install httpx")
    else:
        try:
            from openai import AsyncOpenAI
            return AsyncOpenAI(api_key=api_key, base_url=base_url), model
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")


class _GeminiClient:
    """Minimal Gemini API client that mimics AsyncOpenAI interface for chat completions."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = _GeminiClient.ChatCompletions(client)

    class ChatCompletions:
        def __init__(self, client):
            self.client = client

        async def create(self, model: str, messages: list, temperature: float = 0, timeout: int = 180, **kwargs):
            import httpx

            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    contents.append({
                        "parts": [{"text": f"System: {msg['content']}\n\nUser: "}]
                    })
                elif msg["role"] == "user":
                    if contents and "User: " in contents[-1]["parts"][0]["text"]:
                        contents[-1]["parts"][0]["text"] += msg["content"]
                    else:
                        contents.append({"parts": [{"text": msg["content"]}]})

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": kwargs.get("max_tokens", 65535),
                    "topP": 0.95,
                    "topK": 40,
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            }

            url = f"{self.client.base_url}/models/{model}:generateContent"
            headers = {"x-goog-api-key": self.client.api_key, "Content-Type": "application/json"}

            async with httpx.AsyncClient(timeout=timeout) as http:
                response = await http.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            content = ""
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    content = "".join(p.get("text", "") for p in candidate["content"]["parts"])

            class _Msg:
                def __init__(self, c): self.content = c
            class _Choice:
                def __init__(self, c): self.message = _Msg(c)
            class _Resp:
                def __init__(self, c): self.choices = [_Choice(c)]

            return _Resp(content)

    @property
    def chat(self):
        return self.Chat(self)


def _parse_llm_json(raw: str, label: str) -> list | None:
    """Strip markdown fences and parse JSON list. Returns None on failure."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"  [!] {label} JSON parse failed: {e}\n  Response: {raw[:400]}")
        return None


# ============================================================================
# LLM-based semantic deduplication (--llm-dedup)
# ============================================================================

_LLM_SYSTEM_PROMPT = (
    "You are a knowledge graph expert for an LMS (Learning Management System). "
    "Your task is to identify entity names that refer to the EXACT SAME real-world concept."
)

_LLM_USER_TEMPLATE = """\
You will see groups of entity names from a knowledge graph about an LMS system.
For each group, decide: are ANY two or more of these the EXACT same real-world thing?

STRICT RULES — read every rule before answering:
1. ONLY merge if entities are literally the same concept with different names/spellings.
2. Different numbered items are NEVER the same:
   BR 2 ≠ BR 3,  NT1 ≠ NT2,  CFD 1 ≠ CFD 2,  Step 1 ≠ Step 2
3. A button/action is NOT the same as the screen it belongs to.
4. A specific rule (BR 3: Saving Rules) is NOT the same as its category (Saving Rules).
5. An action (Validate Data) is NOT the same as the abstract concept (Validation).
6. An actor/role (User, Creator) is NOT the same as an action (User Login, Creator Validation).
7. Different UI elements are NOT the same: Button ≠ Screen ≠ Popup ≠ Message.
8. ONLY merge:
   - Exact bilingual equivalents: "Người Tạo" = "Creator", "Bộ Đề" = "Quiz Set"
   - Pure casing/spacing variants: "quiz set" = "Quiz Set", "quizset" = "Quiz Set"
   - Unambiguous abbreviation = full form: "UI" = "User Interface"

DEFAULT: output null.
Aim to merge FEWER THAN 20% of groups. If you are not 100% certain → output null.

{groups_block}

Respond ONLY with a valid JSON array — no markdown fences, no explanation:
[
  {{"group": 1, "merge": null}},
  {{"group": 2, "merge": {{"canonical": "CanonicalName", "aliases": ["OtherName"]}}}}
]
"""


async def fetch_entity_descriptions(
    conn: asyncpg.Connection,
    workspace: str,
    entity_table: str,
    names: list[str],
) -> dict[str, str]:
    """Return {entity_name: description} for the given names (fetched from content field)."""
    if not names:
        return {}
    ws_clause, ws_params = _workspace_clause(workspace)
    p = len(ws_params)
    rows = await conn.fetch(
        f'SELECT entity_name, content FROM "{entity_table}" WHERE {ws_clause} AND entity_name = ANY(${p+1})',
        *ws_params, names,
    )
    result = {}
    for r in rows:
        content = r["content"] or ""
        desc = content.split("\n", 1)[1].strip() if "\n" in content else content
        result[r["entity_name"]] = desc[:300]
    return result



def _build_llm_dedup_prompt(clusters: list[list[dict]], start_index: int = 1) -> str:
    """
    Build LLM prompt from similarity clusters.
    Each dict has entity_name, description, max_sim.
    start_index: global group number of the first cluster in this batch.
    """
    groups_block_parts = []
    for i, cluster in enumerate(clusters, start_index):
        lines = [f"Group {i}:"]
        for e in cluster:
            name = e["entity_name"]
            desc = e.get("description", "")
            sim = e.get("max_sim")
            sim_str = f"  (name similarity: {sim:.3f})" if sim is not None else ""
            lines.append(f'  - "{name}"{sim_str}')
            if desc:
                lines.append(f'    {desc[:200]}')
        groups_block_parts.append("\n".join(lines))
    return "\n\n".join(groups_block_parts)


async def _call_llm_for_dedup(
    clusters: list[list[dict]],
    env_path: str,
    batch_size: int = 30 ,  # groups per LLM call
) -> list[dict]:
    """
    Call the configured LLM to decide merges for the given similarity clusters.
    Clusters are split into batches of batch_size to avoid token limits.
    Group numbers in each batch use the original global index so the LLM
    response can be mapped back correctly.

    Uses _make_llm_client() — supports both OpenAI-compatible and Gemini bindings.
    Returns list of {canonical, aliases} dicts for groups decided to MERGE.
    """
    try:
        client, model = _make_llm_client(env_path)
    except RuntimeError as e:
        print(f"  [!] {e}")
        return []

    batches = [clusters[i : i + batch_size] for i in range(0, len(clusters), batch_size)]
    print(f"    {len(clusters)} groups → {len(batches)} batch(es) of ~{batch_size}  |  model: {model}")

    all_merges: list[dict] = []

    for batch_idx, group_batch in enumerate(batches, 1):
        # Build prompt: use GLOBAL group numbers so the LLM's group references map back
        global_offset = (batch_idx - 1) * batch_size   # 0-indexed start of this batch
        groups_block = _build_llm_dedup_prompt(group_batch, start_index=global_offset + 1)
        user_prompt = _LLM_USER_TEMPLATE.format(groups_block=groups_block)

        print(f"    Batch {batch_idx}/{len(batches)}: {len(user_prompt):,} chars  |  {len(group_batch)} groups")

        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
                timeout=180,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [!] Batch {batch_idx} LLM call failed: {e}")
            continue

        decisions = _parse_llm_json(raw, f"Batch {batch_idx}")
        if not isinstance(decisions, list):
            continue

        batch_start = global_offset + 1   # first global group number in this batch
        batch_end   = batch_start + len(group_batch) - 1

        batch_merges = []
        for item in decisions:
            if not isinstance(item, dict):
                continue
            merge = item.get("merge")
            if not (merge and isinstance(merge, dict)):
                continue
            # Validate group number belongs to this batch
            grp_num = item.get("group", 0)
            if grp_num < batch_start or grp_num > batch_end:
                continue
            canonical = merge.get("canonical", "").strip()
            aliases = merge.get("aliases", [])
            if canonical and aliases:
                batch_merges.append({"canonical": canonical, "aliases": aliases})

        print(f"    Batch {batch_idx} confirmed {len(batch_merges)} merge(s)")
        all_merges.extend(batch_merges)

    print(f"    Total merges from all batches: {len(all_merges)}")
    return all_merges


async def run_llm_dedup(
    workspace: str,
    env_path: str,
    threshold: float,
    apply: bool,
    forced_pairs: list[tuple[str, str]] | None = None,
) -> None:
    """
    Post-hoc LLM-based deduplication:
      1. Fetch all entity embeddings from VDB
      2. Find similarity clusters (union-find at threshold)
      3. Ask LLM to decide MERGE / KEEP for each cluster
      4. Show results (dry-run) or apply merges (--apply)
    """
    if not _NUMPY_OK:
        print("numpy is required for --llm-dedup. Run: pip install numpy")
        return

    dry_run = not apply
    print(f"\n{'=' * 62}")
    print(f"  LLM-Based Semantic Deduplication")
    print(f"  Workspace : '{workspace}'")
    print(f"  Threshold : {threshold:.2f}  (cosine similarity)")
    print(f"  Mode      : {'DRY RUN -- no changes' if dry_run else '[!] APPLY -- merging'}")
    print(f"  Env file  : {env_path}")
    print(f"{'=' * 62}\n")

    conn = await make_connection(env_path)
    try:
        entity_table, relation_table = await discover_tables(conn)
        graphml_files = discover_graphml_files(env_path, workspace)
        age_graphs = await discover_age_graphs(conn, workspace)

        print(f"  Entity table : {entity_table}")
        if graphml_files:
            print(f"  GraphML files: {[os.path.basename(f) for f in graphml_files]}")

        # Diagnostic: detect relation key type and show relation count
        key_type = await _get_relation_key_type(conn, relation_table)
        ws_clause_d, ws_params_d = _workspace_clause(workspace)
        rel_count = await conn.fetchval(
            f'SELECT COUNT(*) FROM "{relation_table}" WHERE {ws_clause_d}',
            *ws_params_d,
        )
        sample_rels = await conn.fetch(
            f'SELECT source_id, target_id FROM "{relation_table}" LIMIT 3'
        )
        print(f"  Relation key : {key_type}  (relations in workspace: {rel_count})")
        if sample_rels:
            samples = [
                (str(r["source_id"])[:40], str(r["target_id"])[:40])
                for r in sample_rels
            ]
            print(f"  Sample source/target: {samples}")
        print()

        # Step 0: Global text-dup detection across ALL entities (before embedding clustering).
        # This catches cross-cluster text duplicates such as "Bộ đề" / "Bộ Đề" that happen
        # to end up in different embedding clusters and would be missed by within-cluster checks.
        all_entities_plain = await fetch_entities(conn, workspace, entity_table)
        global_text_dup_groups = find_duplicate_groups(all_entities_plain)
        global_text_dup_merges: list[dict] = []
        text_dup_aliases: set[str] = set()
        for group in global_text_dup_groups:
            group_names = [e["entity_name"] for e in group["entities"]]
            canonical_g = group["canonical"]
            aliases_g = [n for n in group_names if n != canonical_g]
            global_text_dup_merges.append({"canonical": canonical_g, "aliases": aliases_g})
            text_dup_aliases.update(aliases_g)  # only aliases removed; canonicals stay

        if global_text_dup_merges:
            print(f"  Global text duplicates : {len(global_text_dup_merges)} group(s)")
            for m in global_text_dup_merges:
                print(f"    canonical=\"{m['canonical']}\"  aliases={m['aliases']}")
            print()

        # Step 1: Re-embed entity names only — text-dup aliases excluded so every cluster
        # member has a unique normalised key (no within-cluster text dups possible).
        # NAME-ONLY embedding avoids description noise in content_vector.
        entities_all = await fetch_entities_with_name_vectors(conn, workspace, entity_table, env_path)
        entities = [e for e in entities_all if e["entity_name"] not in text_dup_aliases]
        print(f"  Entities embedded     : {len(entities_all)}  ({len(entities)} after text-dup filter)")
        if len(entities) < 2:
            print("  Not enough entities to compare.")
            return

        # Step 2: Find similarity clusters
        clusters = _find_similar_clusters(entities, threshold)

        # Inject forced pairs (bypass cosine threshold) — go straight to LLM review
        if forced_pairs:
            entity_name_set = {e["entity_name"] for e in entities}
            for p1, p2 in forced_pairs:
                missing = [n for n in (p1, p2) if n not in entity_name_set]
                if missing:
                    print(f"  [!] Force-pair: not in entities (missing or text-dup alias): {missing}")
                    continue
                if any(p1 in c and p2 in c for c in clusters):
                    print(f"  [~] Force-pair: \"{p1}\" + \"{p2}\" already clustered together")
                    continue
                clusters.append([p1, p2])
                print(f"  [+] Force-pair injected into LLM review: \"{p1}\" + \"{p2}\"")
            if forced_pairs:
                print()

        if not clusters:
            print(f"\n  [OK] No similar clusters found at threshold {threshold:.2f}")
            print(f"{'=' * 62}\n")
            return

        print(f"  Similar clusters found : {len(clusters)}\n")

        # Step 3: Fetch descriptions for cluster members and compute max intra-cluster sim
        vec_by_name = {e["entity_name"]: e["vector"] for e in entities}

        all_member_names = [name for cluster in clusters for name in cluster]
        descriptions = await fetch_entity_descriptions(
            conn, workspace, entity_table, all_member_names
        )

        # Enrich clusters with description + max similarity for the LLM prompt
        enriched_clusters: list[list[dict]] = []
        for cluster in clusters:
            max_sim = 0.0
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    s = float(vec_by_name[cluster[i]] @ vec_by_name[cluster[j]])
                    if s > max_sim:
                        max_sim = s
            enriched = [
                {
                    "entity_name": name,
                    "description": descriptions.get(name, ""),
                    "max_sim": max_sim,
                }
                for name in cluster
            ]
            enriched_clusters.append(enriched)

        # ── All clusters are semantic: text-dup aliases were filtered before clustering ──
        # Global text-dup merges are already in global_text_dup_merges (Step 0).
        text_dup_merges = global_text_dup_merges
        semantic_clusters = enriched_clusters

        # Log cluster details
        for ci, cluster in enumerate(enriched_clusters, 1):
            sim_val = cluster[0]["max_sim"] if cluster else 0.0
            print(f"  Cluster {ci:02d}  (max_sim={sim_val:.4f})")
            for e in cluster:
                desc_short = (e.get("description") or "")[:120].replace("\n", " ")
                kind = "[text-dup]" if any(
                    e["entity_name"] in m["aliases"] + [m["canonical"]]
                    for m in text_dup_merges
                ) else "[semantic]"
                print(f"    {kind} \"{e['entity_name']}\"")
                if desc_short:
                    print(f"      {desc_short}")
            print()

        # Step 4: LLM decision (only for semantic clusters)
        llm_merges: list[dict] = []
        if semantic_clusters:
            print("  Calling LLM to decide merges...")
            llm_merges = await _call_llm_for_dedup(semantic_clusters, env_path)
            if not llm_merges:
                print("  [OK] LLM decided: no semantic merges needed.")
        else:
            print("  (No semantic clusters — LLM call skipped)")

        merges = text_dup_merges + llm_merges

        # Bug #1 fix — dedup: each entity name may appear in at most one merge group.
        # LLM can return overlapping/duplicate groups that would otherwise cause
        # the same entity to be processed multiple times (infinite-loop symptom).
        seen_names: set[str] = set()
        deduped: list[dict] = []
        for m in merges:
            new_aliases = [a for a in m["aliases"] if a not in seen_names]
            if m["canonical"] in seen_names or not new_aliases:
                continue
            seen_names.add(m["canonical"])
            seen_names.update(new_aliases)
            deduped.append({"canonical": m["canonical"], "aliases": new_aliases})
        merges = deduped

        if not merges:
            print(f"\n  [OK] No merges needed.")
            print(f"{'=' * 62}\n")
            return

        print(f"\n  Total groups to merge: {len(merges)}  "
              f"(text-dup: {len(text_dup_merges)}, semantic/LLM: {len(llm_merges)})\n")

        # Step 5: Show / apply
        total_removed = 0
        total_relations = 0

        for merge_decision in merges:
            all_names = [merge_decision["canonical"]] + merge_decision["aliases"]

            # Fetch entity records from VDB
            ws_clause, ws_params = _workspace_clause(workspace)
            p = len(ws_params)
            rows = await conn.fetch(
                f'SELECT id, entity_name, content, chunk_ids, file_path '
                f'FROM "{entity_table}" WHERE {ws_clause} AND entity_name = ANY(${p+1})',
                *ws_params, all_names,
            )
            all_existing = [dict(r) for r in rows]

            if not all_existing:
                print(f"  [!] None of {all_names} found in VDB — skipping")
                continue

            # Bug #4 fix — pick keeper by relation ref count.
            # Most-connected entity is the "primary" node; it keeps its name as canonical.
            # This prevents low-ref English names from overwriting high-ref primary names
            # (e.g. "Bộ Đề" with 7 refs beats "Quiz Set Record" with 2 refs).
            ref_src_tgt: dict[str, tuple[int, int]] = {}
            for ent in all_existing:
                s, t = await count_relation_refs(
                    conn, workspace, ent["id"], relation_table, ent["entity_name"]
                )
                ref_src_tgt[ent["entity_name"]] = (s, t)

            keeper = max(
                all_existing,
                key=lambda e: (
                    sum(ref_src_tgt.get(e["entity_name"], (0, 0))),
                    _title_score(e["entity_name"]),
                    len(e["entity_name"]),
                ),
            )
            canonical = keeper["entity_name"]
            drops = [e for e in all_existing if e["id"] != keeper["id"]]

            src_r, tgt_r = ref_src_tgt.get(canonical, (0, 0))
            print(f"  Canonical: \"{canonical}\"")
            print(f"    [K] KEEP  : \"{canonical}\"  (refs: {src_r}-> {tgt_r}<-)")

            for drop in drops:
                src_d, tgt_d = ref_src_tgt.get(drop["entity_name"], (0, 0))
                print(f"    [M] MERGE : \"{drop['entity_name']}\"  (refs: {src_d}-> {tgt_d}<-)")
                total_relations += src_d + tgt_d

                if not dry_run:
                    updated = await apply_merge(
                        conn, workspace, keeper, drop, canonical,
                        entity_table, relation_table, age_graphs, graphml_files,
                    )
                    print(f"              → merged. {updated} relation(s) redirected.")
                    total_removed += 1
                else:
                    total_removed += 1
            print()

        print(f"{'=' * 62}")
        if dry_run:
            print(f"  DRY RUN SUMMARY")
            print(f"  Groups to merge    : {len(merges)}")
            print(f"  Entities to remove : {total_removed}")
            print(f"  Relations affected : {total_relations} (estimated)")
            print(f"\n  → Run with --apply to execute.")
        else:
            print(f"  APPLY SUMMARY")
            print(f"  Groups merged      : {len(merges)}")
            print(f"  Entities removed   : {total_removed}")
            print(f"  Relations updated  : {total_relations}")
            print(f"\n  [!] Restart the LightRAG server to reload the graph.")
        print(f"{'=' * 62}\n")

    finally:
        await conn.close()



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge duplicate entity nodes in a LightRAG PostgreSQL knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--workspace", nargs="?", const="", default="",
        help="Workspace name. Omit or pass without value = default workspace (old KG).",
    )
    parser.add_argument(
        "--env", default=".env",
        help="Path to the .env file with PostgreSQL credentials. Default: .env",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually apply merges. Without this flag the script runs in dry-run mode.",
    )
    parser.add_argument(
        "--llm-dedup", action="store_true",
        help=(
            "Find similar clusters via cosine similarity, then ask LLM to decide "
            "which to merge. Dry-run by default; use --apply to execute merges."
        ),
    )
    parser.add_argument(
        "--threshold", type=float, default=0.9,
        help="Cosine similarity threshold for --llm-dedup (default: 0.9).",
    )
    parser.add_argument(
        "--force-pair", nargs=2, metavar=("NAME1", "NAME2"), action="append",
        dest="force_pairs", default=[],
        help=(
            "Force a name pair into LLM review regardless of cosine similarity "
            "(use with --llm-dedup). Can be repeated for multiple pairs."
        ),
    )
    parser.add_argument(
        "--log", nargs="?", const="", default=None, metavar="FILE",
        help=(
            "Write all output to a log file in addition to the terminal. "
            "If FILE is omitted, a timestamped filename is auto-generated "
            "(e.g. dedup_20260226_153045.log)."
        ),
    )
    args = parser.parse_args()

    # --- set up tee logging if --log was passed ---
    _log_file = None
    if args.log is not None:
        import datetime
        log_path = args.log if args.log else (
            f"dedup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        _log_file = open(log_path, "w", encoding="utf-8", errors="replace")
        sys.stdout = _Tee(sys.stdout, _log_file)
        sys.stderr = _Tee(sys.stderr, _log_file)
        print(f"[log] Writing output to: {log_path}")

    if args.llm_dedup:
        asyncio.run(run_llm_dedup(
            workspace=args.workspace,
            env_path=args.env,
            threshold=args.threshold,
            apply=args.apply,
            forced_pairs=[tuple(p) for p in args.force_pairs],
        ))
    else:
        asyncio.run(run(
            workspace=args.workspace,
            env_path=args.env,
            apply=args.apply,
        ))

    if _log_file is not None:
        _log_file.close()
        sys.stdout = sys.stdout._stream
        sys.stderr = sys.stderr._stream


if __name__ == "__main__":
    main()
