#!/usr/bin/env python3
"""
KG Post-Processing — Duplicate Entity Merger for LightRAG (PostgreSQL backend)

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


# ── Garbage Entity Filter Config ─────────────────────────────────────────────
# Edit _GARBAGE_PATTERNS or _MAX_ENTITY_NAME_LEN to adapt to a new project.
_GARBAGE_PATTERNS: list[str] = [
    # English section headings
    r"Overview", r"Activity Flow", r"Business Rules?", r"Trigger(\s+Event)?",
    r"Pre-conditions?", r"Post-conditions?", r"Objective", r"Instructions?", r"Notes?",
    # Vietnamese section headings
    r"Hậu điều kiện", r"Tiền điều kiện", r"Luồng sự kiện", r"Quy tắc nghiệp vụ",
    r"Thông tin chung", r"Kích hoạt", r"Mục tiêu", r"Tác nhân",
    # Table column headers
    r"User", r"System", r"Người dùng", r"Hệ thống",
    # Step labels (with optional number: "Step 1", "Bước 2")
    r"Bước\s*\d*", r"Step\s*\d*",
]
_MAX_ENTITY_NAME_LEN: int = 80  # names longer than this are likely flow descriptions


def _build_garbage_regex(patterns: list[str]) -> re.Pattern:
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(f"^({combined})$", re.IGNORECASE)


_GARBAGE_ENTITY = _build_garbage_regex(_GARBAGE_PATTERNS)

# Standalone code tokens with no descriptive name → garbage
_CODE_ONLY = re.compile(
    r"^("
    r"(?:UC|CMUC)(?:-[A-Z]+)*-\d+"           # UC-QAS-001, UC-CL-004
    r"|CMUC\s+\d+"                             # CMUC 5
    r"|(?:BR|CBR|CFD|SCD|IEM|EMSG|NT)\s*\d*"  # BR 2, CBR14, NT1, SCD (no number)
    r")$",
    re.IGNORECASE,
)
# ─────────────────────────────────────────────────────────────────────────────


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

    _code_format() is always applied to the winner so that names like
    "BR 3 - Saving Rules" or "BR 3 Saving Rules" are normalised to
    "BR 3: Saving Rules" even when no candidate already has a colon.
    """
    code_re = re.compile(r'^[A-Z]{1,5}\s?\d+[\w-]*:\s+.+$')
    code_names = [n for n in names if code_re.match(n)]
    if code_names:
        best = max(
            code_names,
            key=lambda n: (_code_quality_score(n), _title_score(n), len(n)),
        )
    else:
        best = max(names, key=lambda n: (_code_quality_score(n), _title_score(n), len(n)))
    return _code_format(best)


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
# GraphML-based data fetching functions
# ============================================================================

def fetch_entities_from_graphml(env_path: str, workspace: str) -> list[dict]:
    """
    Fetch entity names from GraphML files instead of PostgreSQL database.
    Returns list of entities in same format as fetch_entities() for compatibility.
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed. Run: pip install networkx")
        return []

    graphml_files = discover_graphml_files(env_path, workspace)
    
    if not graphml_files:
        print("No GraphML files found.")
        return []
    
    entities = []
    entity_id_counter = 1
    
    for gml_path in graphml_files:
        try:
            G = nx.read_graphml(gml_path) 
            nodes = list(G.nodes())
            
            for node in nodes:
                # Create fake entity record compatible with existing code
                entities.append({
                    "id": f"graphml_{entity_id_counter:08d}",  # fake UUID-like ID
                    "entity_name": node,
                    "content": f"Entity from GraphML: {node}",
                    "chunk_ids": [],
                    "file_path": os.path.basename(gml_path),
                })
                entity_id_counter += 1
                
        except Exception as e:
            print(f"Error reading {gml_path}: {e}")
    
    return entities


def count_relation_refs_graphml(
    env_path: str, workspace: str, entity_name: str
) -> tuple[int, int]:
    """
    Count incoming and outgoing edges for an entity in GraphML files.
    Returns (outgoing_count, incoming_count) to match PostgreSQL function signature.
    """
    try:
        import networkx as nx
    except ImportError:
        return (0, 0)

    graphml_files = discover_graphml_files(env_path, workspace)
    
    outgoing = 0
    incoming = 0
    
    for gml_path in graphml_files:
        try:
            G = nx.read_graphml(gml_path)
            if entity_name in G:
                # For undirected graphs, neighbors count as both incoming and outgoing
                neighbors = list(G.neighbors(entity_name))
                degree = len(neighbors)
                outgoing += degree
                incoming += degree
        except Exception:
            continue
            
    return (outgoing, incoming)


# ============================================================================
# Main
# ============================================================================

# ============================================================================
# Embedding-based similarity detection
# ============================================================================

async def fetch_entities_with_vectors(
    conn: asyncpg.Connection, workspace: str, entity_table: str
) -> list[dict]:
    """Fetch entity names + their embedding vectors from the VDB table."""
    ws_clause, ws_params = _workspace_clause(workspace)
    rows = await conn.fetch(
        f"""
        SELECT id, entity_name, content_vector::text AS vec_str
        FROM "{entity_table}"
        WHERE {ws_clause} AND content_vector IS NOT NULL
        ORDER BY entity_name
        """,
        *ws_params,
    )
    result = []
    for r in rows:
        vec_str = r["vec_str"]
        try:
            vec = np.array([float(x) for x in vec_str.strip("[]").split(",")], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm   # pre-normalise for fast cosine via dot product
            result.append({"id": r["id"], "entity_name": r["entity_name"], "vector": vec})
        except Exception:
            pass   # skip if vector is malformed
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


async def run_detect_similar(
    workspace: str, env_path: str, threshold: float
) -> None:
    """
    Detect semantically similar entity pairs/clusters using embedding cosine similarity.
    Output is informational only — no DB changes.
    """
    if not _NUMPY_OK:
        print("numpy is required for --detect-similar. Run: pip install numpy")
        return

    print(f"\n{'=' * 62}")
    print(f"  Embedding Similarity Detector")
    print(f"  Workspace : '{workspace}'")
    print(f"  Threshold : {threshold:.2f}  (cosine similarity)")
    print(f"  Env file  : {env_path}")
    print(f"{'=' * 62}\n")

    conn = await make_connection(env_path)
    try:
        entity_table, _ = await discover_tables(conn)
        print(f"  Entity table : {entity_table}\n")

        entities = await fetch_entities_with_vectors(conn, workspace, entity_table)
        print(f"  Entities with vectors : {len(entities)}")

        if len(entities) < 2:
            print("  Not enough entities to compare.")
            return

        clusters = _find_similar_clusters(entities, threshold)

        if not clusters:
            print(f"\n  [OK] No entity pairs found with similarity >= {threshold:.2f}")
            print(f"{'=' * 62}\n")
            return

        # Sort clusters by highest intra-cluster similarity (most similar first)
        vec_by_name = {e["entity_name"]: e["vector"] for e in entities}
        def max_sim(group):
            sims = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    s = float(vec_by_name[group[i]] @ vec_by_name[group[j]])
                    sims.append(s)
            return max(sims) if sims else 0.0

        clusters_sorted = sorted(clusters, key=max_sim, reverse=True)

        print(f"\n  Similar clusters found : {len(clusters_sorted)}\n")

        for i, group in enumerate(clusters_sorted, 1):
            # Find the pair with highest similarity in this cluster
            best_sim = 0.0
            for a_idx in range(len(group)):
                for b_idx in range(a_idx + 1, len(group)):
                    s = float(vec_by_name[group[a_idx]] @ vec_by_name[group[b_idx]])
                    if s > best_sim:
                        best_sim = s

            canonical = pick_canonical(group)

            print(f"  [{i:03d}] similarity={best_sim:.4f}  canonical → \"{canonical}\"")
            for name in sorted(group):
                marker = "[K]" if name == canonical else "[~]"
                print(f"         {marker} \"{name}\"")
            print()

        print(f"  Tip: Add confirmed pairs to FIXED_CANONICAL in kg_dedup.py,")
        print(f"       then re-run  kg_dedup.py --apply  to merge them.")
        print(f"{'=' * 62}\n")

    finally:
        await conn.close()


def sync_graphml_with_vdb(graphml_path: str, vdb_entity_names: set[str]) -> tuple[int, int]:
    """
    Remove from the GraphML file any node whose ID is not in vdb_entity_names.
    Edges to/from removed nodes are also removed (networkx handles this with remove_node).
    Returns (nodes_removed, edges_removed).
    """
    try:
        import networkx as nx
    except ImportError:
        print("  [!] networkx not installed -- GraphML sync skipped.")
        return 0, 0

    G = nx.read_graphml(graphml_path)
    orphans = [n for n in list(G.nodes()) if n not in vdb_entity_names]

    if not orphans:
        return 0, 0

    edges_removed = sum(G.degree(n) for n in orphans)
    for n in orphans:
        G.remove_node(n)   # also removes attached edges

    nx.write_graphml(G, graphml_path)
    return len(orphans), edges_removed


async def delete_entities(
    conn: asyncpg.Connection,
    workspace: str,
    names_to_delete: list[str],
    entity_table: str,
    relation_table: str,
    graphml_files: list[str],
    dry_run: bool,
) -> None:
    """
    Delete specific entity nodes (by name) from all storage layers.
    Relations touching those entities are also removed.
    """
    ws_clause, ws_params = _workspace_clause(workspace)
    p = len(ws_params)

    not_found = []
    deleted = 0
    relations_removed = 0

    for name in names_to_delete:
        # Look up the entity
        rows = await conn.fetch(
            f'SELECT id, entity_name FROM "{entity_table}" WHERE {ws_clause} AND entity_name=${ p+1}',
            *ws_params, name,
        )
        if not rows:
            not_found.append(name)
            continue

        for row in rows:
            eid = row["id"]
            src, tgt = await count_relation_refs(conn, workspace, eid, relation_table, row["entity_name"])
            total_refs = src + tgt
            print(f"  [D] DELETE: \"{name}\"  (refs: {src}-> {tgt}<-  total:{total_refs})")

            if not dry_run:
                async with conn.transaction():
                    # Remove relations
                    await conn.execute(
                        f'DELETE FROM "{relation_table}" WHERE {ws_clause} AND source_id=${p+1}',
                        *ws_params, eid,
                    )
                    await conn.execute(
                        f'DELETE FROM "{relation_table}" WHERE {ws_clause} AND target_id=${p+1}',
                        *ws_params, eid,
                    )
                    # Remove entity
                    await conn.execute(
                        f'DELETE FROM "{entity_table}" WHERE {ws_clause} AND id=${p+1}',
                        *ws_params, eid,
                    )
                # Remove from GraphML
                for gml_path in graphml_files:
                    try:
                        import networkx as nx
                        G = nx.read_graphml(gml_path)
                        if name in G:
                            G.remove_node(name)
                            nx.write_graphml(G, gml_path)
                            print(f"         [+] GraphML node removed: {os.path.basename(gml_path)}")
                    except Exception as e:
                        print(f"         [!] GraphML update failed: {e}")

                deleted += 1
                relations_removed += total_refs
            else:
                deleted += 1
                relations_removed += total_refs

    if not_found:
        print(f"\n  [!] Not found in workspace: {not_found}")
    print(f"\n  Entities {'to delete' if dry_run else 'deleted'}: {deleted}")
    print(f"  Relations {'affected' if dry_run else 'removed'} : {relations_removed}")


async def run_delete(workspace: str, env_path: str, names: list[str], apply: bool) -> None:
    """Delete specific entities by name from all storage layers."""
    dry_run = not apply
    print(f"\n{'=' * 62}")
    print(f"  Entity Deletion")
    print(f"  Workspace : '{workspace}'")
    print(f"  Mode      : {'DRY RUN -- no changes' if dry_run else '[!] APPLY -- deleting entities'}")
    print(f"  Targets   : {len(names)} entity name(s)")
    print(f"{'=' * 62}\n")

    conn = await make_connection(env_path)
    try:
        entity_table, relation_table = await discover_tables(conn)
        print(f"  Entity table  : {entity_table}")
        graphml_files = discover_graphml_files(env_path, workspace)
        if graphml_files:
            print(f"  GraphML files : {[os.path.basename(f) for f in graphml_files]}")
        print()

        await delete_entities(
            conn, workspace, names,
            entity_table, relation_table,
            graphml_files, dry_run,
        )

        if dry_run:
            print(f"\n  → Run with --apply to execute.")
        else:
            print(f"\n  [!] Restart the LightRAG server to reload the graph.")
        print(f"{'=' * 62}\n")
    finally:
        await conn.close()


async def run_fix_graphml(workspace: str, env_path: str) -> None:
    """
    Sync the GraphML file with the VDB entity table:
    remove any GraphML node whose entity name is not present in the VDB table.
    This repairs the GraphML when --apply was run before GraphML support was added.
    """
    print(f"\n{'=' * 62}")
    print(f"  GraphML Sync (fix-graphml)")
    print(f"  Workspace : '{workspace}'")
    print(f"  Env file  : {env_path}")
    print(f"{'=' * 62}\n")

    conn = await make_connection(env_path)
    try:
        entity_table, _ = await discover_tables(conn)
        print(f"  Entity table  : {entity_table}")

        graphml_files = discover_graphml_files(env_path, workspace)
        if not graphml_files:
            print("  [!] No GraphML files found for this workspace.")
            return
        print(f"  GraphML files : {[os.path.basename(f) for f in graphml_files]}\n")

        entities = await fetch_entities(conn, workspace, entity_table)
        vdb_names = {e["entity_name"] for e in entities}
        print(f"  VDB entities  : {len(vdb_names)}")

        for gml_path in graphml_files:
            try:
                import networkx as nx
                G = nx.read_graphml(gml_path)
                print(f"  GraphML nodes : {G.number_of_nodes()}  edges: {G.number_of_edges()}")
            except Exception as e:
                print(f"  [!] Could not read GraphML: {e}")
                continue

            removed_nodes, removed_edges = sync_graphml_with_vdb(gml_path, vdb_names)
            if removed_nodes:
                print(f"\n  [+] Removed {removed_nodes} orphan node(s) and {removed_edges} edge(s)")
                print(f"  [+] GraphML saved: {os.path.basename(gml_path)}")
                try:
                    G2 = nx.read_graphml(gml_path)
                    print(f"  GraphML now   : {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
                except Exception:
                    pass
            else:
                print(f"\n  [OK] GraphML is already in sync with VDB -- no changes needed.")

        print(f"\n  [!] Restart the LightRAG server to reload the graph.")
        print(f"{'=' * 62}\n")
    finally:
        await conn.close()


def find_garbage_entities(entities: list[dict]) -> list[tuple[dict, str]]:
    """
    Return (entity, reason) pairs that match garbage patterns and should be purged.
    Three checks in priority order:
      1. Blocklist regex  — section headings, column headers, step labels
      2. Name too long    — likely a flow description, not an entity name
      3. Standalone code  — code token without a descriptive name
    """
    result = []
    for ent in entities:
        name = ent["entity_name"]
        if _GARBAGE_ENTITY.match(name):
            result.append((ent, "blocklist match"))
        elif len(name) > _MAX_ENTITY_NAME_LEN:
            result.append((ent, f"name too long ({len(name)} chars)"))
        elif _CODE_ONLY.match(name):
            result.append((ent, "standalone code without descriptive name"))
    return result


async def run_purge_garbage(workspace: str, env_path: str, apply: bool) -> None:
    """
    Detect and optionally delete garbage entity nodes from all storage layers.
    Garbage = section headings, column headers, standalone codes, overly-long names.
    """
    dry_run = not apply
    print(f"\n{'=' * 62}")
    print(f"  Garbage Entity Purge")
    print(f"  Workspace : '{workspace}'")
    print(f"  Mode      : {'DRY RUN -- no changes' if dry_run else '[!] APPLY -- deleting entities'}")
    print(f"  Env file  : {env_path}")
    print(f"{'=' * 62}\n")

    conn = await make_connection(env_path)
    try:
        entity_table, relation_table = await discover_tables(conn)
        print(f"  Entity table  : {entity_table}")
        graphml_files = discover_graphml_files(env_path, workspace)
        if graphml_files:
            print(f"  GraphML files : {[os.path.basename(f) for f in graphml_files]}")
        print()

        entities = await fetch_entities(conn, workspace, entity_table)
        garbage = find_garbage_entities(entities)

        print(f"  Total entities   : {len(entities)}")
        print(f"  Garbage detected : {len(garbage)}")
        print(f"  Clean remaining  : {len(entities) - len(garbage)}\n")

        if not garbage:
            print("  [OK] No garbage entities detected.")
            print(f"{'=' * 62}\n")
            return

        # Print each garbage entity with reason and relation ref counts
        for ent, reason in sorted(garbage, key=lambda x: x[0]["entity_name"]):
            src, tgt = await count_relation_refs(
                conn, workspace, ent["id"], relation_table, ent["entity_name"]
            )
            print(f"  [G] \"{ent['entity_name']}\"")
            print(f"       reason: {reason}  |  refs: {src}-> {tgt}<-")

        if not dry_run:
            print()
            names_to_delete = [ent["entity_name"] for ent, _ in garbage]
            await delete_entities(
                conn, workspace, names_to_delete,
                entity_table, relation_table,
                graphml_files, dry_run=False,
            )
        else:
            print(f"\n  → Run with --apply to execute.")

        print(f"{'=' * 62}\n")
    finally:
        await conn.close()


def run(workspace: str, env_path: str, apply: bool) -> None:
    dry_run = not apply
    mode_label = "DRY RUN -- no changes" if dry_run else "[!] APPLY -- modifying GraphML files"

    print(f"\n{'=' * 62}")
    print(f"  KG Duplicate Entity Merger (GraphML Mode)")
    print(f"  Workspace : '{workspace}' (empty string = default workspace)")
    print(f"  Mode      : {mode_label}")
    print(f"  Env file  : {env_path}")
    print(f"{'=' * 62}\n")

    # Discover GraphML files (primary data source)
    graphml_files = discover_graphml_files(env_path, workspace)

    if not graphml_files:
        print("No GraphML files found.")
        return

    print(f"  GraphML files : {[os.path.basename(f) for f in graphml_files]}")
    print()

    # Fetch entities from GraphML instead of PostgreSQL
    entities = fetch_entities_from_graphml(env_path, workspace)
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
        
        # Count relations from GraphML instead of PostgreSQL
        src_r, tgt_r = count_relation_refs_graphml(env_path, workspace, keeper["entity_name"])
        print(f"         [K] KEEP  : \"{keeper['entity_name']}\"  "
              f"(id: ...{keeper['id'][-8:]}  refs: {src_r}→ {tgt_r}←)")

        group_relations = 0
        for drop in drops:
            src_d, tgt_d = count_relation_refs_graphml(env_path, workspace, drop["entity_name"])
            print(f"         [M] MERGE : \"{drop['entity_name']}\"  "
                  f"(id: ...{drop['id'][-8:]}  refs: {src_d}→ {tgt_d}←)")
            group_relations += src_d + tgt_d

            if not dry_run:
                # Apply merge only to GraphML files
                updated = 0
                for gml_path in graphml_files:
                    changed = graphml_merge_nodes(gml_path, canonical, drop["entity_name"])
                    if keeper["entity_name"] != canonical and not changed:
                        # Also rename keep node in GraphML if name changed
                        graphml_merge_nodes(gml_path, canonical, keeper["entity_name"])
                    if changed:
                        updated += 1
                        print(f"         [+] GraphML updated: {os.path.basename(gml_path)}")
                
                total_relations_updated += src_d + tgt_d
                total_removed += 1
            else:
                total_removed += 1

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
        print(f"  GraphML files updated   : {len(graphml_files)}")
        print(f"\n  [!] Restart the LightRAG server to reload the graph.")
    print(f"{'=' * 62}\n")


# ============================================================================
# LLM-based semantic deduplication (--llm-dedup)  — 3-phase pipeline
# ============================================================================
#
# Phase 1 (rule)        : text-dup detection — casing, punctuation, bilingual
# Phase 2 (LLM liberal) : scan all entity names → candidate groups (false-pos OK)
# Phase 3 (LLM strict)  : confirm each candidate with full descriptions
# ============================================================================

# ── Phase 2 prompts ───────────────────────────────────────────────────────────

# ── Phase 2b prompts ──────────────────────────────────────────────────────────

_LLM_PHASE2B_SYSTEM = (
    "You are a knowledge graph expert for an LMS (Learning Management System). "
    "Embedding similarity pre-filtering has already identified candidate groups. "
    "Your job is to verify which groups (or sub-groups) are actually the same real-world concept. "
    "Be LIBERAL — false positives are acceptable; they will be filtered in the next step."
)

_LLM_PHASE2B_TEMPLATE = """\
Embedding similarity (cosine >= {embed_threshold:.2f}) found the following candidate groups.
Verify which groups — or sub-groups — refer to the SAME real-world concept \
(same entity with different spelling / language / casing / abbreviation).

Guidelines:
- Be LIBERAL: include uncertain pairs; they will be verified with full descriptions next.
- You may CONFIRM the whole group, SPLIT it into sub-groups, or REJECT it entirely.
- Each result must have at least 2 members.
- Numbered codes are NEVER duplicates (CFD 1 ≠ CFD 2, BR 2 ≠ BR 3, Step 1 ≠ Step 2).
- Do NOT invent names that are not listed below.

Candidate groups from embedding:
{groups_block}

Respond ONLY with a JSON array. Output [] if no candidates confirmed.
[
  {{"members": ["Name A", "Name B"], "reason": "brief reason"}},
  ...
]
"""

# ── Phase 3 prompts ───────────────────────────────────────────────────────────

_LLM_PHASE3_SYSTEM = (
    "You are a knowledge graph expert for an LMS (Learning Management System). "
    "For each candidate group, identify which members (if any) should be merged. "
    "You may merge ALL members, a SUBSET, or NONE — be CONSERVATIVE, only merge when certain."
)

_LLM_PHASE3_TEMPLATE = """\
For each candidate group, identify which members (if any) refer to the EXACT same \
real-world concept and should be merged.

You may merge ALL members, a SUBSET, or NONE from each group.
If only some members are duplicates, list only those in a subgroup — leave the rest out.

STRICT RULES — any violation means do NOT merge those members:
1. ONLY merge: bilingual equivalents ("Sinh Đề Thi" = "Generate Quiz & Assessment", \
"Người Tạo" = "Creator"), pure casing/spacing variants, abbreviation = full form, \
OR synonymous event/status names across languages \
("Đã Sinh Đề Thi" = "Quiz Generated Successfully" — both describe the same system event).
2. Different numbered items are NEVER the same: \
CFD 1 ≠ CFD 2, BR 2 ≠ BR 3, Step 1 ≠ Step 2.
3. An action is NOT the screen / form it belongs to.
4. A specific rule (BR 3: Saving Rules) is NOT its category (Saving Rules).
5. An action (Validate Data) ≠ abstract concept (Validation).
6. Different UI elements: Button ≠ Screen ≠ Popup ≠ Message.
7. Structured identifiers — any keyword prefix with or without a trailing number/label \
(e.g., BR, BR001, Step 3) — must NEVER be merged unless \
their FULL names match exactly (modulo language/casing). Specifically: \
(a) bare prefix ≠ numbered form: "BR" ≠ "BR001"; \
(b) code ≠ verbose description: "BR001" ≠ "Business Rule BR001"; \

DEFAULT: empty subgroups. Only add a subgroup when you are 100% certain.

{groups_block}

Respond ONLY with a JSON array — one entry per group, in order.
Each entry has a "subgroups" list (empty = nothing merges from this group).
Each subgroup lists the members to merge, the canonical name to keep, and a confidence \
score (0.0–1.0) reflecting how certain you are they refer to the EXACT same concept \
(1.0 = absolutely certain, 0.5 = uncertain, 0.0 = different).
[
  {{"group": 1, "subgroups": []}},
  {{"group": 2, "subgroups": [{{"members": ["Name A", "Name B"], "canonical": "Name A", "confidence": 0.95}}]}},
  {{"group": 3, "subgroups": [
    {{"members": ["X", "Y"], "canonical": "X", "confidence": 0.90}},
    {{"members": ["P", "Q", "R"], "canonical": "Q", "confidence": 0.60}}
  ]}}
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
        result[r["entity_name"]] = desc[:400]
    return result


def _make_llm_client(env_path: str):
    """Load env and return (LLM client, model name) - supports both OpenAI and Gemini."""
    load_dotenv(env_path, override=True)
    
    binding = os.getenv("LLM_BINDING", "openai").lower()
    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("LLM_BINDING_HOST") or None  
    model = os.getenv("LLM_MODEL", "google/gemini-2.5-flash").strip()
    
    if binding == "gemini":
        # Use Gemini-compatible client
        try:
            import httpx
            return _GeminiClient(api_key=api_key, base_url=base_url), model
        except ImportError:
            raise RuntimeError("httpx package not installed. Run: pip install httpx")
    else:
        # Default to OpenAI client
        try:
            from openai import AsyncOpenAI
            return AsyncOpenAI(api_key=api_key, base_url=base_url), model  
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")


class _GeminiClient:
    """Minimal Gemini API client that mimics AsyncOpenAI interface for chat completions."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        
    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = _GeminiClient.ChatCompletions(client)
            
    class ChatCompletions:
        def __init__(self, client):
            self.client = client
            
        async def create(self, model: str, messages: list, temperature: float = 0, timeout: int = 180, **kwargs):
            import httpx
            import json
            
            # Convert OpenAI messages to Gemini format
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    # Gemini doesn't have explicit system role, prepend to first user message
                    contents.append({
                        "parts": [{"text": f"System: {msg['content']}\n\nUser: "}]
                    })
                elif msg["role"] == "user":
                    if contents and "User: " in contents[-1]["parts"][0]["text"]:
                        # Append to existing user message that had system prepended
                        contents[-1]["parts"][0]["text"] += msg["content"]
                    else:
                        contents.append({
                            "parts": [{"text": msg["content"]}]
                        })
            
            # Gemini API payload
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": kwargs.get("max_tokens", 65535),  # Increased to 32K
                    "topP": 0.95,
                    "topK": 40
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            }
            
            # Build URL: base_url already includes /gemini path
            # Full URL will be: http://localhost:1305/gemini/models/{model}:generateContent  
            url = f"{self.client.base_url}/models/{model}:generateContent"
            
            headers = {
                "x-goog-api-key": self.client.api_key,
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                # Convert Gemini response to OpenAI format
                content = ""
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        content = "".join(part.get("text", "") for part in parts)
                
                # Create mock OpenAI response object
                class MockResponse:
                    def __init__(self, content):
                        self.choices = [MockChoice(content)]
                        
                class MockChoice:
                    def __init__(self, content):
                        self.message = MockMessage(content)
                        
                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                
                return MockResponse(content)
    
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



async def _call_llm_phase2b_verify_clusters(
    embed_candidates: list[dict],
    embed_threshold: float,
    env_path: str,
) -> list[dict]:
    """
    Phase 2b — LLM liberal verification on embedding candidate clusters.
    embed_candidates: [{"members": [...], "reason": "..."}, ...]  (from Phase 2a)
    Returns verified candidate groups: [{"members": [...], "reason": "..."}, ...].
    """
    try:
        client, model = _make_llm_client(env_path)
    except RuntimeError as e:
        print(f"  [!] {e}")
        return []

    # Format each cluster as a numbered group block
    lines = []
    for i, c in enumerate(embed_candidates, 1):
        members_str = ", ".join(f'"{m}"' for m in c["members"])
        lines.append(f"Group {i}: {members_str}")
    groups_block = "\n".join(lines)

    user_prompt = _LLM_PHASE2B_TEMPLATE.format(
        embed_threshold=embed_threshold,
        groups_block=groups_block,
    )
    print(f"    Sending {len(embed_candidates)} embedding group(s) to LLM  |  {len(user_prompt):,} chars  |  model: {model}")

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_PHASE2B_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
            timeout=240,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  [!] Phase 2b LLM call failed: {e}")
        return []

    data = _parse_llm_json(raw, "Phase 2b")
    if not isinstance(data, list):
        return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        members = item.get("members", [])
        if isinstance(members, list) and len(members) >= 2:
            results.append({
                "members": [str(m).strip() for m in members],
                "reason":  str(item.get("reason", "")),
            })
    return results


async def _call_llm_phase3_confirm(
    candidate_groups: list[dict],
    env_path: str,
    confidence_threshold: float = 0.85,
    batch_size: int = 60,  # groups per batch
) -> list[dict]:
    """
    Phase 3 — Conservative confirmation (with batching).
    Each candidate_group: {"members": [...], "reason": "...", "descriptions": {name: desc}}.
    Returns confirmed merges: [{"canonical": ..., "aliases": [...], "confidence": float}, ...].
    Only subgroups with confidence >= confidence_threshold are included.
    """
    try:
        client, model = _make_llm_client(env_path)
    except RuntimeError as e:
        print(f"  [!] {e}")
        return []

    # Split candidate groups into batches
    batches = [candidate_groups[i:i + batch_size] for i in range(0, len(candidate_groups), batch_size)]
    print(f"    Processing {len(candidate_groups)} groups in {len(batches)} batch(es) of ~{batch_size} each")
    
    all_merges = []
    
    for batch_idx, group_batch in enumerate(batches, 1):
        parts = []
        for i, grp in enumerate(group_batch, 1):
            # Use original index for group numbering
            original_idx = (batch_idx - 1) * batch_size + i
            lines = [f"Group {original_idx}  (Phase-2 reason: {grp.get('reason', '?')})"]
            for name in grp["members"]:
                desc = (grp.get("descriptions") or {}).get(name, "")
                lines.append(f'  • "{name}"')
                if desc:
                    lines.append(f'    {desc[:400]}')
            parts.append("\n".join(lines))
        
        groups_block = "\n\n".join(parts)
        user_prompt = _LLM_PHASE3_TEMPLATE.format(groups_block=groups_block)
        print(f"    Batch {batch_idx}/{len(batches)}: {len(user_prompt):,} chars  |  {len(group_batch)} groups")

        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _LLM_PHASE3_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
                timeout=120,
            )
            raw = resp.choices[0].message.content or ""
            print(f"    Batch {batch_idx} response length: {len(raw)} chars")
        except Exception as e:
            print(f"  [!] Phase 3 Batch {batch_idx} LLM call failed: {e}")
            continue

        data = _parse_llm_json(raw, f"Phase 3 Batch {batch_idx}")
        if not isinstance(data, list):
            print(f"    Batch {batch_idx} parsed data is not list, type: {type(data)}")
            continue

        print(f"    Batch {batch_idx} raw response: {raw[:500]}...")  # DEBUG
        batch_merges = []
        for item in data:
            if not isinstance(item, dict):
                continue
            # LLM returns original group number, convert to batch-relative index
            original_group_num = item.get("group", 0)  # 1-indexed original number
            batch_start = (batch_idx - 1) * batch_size + 1  # first group number in this batch
            batch_end = batch_start + len(group_batch) - 1   # last group number in this batch
            
            print(f"    Batch {batch_idx}: Processing group {original_group_num}, batch range: {batch_start}-{batch_end}")  # DEBUG
            
            # Check if this group belongs to current batch
            if original_group_num < batch_start or original_group_num > batch_end:
                print(f"    Batch {batch_idx}: Ignoring group {original_group_num} (outside range {batch_start}-{batch_end})")
                continue
                
            # Convert to 0-indexed position within this batch
            gi = original_group_num - batch_start  # 0-indexed within batch
            if gi < 0 or gi >= len(group_batch):
                print(f"    Batch {batch_idx}: Invalid group index {gi} for group {original_group_num}")
                continue
                
            group_members_set = set(group_batch[gi]["members"])

            # New format: subgroups list — each subgroup is an independent merge
            subgroups = item.get("subgroups")
            if isinstance(subgroups, list):
                for sg in subgroups:
                    if not isinstance(sg, dict):
                        continue
                    sg_members = [m for m in sg.get("members", []) if m in group_members_set]
                    if len(sg_members) < 2:
                        continue
                    has_confidence = "confidence" in sg
                    confidence = float(sg.get("confidence", 1.0))
                    conf_tag = f"{confidence:.2f}" if has_confidence else f"N/A (defaulted to 1.0)"
                    if confidence < confidence_threshold:
                        print(f"      [~] SKIP  conf={conf_tag} < {confidence_threshold:.2f}: {sg_members}")
                        continue
                    llm_canonical = str(sg.get("canonical", "")).strip()
                    canonical = llm_canonical if llm_canonical in sg_members else pick_canonical(sg_members)
                    aliases = [n for n in sg_members if n != canonical]
                    if aliases:
                        print(f"      [+] MERGE conf={conf_tag}: \"{canonical}\" ← {aliases}")
                        batch_merges.append({"canonical": canonical, "aliases": aliases, "confidence": confidence})
            # Backward-compat: old format merge: true / false
            elif item.get("merge"):
                members = group_batch[gi]["members"]
                llm_canonical = str(item.get("canonical", "")).strip()
                canonical = llm_canonical if llm_canonical in members else pick_canonical(members)
                aliases = [n for n in members if n != canonical]
                if aliases:
                    batch_merges.append({"canonical": canonical, "aliases": aliases})
        
        print(f"    Batch {batch_idx} confirmed {len(batch_merges)} merges")
        all_merges.extend(batch_merges)
    
    print(f"    Total merges from all batches: {len(all_merges)}")
    return all_merges


async def run_llm_dedup(
    workspace: str,
    env_path: str,
    threshold: float,       # confidence threshold for Phase 3 LLM merges (0.0–1.0)
    embed_threshold: float, # cosine similarity threshold for Phase 2 embedding pre-filter
    apply: bool,
) -> None:
    """
    3-phase LLM deduplication pipeline:
      Phase 1 (rule-based)  : text-dup detection
      Phase 2 (embedding)   : cosine similarity pre-filter → candidate clusters
      Phase 3 (LLM strict)  : confirm each candidate with full descriptions
    """
    dry_run = not apply
    print(f"\n{'=' * 62}")
    print(f"  LLM-Based Semantic Deduplication  (3-phase)")
    print(f"  Workspace      : '{workspace}'")
    print(f"  Mode           : {'DRY RUN -- no changes' if dry_run else '[!] APPLY -- merging'}")
    print(f"  Embed threshold: >= {embed_threshold:.2f}  (Phase 2 cosine similarity)")
    print(f"  Confidence     : >= {threshold:.2f}  (Phase 3 LLM merge threshold)")
    print(f"  Env file       : {env_path}")
    print(f"{'=' * 62}\n")

    conn = await make_connection(env_path)
    try:
        entity_table, relation_table = await discover_tables(conn)
        graphml_files = discover_graphml_files(env_path, workspace)
        age_graphs    = await discover_age_graphs(conn, workspace)

        print(f"  Entity table : {entity_table}")
        if graphml_files:
            print(f"  GraphML files: {[os.path.basename(f) for f in graphml_files]}")
        print()

        # ── Phase 1: Rule-based text-dup ─────────────────────────────────────
        print("  [Phase 1] Text-dup detection (rule-based)...")
        all_entities = await fetch_entities(conn, workspace, entity_table)
        print(f"  Total entities: {len(all_entities)}")

        text_dup_groups = find_duplicate_groups(all_entities)
        text_dup_merges: list[dict] = []
        text_dup_alias_set: set[str] = set()
        for group in text_dup_groups:
            group_names = [e["entity_name"] for e in group["entities"]]
            canonical_g = group["canonical"]
            aliases_g   = [n for n in group_names if n != canonical_g]
            text_dup_merges.append({"canonical": canonical_g, "aliases": aliases_g})
            text_dup_alias_set.update(aliases_g)

        if text_dup_merges:
            print(f"  Found {len(text_dup_merges)} text-dup group(s):")
            for m in text_dup_merges:
                print(f"    \"{m['canonical']}\" ← {m['aliases']}")
        else:
            print("  No text duplicates found.")
        print()

        # Exclude text-dup aliases from LLM phases (already handled)
        semantic_names = [
            e["entity_name"] for e in all_entities
            if e["entity_name"] not in text_dup_alias_set
        ]
        print(f"  Entities for LLM: {len(semantic_names)} (after text-dup filter)\n")

        # ── Phase 2a: Embedding similarity pre-filter ────────────────────────
        print("  [Phase 2a] Embedding similarity pre-filter...")
        embed_candidates: list[dict] = []
        if not _NUMPY_OK:
            print("  [!] numpy required for embedding pre-filter (pip install numpy). Skipping.")
        else:
            vec_entities = await fetch_entities_with_vectors(conn, workspace, entity_table)
            semantic_name_set = set(semantic_names)
            vec_filtered = [e for e in vec_entities if e["entity_name"] in semantic_name_set]
            no_vec = len(semantic_names) - len(vec_filtered)
            print(f"  Entities with vectors : {len(vec_filtered)} / {len(semantic_names)}"
                  + (f"  ({no_vec} skipped — no vector)" if no_vec else ""))
            if len(vec_filtered) >= 2:
                clusters = _find_similar_clusters(vec_filtered, embed_threshold)
                print(f"  Embedding clusters    : {len(clusters)}  (cosine >= {embed_threshold:.2f})")
                embed_candidates = [
                    {"members": group, "reason": f"cosine similarity >= {embed_threshold:.2f}"}
                    for group in clusters
                ]
            else:
                print("  Not enough entities with vectors to compare.")
        print()

        # ── Phase 2b: LLM liberal verification on embedding clusters ─────────
        # Send the actual clusters from Phase 2a (not a flat list) so LLM can
        # confirm / split / reject each group with full context.
        valid_candidates: list[dict] = []
        if not embed_candidates:
            print("  Phase 2b: no embedding candidates — skipping LLM verification.\n")
        else:
            print(f"  [Phase 2b] LLM liberal verification — {len(embed_candidates)} embedding group(s)...")
            raw_phase2b = await _call_llm_phase2b_verify_clusters(
                embed_candidates, embed_threshold, env_path
            )

            # Validate: discard names not in embedding candidates (hallucinations)
            embed_name_set = {n for c in embed_candidates for n in c["members"]}
            hallucinated = 0
            for c in raw_phase2b:
                known = [m for m in c["members"] if m in embed_name_set]
                hallucinated += len(c["members"]) - len(known)
                if len(known) >= 2:
                    valid_candidates.append({"members": known, "reason": c["reason"]})
            if hallucinated:
                print(f"  [!] {hallucinated} hallucinated name(s) discarded")

            print(f"  Phase 2b: {len(valid_candidates)} candidate group(s) after LLM verification:")
            for i, c in enumerate(valid_candidates, 1):
                print(f"    [{i:02d}] {c['members']}")
                print(f"          reason: {c['reason']}")
            print()

        # ── Phase 3: Conservative confirmation ───────────────────────────────
        llm_merges: list[dict] = []
        if not valid_candidates:
            print("  Phase 3: skipped (no candidates from Phase 2).\n")
        else:
            print("  [Phase 3] Conservative confirmation — LLM reads full descriptions...")
            candidate_names = list({n for c in valid_candidates for n in c["members"]})
            descriptions = await fetch_entity_descriptions(
                conn, workspace, entity_table, candidate_names
            )
            enriched = [
                {**c, "descriptions": {n: descriptions.get(n, "") for n in c["members"]}}
                for c in valid_candidates
            ]
            llm_merges = await _call_llm_phase3_confirm(enriched, env_path,
                                                          confidence_threshold=threshold)
            if not llm_merges:
                print("  Phase 3: no merges confirmed.\n")
            else:
                print(f"  Phase 3: {len(llm_merges)} merge(s) confirmed\n")

        # ── Combine + deduplicate ─────────────────────────────────────────────
        all_merges = text_dup_merges + llm_merges
        seen_names: set[str] = set()
        deduped: list[dict] = []
        for m in all_merges:
            new_aliases = [a for a in m["aliases"] if a not in seen_names]
            if m["canonical"] in seen_names or not new_aliases:
                continue
            seen_names.add(m["canonical"])
            seen_names.update(new_aliases)
            deduped.append({"canonical": m["canonical"], "aliases": new_aliases})
        merges = deduped

        if not merges:
            print(f"  [OK] No merges needed.")
            print(f"{'=' * 62}\n")
            return

        print(f"  Total groups to merge: {len(merges)}  "
              f"(text-dup: {len(text_dup_merges)}, LLM-confirmed: {len(llm_merges)})\n")

        # ── Fetch entity records for all involved names (batch) ───────────────
        all_merge_names = list({n for m in merges for n in [m["canonical"]] + m["aliases"]})
        ws_clause, ws_params = _workspace_clause(workspace)
        p = len(ws_params)
        rows = await conn.fetch(
            f'SELECT id, entity_name, content, chunk_ids, file_path '
            f'FROM "{entity_table}" WHERE {ws_clause} AND entity_name = ANY(${p+1})',
            *ws_params, all_merge_names,
        )
        entity_by_name: dict[str, dict] = {dict(r)["entity_name"]: dict(r) for r in rows}

        # ── Show / apply ──────────────────────────────────────────────────────
        total_removed   = 0
        total_relations = 0

        for merge in merges:
            canonical = merge["canonical"]
            aliases   = merge["aliases"]
            keep = entity_by_name.get(canonical)
            if not keep:
                print(f"  [!] Canonical \"{canonical}\" not in VDB — skipping")
                continue
            drops   = [entity_by_name[a] for a in aliases if a in entity_by_name]
            missing = [a for a in aliases if a not in entity_by_name]
            if missing:
                print(f"  [!] Aliases not found in VDB: {missing}")

            src_r, tgt_r = await count_relation_refs(
                conn, workspace, keep["id"], relation_table, keep["entity_name"]
            )
            print(f"  Canonical: \"{canonical}\"")
            print(f"    [K] KEEP  : \"{canonical}\"  (refs: {src_r}-> {tgt_r}<-)")

            for drop in drops:
                src_d, tgt_d = await count_relation_refs(
                    conn, workspace, drop["id"], relation_table, drop["entity_name"]
                )
                print(f"    [M] MERGE : \"{drop['entity_name']}\"  (refs: {src_d}-> {tgt_d}<-)")
                total_relations += src_d + tgt_d

                if not dry_run:
                    updated = await apply_merge(
                        conn, workspace, keep, drop, canonical,
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
        "--fix-graphml", action="store_true",
        help="Sync the GraphML file with the VDB entity table (remove orphan nodes not in VDB).",
    )
    parser.add_argument(
        "--delete", nargs="+", metavar="NAME",
        help="Delete specific entity name(s) from all storage layers (use with --apply to execute).",
    )
    parser.add_argument(
        "--detect-similar", action="store_true",
        help="Find semantically similar entities using embedding cosine similarity (read-only).",
    )
    parser.add_argument(
        "--llm-dedup", action="store_true",
        help=(
            "Find similar clusters via cosine similarity, then ask LLM to decide "
            "which to merge. Dry-run by default; use --apply to execute merges."
        ),
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Similarity threshold for --detect-similar; LLM confidence threshold for --llm-dedup Phase 3 (default: 0.85).",
    )
    parser.add_argument(
        "--embed-threshold", type=float, default=0.88,
        help="Cosine similarity threshold for --llm-dedup Phase 2 embedding pre-filter (default: 0.88).",
    )
    parser.add_argument(
        "--purge-garbage", action="store_true",
        help=(
            "Delete entities matching garbage patterns (section headings, column headers, "
            "standalone codes, overly-long names). Dry-run by default; use --apply to execute."
        ),
    )
    parser.add_argument(
        "--list-nodes", action="store_true",
        help="List all entity nodes in the database for the specified workspace.",
    )
    args = parser.parse_args()

    if args.purge_garbage:
        asyncio.run(run_purge_garbage(
            workspace=args.workspace,
            env_path=args.env,
            apply=args.apply,
        ))
    elif args.llm_dedup:
        asyncio.run(run_llm_dedup(
            workspace=args.workspace,
            env_path=args.env,
            threshold=args.threshold,
            embed_threshold=args.embed_threshold,
            apply=args.apply,
        ))
    elif args.detect_similar:
        asyncio.run(run_detect_similar(
            workspace=args.workspace,
            env_path=args.env,
            threshold=args.threshold,
        ))
    elif args.delete:
        asyncio.run(run_delete(
            workspace=args.workspace,
            env_path=args.env,
            names=args.delete,
            apply=args.apply,
        ))
    elif args.fix_graphml:
        asyncio.run(run_fix_graphml(
            workspace=args.workspace,
            env_path=args.env,
        ))
    else:
        run(
            workspace=args.workspace,
            env_path=args.env,
            apply=args.apply,
        )


if __name__ == "__main__":
    main()
