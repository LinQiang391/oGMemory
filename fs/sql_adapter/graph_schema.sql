-- Heterogeneous Graph Schema for Multimodal Memory System
-- This schema extends the existing context engine with graph-based memory capabilities.

-- Graph Nodes Table
-- Stores all types of graph nodes: raw, extracted, and memory
CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id UUID PRIMARY KEY,
    node_type VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'active',

    -- Raw node specific fields
    raw_type VARCHAR(32),  -- Only for raw nodes (prompt, image, video, etc.)

    -- Content fields
    content TEXT,           -- Raw content or memory content
    summary TEXT,           -- Memory summary

    -- Heterogeneous properties (JSONB) for extensible attributes
    properties JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',  -- Additional metadata
    extracted_content JSONB NOT NULL DEFAULT '{}',  -- For extracted nodes

    -- Extracted node specific fields
    source_raw_id UUID,     -- Source raw node ID
    extractor_type VARCHAR(128),  -- Which extractor produced this
    extraction_schema VARCHAR(128),  -- Schema used for extraction
    extraction_status VARCHAR(32) DEFAULT 'pending',  -- pending, in_progress, completed, failed
    confidence DOUBLE PRECISION DEFAULT 0.0,  -- Extracted node confidence

    -- Memory node specific fields
    supporting_node_ids UUID[] NOT NULL DEFAULT '{}',  -- Array of supporting node IDs
    topic VARCHAR(256) DEFAULT '',
    tags TEXT[] NOT NULL DEFAULT '{}',
    parent_memory_id UUID,  -- For memory evolution

    -- PageRank
    pagerank DOUBLE PRECISION NOT NULL DEFAULT 0.0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Multi-tenant isolation
    account_id VARCHAR(128) NOT NULL,

    -- Constraints
    CONSTRAINT valid_node_type CHECK (node_type IN ('raw', 'extracted', 'memory')),
    CONSTRAINT valid_status CHECK (status IN ('active', 'archived', 'pending')),
    CONSTRAINT valid_extraction_status CHECK (
        extraction_status IN ('pending', 'in_progress', 'completed', 'failed')
    )
);

-- Indexes for graph_nodes
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_raw_type ON graph_nodes(raw_type)
    WHERE raw_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_graph_nodes_status ON graph_nodes(status);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_account_id ON graph_nodes(account_id);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_pagerank ON graph_nodes(pagerank DESC);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_topic ON graph_nodes(topic)
    WHERE topic != '';
CREATE INDEX IF NOT EXISTS idx_graph_nodes_updated_at ON graph_nodes(updated_at DESC);

-- Graph Edges Table
-- Stores directed edges between graph nodes
CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id UUID PRIMARY KEY,
    source_id UUID NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    edge_type VARCHAR(64) NOT NULL,

    -- Edge weight for PageRank calculation
    weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,

    -- Additional edge properties
    properties JSONB NOT NULL DEFAULT '{}',

    -- Multi-tenant isolation
    account_id VARCHAR(128) NOT NULL,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_edge_type CHECK (
        edge_type IN ('extracted_from', 'supports', 'references', 'evolves', 'co_occurs_in', 'same_identity')
    ),
    CONSTRAINT unique_edge UNIQUE (source_id, target_id, edge_type)
);

-- Indexes for graph_edges
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_graph_edges_account_id ON graph_edges(account_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source_target ON graph_edges(source_id, target_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_graph_node_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
DROP TRIGGER IF EXISTS trigger_graph_node_updated_at ON graph_nodes;
CREATE TRIGGER trigger_graph_node_updated_at
    BEFORE UPDATE ON graph_nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_graph_node_timestamp();

-- Function to calculate initial PageRank
-- This is a simplified PageRank using iterative method
CREATE OR REPLACE FUNCTION calculate_pagerank_incremental(
    p_node_id UUID,
    p_damping_factor DOUBLE PRECISION DEFAULT 0.85,
    p_max_iterations INTEGER DEFAULT 10
)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    v_new_pagerank DOUBLE PRECISION;
    v_incoming_edges INTEGER;
    v_outgoing_edges INTEGER;
BEGIN
    -- These helpers must also be tenant-scoped when invoked directly.
    -- An unset transaction identity fails closed, even for the table owner.
    IF NOT EXISTS (
        SELECT 1 FROM graph_nodes
        WHERE node_id = p_node_id
          AND account_id = current_setting('app.account_id', true)
    ) THEN
        RETURN 0.0;
    END IF;
    -- Count incoming edges
    SELECT COUNT(*) INTO v_incoming_edges
    FROM graph_edges
    WHERE target_id = p_node_id
      AND account_id = current_setting('app.account_id', true);

    -- Count outgoing edges
    SELECT COUNT(*) INTO v_outgoing_edges
    FROM graph_edges
    WHERE source_id = p_node_id
      AND account_id = current_setting('app.account_id', true);

    -- Base PageRank calculation
    IF v_incoming_edges > 0 AND v_outgoing_edges > 0 THEN
        v_new_pagerank := (1 - p_damping_factor) + p_damping_factor * (
            SELECT COALESCE(SUM(e.weight / NULLIF(
                (SELECT COUNT(*) FROM graph_edges e2 WHERE e2.source_id = e.source_id
                 AND e2.account_id = current_setting('app.account_id', true)), 0
            )), 0)
            FROM graph_edges e
            WHERE e.target_id = p_node_id
              AND e.account_id = current_setting('app.account_id', true)
        );
    ELSE
        v_new_pagerank := (1 - p_damping_factor);
    END IF;

    RETURN v_new_pagerank;
END;
$$ LANGUAGE plpgsql;

-- Function to get neighbors of a node
CREATE OR REPLACE FUNCTION get_node_neighbors(
    p_node_id UUID,
    p_edge_types TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    neighbor_id UUID,
    edge_id UUID,
    edge_type VARCHAR(64),
    weight DOUBLE PRECISION,
    direction VARCHAR(10)
) AS $$
BEGIN
    -- Get outgoing neighbors
    RETURN QUERY
    SELECT
        ge.target_id,
        ge.edge_id,
        ge.edge_type,
        ge.weight,
        'outgoing'::VARCHAR(10)
    FROM graph_edges ge
    JOIN graph_nodes src ON src.node_id = ge.source_id
    JOIN graph_nodes dst ON dst.node_id = ge.target_id
    WHERE ge.source_id = p_node_id
      AND ge.account_id = current_setting('app.account_id', true)
      AND src.account_id = ge.account_id AND dst.account_id = ge.account_id
      AND (p_edge_types IS NULL OR ge.edge_type = ANY(p_edge_types));

    -- Get incoming neighbors
    RETURN QUERY
    SELECT
        ge.source_id,
        ge.edge_id,
        ge.edge_type,
        ge.weight,
        'incoming'::VARCHAR(10)
    FROM graph_edges ge
    JOIN graph_nodes src ON src.node_id = ge.source_id
    JOIN graph_nodes dst ON dst.node_id = ge.target_id
    WHERE ge.target_id = p_node_id
      AND ge.account_id = current_setting('app.account_id', true)
      AND src.account_id = ge.account_id AND dst.account_id = ge.account_id
      AND (p_edge_types IS NULL OR ge.edge_type = ANY(p_edge_types));
END;
$$ LANGUAGE plpgsql;

-- Function to get supporting chain for a memory node
CREATE OR REPLACE FUNCTION get_supporting_chain(p_memory_id UUID)
RETURNS TABLE (
    node_id UUID,
    node_type VARCHAR(32),
    raw_type VARCHAR(32),
    content TEXT,
    summary TEXT,
    depth INTEGER
) AS $$
WITH RECURSIVE chain AS (
    -- Base case: the memory node itself
    SELECT
        gn.node_id,
        gn.node_type,
        gn.raw_type,
        gn.content,
        gn.summary,
        0 AS depth
    FROM graph_nodes gn
    WHERE gn.node_id = p_memory_id
      AND gn.account_id = current_setting('app.account_id', true)

    UNION ALL

    -- Recursive case:
    --   supports:       supporting extracted/raw node -> memory node
    --   extracted_from: extracted node -> source raw node
    SELECT
        gn.node_id,
        gn.node_type,
        gn.raw_type,
        gn.content,
        gn.summary,
        c.depth + 1 AS depth
    FROM chain c
    JOIN graph_edges ge ON (
        (ge.target_id = c.node_id AND ge.edge_type = 'supports')
        OR (ge.source_id = c.node_id AND ge.edge_type = 'extracted_from')
    )
    JOIN graph_nodes gn ON (
        (ge.edge_type = 'supports' AND gn.node_id = ge.source_id)
        OR (ge.edge_type = 'extracted_from' AND gn.node_id = ge.target_id)
    )
    WHERE gn.node_id != c.node_id
      AND gn.account_id = current_setting('app.account_id', true)
      AND ge.account_id = current_setting('app.account_id', true)
      AND c.depth < 5  -- Limit recursion depth
)
SELECT DISTINCT chain.node_id, chain.node_type, chain.raw_type, chain.content, chain.summary, chain.depth
FROM chain;
$$ LANGUAGE sql STABLE;
