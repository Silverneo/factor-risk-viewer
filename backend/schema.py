DDL = """
DROP TABLE IF EXISTS risk_contribution;
DROP TABLE IF EXISTS portfolio_risk;
DROP TABLE IF EXISTS factor_node;
DROP TABLE IF EXISTS portfolio_node;

CREATE TABLE portfolio_node (
    node_id           VARCHAR PRIMARY KEY,
    parent_id         VARCHAR,
    name              VARCHAR NOT NULL,
    level             SMALLINT NOT NULL,
    path              VARCHAR NOT NULL,
    is_leaf           BOOLEAN NOT NULL,
    weight_in_parent  DOUBLE
);

CREATE TABLE factor_node (
    node_id      VARCHAR PRIMARY KEY,
    parent_id    VARCHAR,
    name         VARCHAR NOT NULL,
    level        SMALLINT NOT NULL,
    path         VARCHAR NOT NULL,
    factor_type  VARCHAR NOT NULL,
    is_leaf      BOOLEAN NOT NULL
);

CREATE TABLE portfolio_risk (
    as_of_date        DATE NOT NULL,
    portfolio_node_id VARCHAR NOT NULL REFERENCES portfolio_node(node_id),
    total_vol         DOUBLE NOT NULL,
    factor_vol        DOUBLE NOT NULL,
    specific_vol      DOUBLE NOT NULL,
    PRIMARY KEY (as_of_date, portfolio_node_id)
);

CREATE TABLE risk_contribution (
    as_of_date        DATE NOT NULL,
    portfolio_node_id VARCHAR NOT NULL REFERENCES portfolio_node(node_id),
    factor_node_id    VARCHAR NOT NULL REFERENCES factor_node(node_id),
    exposure          DOUBLE,
    ctr_vol           DOUBLE NOT NULL,
    ctr_pct           DOUBLE NOT NULL,
    mctr              DOUBLE,
    PRIMARY KEY (as_of_date, portfolio_node_id, factor_node_id)
);

CREATE INDEX idx_rc_pf ON risk_contribution(portfolio_node_id);
CREATE INDEX idx_rc_fc ON risk_contribution(factor_node_id);
CREATE INDEX idx_rc_dt ON risk_contribution(as_of_date);

-- Upper-triangle factor covariance: factor_a <= factor_b lexically.
-- Diagonal is included (factor_a = factor_b -> variance).
CREATE TABLE factor_covariance (
    as_of_date  DATE    NOT NULL,
    factor_a    VARCHAR NOT NULL,
    factor_b    VARCHAR NOT NULL,
    cov         DOUBLE  NOT NULL,
    corr        DOUBLE  NOT NULL,
    PRIMARY KEY (as_of_date, factor_a, factor_b)
);
CREATE INDEX idx_fc_a ON factor_covariance(factor_a);
CREATE INDEX idx_fc_b ON factor_covariance(factor_b);
"""
