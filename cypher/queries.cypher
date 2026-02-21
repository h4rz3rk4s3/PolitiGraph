// ============================================================
// PolitiGraph PoC – Cypher Query Library
// Neo4j 5.x syntax
//
// Sections:
//   1. Data Exploration & Health Checks
//   2. Temporal / Longitudinal Queries
//   3. Topic Salience Queries
//   4. Complexity Metrics Queries
//   5. Entity & Relationship Queries
//   6. Graph Similarity & Pathfinding
//   7. Vector Search (Neo4j native)
// ============================================================


// ────────────────────────────────────────────────────────────
// SECTION 1: Data Exploration & Health Checks
// ────────────────────────────────────────────────────────────

// 1.1 Corpus overview: speeches per country per year
MATCH (s:Speech)
RETURN s.country_code AS country,
       s.year         AS year,
       count(s)       AS speech_count,
       count(DISTINCT s.party) AS parties_active
ORDER BY country, year;

// 1.2 Party representation over time (Germany example)
MATCH (s:Speech)
WHERE s.country_code = 'DE'
RETURN s.party AS party,
       s.year  AS year,
       count(s) AS speeches
ORDER BY year, speeches DESC;

// 1.3 Most prolific speakers
MATCH (pol:Politician)-[:DELIVERED_SPEECH]->(s:Speech)
WHERE s.country_code = $country_code
RETURN pol.name AS politician,
       pol.party AS party,
       count(s) AS speech_count
ORDER BY speech_count DESC
LIMIT 20;

// 1.4 Data completeness check (how many speeches have been through NLP extraction)
MATCH (s:Speech)
WHERE s.country_code = $country_code
RETURN
    count(s)                                        AS total_speeches,
    count(s.sentiment_score)                        AS with_sentiment,
    count(s.ttr)                                    AS with_complexity,
    count(s.embedding_model)                        AS with_embedding,
    round(100.0 * count(s.sentiment_score) / count(s), 1) AS pct_extracted;


// ────────────────────────────────────────────────────────────
// SECTION 2: Temporal / Longitudinal Queries
// ────────────────────────────────────────────────────────────

// 2.1 Average sentiment per party per year (longitudinal sentiment trend)
MATCH (s:Speech)
WHERE s.country_code = $country_code
  AND s.year >= $start_year
  AND s.year <= $end_year
  AND s.sentiment_score IS NOT NULL
RETURN
    s.year                          AS year,
    s.party                         AS party,
    round(avg(s.sentiment_score), 4) AS avg_sentiment,
    round(stdev(s.sentiment_score), 4) AS std_sentiment,
    count(s)                         AS speech_count
ORDER BY year, party;

// 2.2 Sentiment divergence between two parties over time
// (measures whether their rhetoric is converging or diverging emotionally)
MATCH (s1:Speech), (s2:Speech)
WHERE s1.country_code = $country_code
  AND s2.country_code = $country_code
  AND s1.party = $party_a
  AND s2.party = $party_b
  AND s1.year = s2.year
  AND s1.sentiment_score IS NOT NULL
  AND s2.sentiment_score IS NOT NULL
WITH s1.year AS year,
     avg(s1.sentiment_score) AS sentiment_a,
     avg(s2.sentiment_score) AS sentiment_b
RETURN year,
       round(sentiment_a, 4) AS avg_sentiment_party_a,
       round(sentiment_b, 4) AS avg_sentiment_party_b,
       round(abs(sentiment_a - sentiment_b), 4) AS sentiment_gap
ORDER BY year;

// 2.3 Speech volume by subcorpus (e.g. COVID vs. regular periods)
MATCH (s:Speech)
WHERE s.country_code = $country_code
RETURN coalesce(s.subcorpus, 'REGULAR') AS subcorpus,
       s.year AS year,
       count(s) AS speeches
ORDER BY year, subcorpus;


// ────────────────────────────────────────────────────────────
// SECTION 3: Topic Salience Queries
// ────────────────────────────────────────────────────────────

// 3.1 Top N most mentioned topics across the whole corpus
MATCH (s:Speech)-[:MENTIONS_TOPIC]->(t:Topic)
WHERE s.country_code = $country_code
RETURN t.label AS topic,
       count(s) AS total_mentions,
       count(DISTINCT s.party) AS parties_mentioning,
       round(avg(s.sentiment_score), 3) AS avg_sentiment_when_mentioned
ORDER BY total_mentions DESC
LIMIT 30;

// 3.2 Topic salience over time for a specific topic
MATCH (s:Speech)-[:MENTIONS_TOPIC]->(t:Topic {label: $topic_label})
WHERE s.country_code = $country_code
  AND s.year >= $start_year AND s.year <= $end_year
RETURN
    s.year  AS year,
    s.party AS party,
    count(s) AS mention_count,
    round(avg(s.sentiment_score), 4) AS avg_sentiment
ORDER BY year, mention_count DESC;

// 3.3 Topics that show the highest increase in mentions (2015 vs. 2023)
MATCH (s:Speech)-[:MENTIONS_TOPIC]->(t:Topic)
WHERE s.country_code = $country_code
WITH t.label AS topic,
     sum(CASE WHEN s.year = 2015 THEN 1 ELSE 0 END) AS count_2015,
     sum(CASE WHEN s.year = 2023 THEN 1 ELSE 0 END) AS count_2023
WHERE count_2015 > 0
RETURN topic,
       count_2015,
       count_2023,
       round(toFloat(count_2023 - count_2015) / count_2015 * 100, 1) AS pct_change
ORDER BY pct_change DESC
LIMIT 20;

// 3.4 Topic co-occurrence: which topics appear together in speeches
MATCH (s:Speech)-[:MENTIONS_TOPIC]->(t1:Topic),
      (s)-[:MENTIONS_TOPIC]->(t2:Topic)
WHERE s.country_code = $country_code
  AND id(t1) < id(t2)
RETURN t1.label AS topic_a,
       t2.label AS topic_b,
       count(s) AS co_occurrence_count
ORDER BY co_occurrence_count DESC
LIMIT 20;

// 3.5 Exclusive topics: topics that ONE party mentions significantly more than others
MATCH (s:Speech)-[:MENTIONS_TOPIC]->(t:Topic)
WHERE s.country_code = $country_code
  AND s.year >= $start_year
WITH t.label AS topic, s.party AS party, count(s) AS cnt
WITH topic,
     collect({party: party, count: cnt}) AS party_counts,
     sum(cnt) AS total
UNWIND party_counts AS pc
WITH topic, pc.party AS party, pc.count AS cnt, total
WHERE toFloat(cnt) / total > 0.5   // one party accounts for >50% of mentions
RETURN topic, party,
       round(toFloat(cnt) / total * 100, 1) AS share_pct,
       cnt AS party_mentions,
       total AS total_mentions
ORDER BY share_pct DESC;


// ────────────────────────────────────────────────────────────
// SECTION 4: Complexity Metrics Queries
// ────────────────────────────────────────────────────────────

// 4.1 Linguistic complexity trends per party
MATCH (s:Speech)
WHERE s.country_code = $country_code
  AND s.year >= $start_year
  AND s.ttr IS NOT NULL
RETURN
    s.year                              AS year,
    s.party                             AS party,
    round(avg(s.ttr), 4)               AS avg_ttr,
    round(avg(s.flesch_kincaid_grade), 2) AS avg_fk_grade,
    round(avg(s.avg_sentence_length), 2)  AS avg_sentence_length,
    round(avg(s.gunning_fog), 2)          AS avg_gunning_fog,
    count(s)                              AS speech_count
ORDER BY year, party;

// 4.2 Simplification index: year-over-year change in FK grade per party
// (Decreasing FK grade = rhetoric becoming more accessible / simpler)
MATCH (s:Speech)
WHERE s.country_code = $country_code
  AND s.flesch_kincaid_grade IS NOT NULL
WITH s.year AS year, s.party AS party, avg(s.flesch_kincaid_grade) AS avg_fk
WITH party,
     collect({year: year, fk: avg_fk}) AS trend
UNWIND trend AS t
WITH party, t.year AS year, t.fk AS fk_grade
ORDER BY party, year
WITH party, collect({year: year, fk: fk_grade}) AS series
RETURN party, series;

// 4.3 Outlier speeches: unusually simple language (potential populist markers)
MATCH (s:Speech)
WHERE s.country_code = $country_code
  AND s.flesch_kincaid_grade < 6.0   // grade school reading level
  AND s.word_count > 200             // not just a procedural remark
RETURN s.speech_id AS speech_id,
       s.party AS party,
       s.year AS year,
       round(s.flesch_kincaid_grade, 2) AS fk_grade,
       round(s.ttr, 4) AS ttr,
       s.word_count AS words
ORDER BY fk_grade ASC
LIMIT 50;


// ────────────────────────────────────────────────────────────
// SECTION 5: Entity & Relationship Queries
// ────────────────────────────────────────────────────────────

// 5.1 Most referenced named entities across the corpus
MATCH (s:Speech)-[r:REFERENCES_ENTITY]->(e:NamedEntity)
WHERE s.country_code = $country_code
  AND s.year >= $start_year
RETURN e.text AS entity,
       e.label AS entity_type,
       count(r) AS mention_count,
       count(DISTINCT s.party) AS parties_mentioning
ORDER BY mention_count DESC
LIMIT 30;

// 5.2 How a specific entity is discussed by different parties
MATCH (s:Speech)-[:REFERENCES_ENTITY]->(e:NamedEntity {text: $entity_text})
WHERE s.country_code = $country_code
  AND s.sentiment_score IS NOT NULL
RETURN
    s.party AS party,
    count(s) AS mentions,
    round(avg(s.sentiment_score), 3) AS avg_sentiment,
    s.year AS year
ORDER BY year, party;

// 5.3 Semantic triples: most common relations used by a party
MATCH (s:Speech)-[:CONTAINS_TRIPLE]->(subj:Concept)-[rel:SEMANTIC_RELATION]->(obj:Concept)
WHERE rel.speech_id IN [
    (pol:Politician {party: $party_name})-[:DELIVERED_SPEECH]->(sp:Speech {country_code: $country_code}) |
    sp.speech_id
]
RETURN rel.relation AS relation_type,
       count(*) AS frequency
ORDER BY frequency DESC
LIMIT 20;


// ────────────────────────────────────────────────────────────
// SECTION 6: Graph Similarity & Pathfinding
// ────────────────────────────────────────────────────────────

// 6.1 Politicians who have discussed the same topic (potential cross-party bridges)
MATCH (pol1:Politician)-[:DELIVERED_SPEECH]->(s1:Speech)-[:MENTIONS_TOPIC]->(t:Topic)<-
      [:MENTIONS_TOPIC]-(s2:Speech)<-[:DELIVERED_SPEECH]-(pol2:Politician)
WHERE pol1.party <> pol2.party
  AND pol1.country_code = $country_code
  AND id(pol1) < id(pol2)
RETURN pol1.name AS politician_a,
       pol1.party AS party_a,
       pol2.name AS politician_b,
       pol2.party AS party_b,
       collect(DISTINCT t.label) AS shared_topics,
       count(DISTINCT t) AS topic_overlap
ORDER BY topic_overlap DESC
LIMIT 20;

// 6.2 Temporal knowledge graph snapshot for a specific year
//     (useful for graph ML / GNN experiments)
MATCH (pol:Politician)-[:DELIVERED_SPEECH]->(s:Speech)-[:MENTIONS_TOPIC]->(t:Topic)
WHERE s.year = $year
  AND s.country_code = $country_code
RETURN pol.name AS speaker,
       pol.party AS party,
       s.speech_id AS speech_id,
       t.label AS topic,
       s.sentiment_score AS sentiment;


// ────────────────────────────────────────────────────────────
// SECTION 7: Vector Search (Neo4j native)
// ────────────────────────────────────────────────────────────

// 7.1 Find speeches most semantically similar to a given speech
//     (requires embedding stored in Speech.embedding)
MATCH (s:Speech {speech_id: $reference_speech_id})
CALL db.index.vector.queryNodes(
    'speech_embeddings',   // vector index name defined in schema.py
    10,                    // top-k results
    s.embedding
) YIELD node AS similar_speech, score
WHERE similar_speech.speech_id <> $reference_speech_id
RETURN similar_speech.speech_id AS speech_id,
       similar_speech.party     AS party,
       similar_speech.year      AS year,
       round(score, 4)          AS cosine_similarity
ORDER BY cosine_similarity DESC;

// 7.2 Cluster speeches around a topic using vector similarity
//     (find speeches thematically similar to "Immigration Control")
MATCH (t:Topic {label: 'Immigration Control'})<-[:MENTIONS_TOPIC]-(anchor:Speech)
WITH anchor LIMIT 1
CALL db.index.vector.queryNodes(
    'speech_embeddings',
    50,
    anchor.embedding
) YIELD node AS sim_speech, score
RETURN sim_speech.party AS party,
       sim_speech.year AS year,
       count(*) AS speeches_in_cluster,
       avg(score) AS avg_similarity
ORDER BY speeches_in_cluster DESC;
