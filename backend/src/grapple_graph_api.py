from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from neo4j import GraphDatabase
from pydantic import BaseModel, Field


HIDDEN_PROPERTY_KEYS = ["transcript", "rawText", "embedding", "vector", "text"]
CANONICAL_LABELS = ["CanonicalConcept", "CanonicalTechnique", "CanonicalDrill"]
STRUCTURAL_GRAPH_LABELS = ["Instructional", "Lesson", "SemanticSegment", "Document", "Chunk", "Session"]


class AccessibleGraphRequest(BaseModel):
    instructionalIds: List[str] = Field(default_factory=list)
    includeStructural: bool = False


class ElementDetailRequest(BaseModel):
    elementId: str


class InstructionalRequest(BaseModel):
    instructionalId: str


class SaveTranscriptRequest(BaseModel):
    instructionalId: str
    lessonId: str
    lessonTitle: str
    lessonOrder: int
    durationSeconds: int
    videoUrl: str
    transcript: str
    ownerUserId: str


class SaveSegmentsRequest(BaseModel):
    instructionalId: str
    lessonId: str
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    ownerUserId: str


class MarkLessonsIngestedRequest(BaseModel):
    rows: List[Dict[str, str]] = Field(default_factory=list)


class MarkLessonsSegmentedRequest(BaseModel):
    rows: List[Dict[str, str]] = Field(default_factory=list)


class SaveCanonicalEntitiesRequest(BaseModel):
    instructionalId: str
    concepts: List[Dict[str, Any]] = Field(default_factory=list)
    techniques: List[Dict[str, Any]] = Field(default_factory=list)
    drills: List[Dict[str, Any]] = Field(default_factory=list)
    ownerUserId: str
    contentRevision: int = 0


class UpdateLessonMetadataRequest(BaseModel):
    instructionalId: str
    lessonId: str
    title: Optional[str] = None
    order: Optional[int] = None
    volume: Optional[int] = None


class UpdateLessonOrdersRequest(BaseModel):
    instructionalId: str
    orders: List[Dict[str, Any]] = Field(default_factory=list)


def _to_plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_plain_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_to_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain_value(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "to_native"):
        native_value = value.to_native()
        if hasattr(native_value, "isoformat"):
            return native_value.isoformat()
        return native_value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _api_success(data: Any = None, message: Optional[str] = None) -> Dict[str, Any]:
    response = {"status": "Success", "data": _to_plain_value(data)}
    if message:
        response["message"] = message
    return response


def _api_failure(message: str, error: Exception) -> Dict[str, Any]:
    logging.exception("Grapple graph API failed: %s", message)
    return {"status": "Failed", "message": message, "error": str(error)}


def _append_unique(items: List[str], value: Optional[str]) -> None:
    if value and value not in items:
        items.append(value)


def _node_properties(node: Any) -> Dict[str, Any]:
    return {key: node.get(key) for key in node}


class GrappleGraphService:
    def __init__(self) -> None:
        self._driver = None

    @property
    def database(self) -> str:
        return (os.environ.get("NEO4J_DATABASE") or "neo4j").strip()

    def get_driver(self):
        if self._driver is None:
            uri = (os.environ.get("NEO4J_URI") or "").strip()
            username = (os.environ.get("NEO4J_USERNAME") or "").strip()
            password = (os.environ.get("NEO4J_PASSWORD") or "").strip()

            if not uri or not username or not password:
                raise RuntimeError("Neo4j credentials are not configured in Cloud Run")

            self._driver = GraphDatabase.driver(uri, auth=(username, password))
        return self._driver

    def run_query(self, cypher: str, params: Dict[str, Any]) -> None:
        with self.get_driver().session(database=self.database) as session:
            session.run(cypher, params).consume()

    def sanitize_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in (properties or {}).items()
            if key not in HIDDEN_PROPERTY_KEYS and value is not None
        }

    def normalize_name(self, value: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9\s]", "", value.lower()).split())

    def classify_node_provenance(self, labels: Optional[List[str]] = None) -> str:
        labels = labels or []
        if any(label in CANONICAL_LABELS for label in labels):
            return "canonical"
        if any(label in STRUCTURAL_GRAPH_LABELS for label in labels):
            return "structural"
        return "dynamic"

    def build_graph_summary(self, nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"canonical": 0, "dynamic": 0, "structural": 0}
        for node in nodes:
            counts[self.classify_node_provenance(node.get("labels"))] += 1

        return {
            "nodeCount": len(nodes),
            "relationshipCount": len(relationships),
            "canonicalCount": counts["canonical"],
            "dynamicCount": counts["dynamic"],
            "structuralCount": counts["structural"]
        }

    def get_accessible_graph(
        self,
        instructional_ids: List[str],
        include_structural: bool = False
    ) -> Dict[str, Any]:
        if not instructional_ids:
            nodes: List[Dict[str, Any]] = []
            relationships: List[Dict[str, Any]] = []
            return {"nodes": nodes, "relationships": relationships, "summary": self.build_graph_summary(nodes, relationships)}

        cypher = """
            MATCH (start:Instructional)
            WHERE start.instructionalId IN $instructionalIds
            WITH collect(start) AS startNodes
            CALL apoc.path.subgraphAll(
                startNodes,
                {
                    relationshipFilter: ">|<",
                    uniqueness: "NODE_GLOBAL"
                }
            ) YIELD nodes AS pathNodes, relationships AS pathRels
            UNWIND pathNodes AS n
            WITH n, pathRels
            WHERE $includeStructural OR none(label IN labels(n) WHERE label IN $structuralLabels)
            WITH collect(DISTINCT n) AS filteredNodes, pathRels
            RETURN
                [n IN filteredNodes | {
                    element_id: elementId(n),
                    labels: labels(n),
                    properties: apoc.map.removeKeys(properties(n), ["transcript", "rawText", "embedding", "vector", "text"])
                }] AS nodes,
                [r IN [rel IN pathRels WHERE startNode(rel) IN filteredNodes AND endNode(rel) IN filteredNodes] | {
                    element_id: elementId(r),
                    type: type(r),
                    start_node_element_id: elementId(startNode(r)),
                    end_node_element_id: elementId(endNode(r))
                }] AS relationships
        """

        with self.get_driver().session(database=self.database) as session:
            result = session.run(
                cypher,
                {
                    "instructionalIds": instructional_ids,
                    "includeStructural": include_structural,
                    "structuralLabels": STRUCTURAL_GRAPH_LABELS
                }
            )
            record = result.single()

        if not record:
            nodes = []
            relationships = []
        else:
            nodes = [
                {**node, "properties": self.sanitize_properties(node.get("properties", {}))}
                for node in (record.get("nodes") or [])
            ]
            relationships = record.get("relationships") or []

        return {"nodes": nodes, "relationships": relationships, "summary": self.build_graph_summary(nodes, relationships)}

    def get_graph_element_detail(self, element_id: str) -> Optional[Dict[str, Any]]:
        node_cypher = """
            MATCH (n)
            WHERE elementId(n) = $elementId
            CALL {
                WITH n
                OPTIONAL MATCH (n)-[adj]-()
                RETURN collect(type(adj)) AS adjacentTypes
            }
            CALL {
                WITH n
                OPTIONAL MATCH (n)-[:SOURCED_FROM]->(s:SemanticSegment)
                RETURN collect(DISTINCT {
                    instructionalId: s.instructionalId,
                    lessonId: s.lessonId,
                    segmentId: s.segmentId,
                    title: s.title,
                    type: s.type
                }) AS sourceRefs
            }
            CALL {
                WITH n
                OPTIONAL MATCH (n)-[*0..2]-(i:Instructional)
                WITH collect(DISTINCT i.instructionalId) AS instructionalIds
                RETURN [instructionalId IN instructionalIds WHERE instructionalId IS NOT NULL] AS linkedInstructionalIds
            }
            CALL {
                WITH n
                OPTIONAL MATCH (n)-[*0..2]-(l:Lesson)
                WITH collect(DISTINCT l.lessonId) AS lessonIds
                RETURN [lessonId IN lessonIds WHERE lessonId IS NOT NULL] AS linkedLessonIds
            }
            RETURN {
                elementType: "node",
                elementId: elementId(n),
                labels: labels(n),
                properties: apoc.map.removeKeys(properties(n), ["transcript", "rawText", "embedding", "vector", "text"]),
                adjacentTypes: adjacentTypes,
                sourceRefs: sourceRefs,
                linkedInstructionalIds: linkedInstructionalIds,
                linkedLessonIds: linkedLessonIds
            } AS detail
        """

        with self.get_driver().session(database=self.database) as session:
            node_result = session.run(node_cypher, {"elementId": element_id})
            node_record = node_result.single()

            if node_record:
                detail = node_record.get("detail")
                adjacent_types = [adjacent_type for adjacent_type in (detail.get("adjacentTypes") or []) if adjacent_type]
                by_type: Dict[str, int] = {}
                for adjacent_type in adjacent_types:
                    by_type[adjacent_type] = by_type.get(adjacent_type, 0) + 1

                return {
                    "elementType": "node",
                    "elementId": detail.get("elementId"),
                    "labels": detail.get("labels"),
                    "properties": self.sanitize_properties(detail.get("properties") or {}),
                    "provenance": self.classify_node_provenance(detail.get("labels")),
                    "adjacentCounts": {"total": len(adjacent_types), "byType": by_type},
                    "sourceRefs": [
                        source
                        for source in (detail.get("sourceRefs") or [])
                        if source and (source.get("segmentId") or source.get("lessonId") or source.get("instructionalId"))
                    ],
                    "linkedInstructionalIds": [item for item in (detail.get("linkedInstructionalIds") or []) if item],
                    "linkedLessonIds": [item for item in (detail.get("linkedLessonIds") or []) if item]
                }

            relationship_cypher = """
                MATCH (start)-[r]->(end)
                WHERE elementId(r) = $elementId
                CALL {
                    WITH start
                    OPTIONAL MATCH (start)-[*0..2]-(i:Instructional)
                    WITH collect(DISTINCT i.instructionalId) AS instructionalIds
                    RETURN [instructionalId IN instructionalIds WHERE instructionalId IS NOT NULL] AS linkedInstructionalIds
                }
                CALL {
                    WITH end
                    OPTIONAL MATCH (end)-[*0..2]-(l:Lesson)
                    WITH collect(DISTINCT l.lessonId) AS lessonIds
                    RETURN [lessonId IN lessonIds WHERE lessonId IS NOT NULL] AS linkedLessonIds
                }
                RETURN {
                    elementType: "relationship",
                    elementId: elementId(r),
                    type: type(r),
                    properties: apoc.map.removeKeys(properties(r), ["embedding", "vector"]),
                    startLabels: labels(start),
                    endLabels: labels(end),
                    linkedInstructionalIds: linkedInstructionalIds,
                    linkedLessonIds: linkedLessonIds
                } AS detail
            """

            relationship_result = session.run(relationship_cypher, {"elementId": element_id})
            relationship_record = relationship_result.single()

        if not relationship_record:
            return None

        detail = relationship_record.get("detail")
        start_provenance = self.classify_node_provenance(detail.get("startLabels"))
        end_provenance = self.classify_node_provenance(detail.get("endLabels"))
        if start_provenance == "structural" or end_provenance == "structural":
            provenance = "structural"
        elif start_provenance == "canonical" and end_provenance == "canonical":
            provenance = "canonical"
        else:
            provenance = "dynamic"

        return {
            "elementType": "relationship",
            "elementId": detail.get("elementId"),
            "type": detail.get("type"),
            "properties": self.sanitize_properties(detail.get("properties") or {}),
            "provenance": provenance,
            "adjacentCounts": {"total": 2},
            "sourceRefs": [],
            "linkedInstructionalIds": [item for item in (detail.get("linkedInstructionalIds") or []) if item],
            "linkedLessonIds": [item for item in (detail.get("linkedLessonIds") or []) if item]
        }

    def save_transcript_to_neo4j(self, params: Dict[str, Any]) -> None:
        transcript = params["transcript"]
        transcript_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()

        cypher = """
            MERGE (i:Instructional {instructionalId: $instructionalId})
            SET i.ownerUserId = $ownerUserId,
                i.createdAt = coalesce(i.createdAt, datetime()),
                i.updatedAt = datetime()
            MERGE (l:Lesson {lessonId: $lessonId})
            SET l.instructionalId = $instructionalId,
                l.title = $lessonTitle,
                l.order = $lessonOrder,
                l.durationSeconds = $durationSeconds,
                l.videoUrl = $videoUrl,
                l.transcript = $transcript,
                l.transcriptHash = $transcriptHash,
                l.ownerUserId = $ownerUserId,
                l.createdAt = coalesce(l.createdAt, datetime()),
                l.updatedAt = datetime()
            MERGE (i)-[:HAS_LESSON]->(l)
        """

        self.run_query(
            cypher,
            {
                **params,
                "transcriptHash": transcript_hash
            }
        )

    def save_segments_to_neo4j(self, params: Dict[str, Any]) -> None:
        instructional_id = params["instructionalId"]
        lesson_id = params["lessonId"]
        segments = params.get("segments") or []
        owner_user_id = params["ownerUserId"]

        with self.get_driver().session(database=self.database) as session:
            session.run(
                """
                    MATCH (l:Lesson {lessonId: $lessonId})-[:HAS_SEGMENT]->(s:SemanticSegment)
                    DETACH DELETE s
                """,
                {"lessonId": lesson_id}
            ).consume()

            for segment in segments:
                session.run(
                    """
                        MATCH (l:Lesson {lessonId: $lessonId})
                        CREATE (s:SemanticSegment {
                            segmentId: $segmentId,
                            lessonId: $lessonId,
                            instructionalId: $instructionalId,
                            ownerUserId: $ownerUserId,
                            type: $type,
                            title: $title,
                            briefSummary: $briefSummary,
                            rawText: $rawText,
                            createdAt: datetime()
                        })
                        CREATE (l)-[:HAS_SEGMENT]->(s)
                    """,
                    {
                        "segmentId": segment.get("id"),
                        "lessonId": lesson_id,
                        "instructionalId": instructional_id,
                        "ownerUserId": owner_user_id,
                        "type": segment.get("type"),
                        "title": segment.get("title"),
                        "briefSummary": segment.get("briefSummary"),
                        "rawText": segment.get("rawText")
                    }
                ).consume()

            for index in range(len(segments) - 1):
                session.run(
                    """
                        MATCH (s1:SemanticSegment {segmentId: $segmentId1})
                        MATCH (s2:SemanticSegment {segmentId: $segmentId2})
                        CREATE (s1)-[:NEXT]->(s2)
                    """,
                    {
                        "segmentId1": segments[index].get("id"),
                        "segmentId2": segments[index + 1].get("id")
                    }
                ).consume()

    def get_lesson_transcripts_by_instructional(self, instructional_id: str) -> List[Dict[str, Any]]:
        cypher = """
            MATCH (i:Instructional {instructionalId: $instructionalId})-[:HAS_LESSON]->(l:Lesson)
            RETURN
                l.lessonId as lessonId,
                l.title as title,
                coalesce(l.transcript, "") as transcript,
                coalesce(l.transcriptHash, "") as transcriptHash,
                coalesce(l.lastIngestedHash, "") as lastIngestedHash,
                coalesce(l.lastSegmentedHash, "") as lastSegmentedHash,
                coalesce(l.graphDocumentName, "") as graphDocumentName,
                coalesce(l.order, 0) as order
            ORDER BY l.order, l.lessonId
        """
        with self.get_driver().session(database=self.database) as session:
            result = session.run(cypher, {"instructionalId": instructional_id})
            return [
                {
                    "lessonId": record.get("lessonId"),
                    "title": record.get("title"),
                    "transcript": record.get("transcript") or "",
                    "transcriptHash": record.get("transcriptHash") or "",
                    "lastIngestedHash": record.get("lastIngestedHash") or None,
                    "lastSegmentedHash": record.get("lastSegmentedHash") or None,
                    "graphDocumentName": record.get("graphDocumentName") or None,
                    "order": int(record.get("order") or 0)
                }
                for record in result
            ]

    def mark_lessons_ingested(self, rows: List[Dict[str, str]]) -> None:
        if not rows:
            return
        self.run_query(
            """
                UNWIND $rows AS row
                MATCH (l:Lesson {lessonId: row.lessonId})
                SET l.lastIngestedHash = row.transcriptHash,
                    l.graphDocumentName = row.graphDocumentName,
                    l.updatedAt = datetime()
            """,
            {"rows": rows}
        )

    def mark_lessons_segmented(self, rows: List[Dict[str, str]]) -> None:
        if not rows:
            return
        self.run_query(
            """
                UNWIND $rows AS row
                MATCH (l:Lesson {lessonId: row.lessonId})
                SET l.lastSegmentedHash = row.transcriptHash,
                    l.updatedAt = datetime()
            """,
            {"rows": rows}
        )

    def save_canonical_entities(self, params: Dict[str, Any]) -> Dict[str, List[str]]:
        instructional_id = params["instructionalId"]
        concepts = params.get("concepts") or []
        techniques = params.get("techniques") or []
        drills = params.get("drills") or []
        owner_user_id = params["ownerUserId"]
        content_revision = params.get("contentRevision") or 0
        added_entity_ids: List[str] = []
        updated_entity_ids: List[str] = []

        with self.get_driver().session(database=self.database) as session:
            for concept in concepts:
                upsert = session.run(
                    """
                        MATCH (i:Instructional {instructionalId: $instructionalId})
                        OPTIONAL MATCH (existing:CanonicalConcept {
                            instructionalId: $instructionalId,
                            normalizedName: $normalizedName
                        })
                        WITH i, existing
                        CALL {
                            WITH i, existing
                            WHERE existing IS NULL
                            CREATE (c:CanonicalConcept {
                                id: $id,
                                instructionalId: $instructionalId,
                                ownerUserId: $ownerUserId,
                                name: $name,
                                normalizedName: $normalizedName,
                                fingerprintText: $fingerprintText,
                                shortDefinition: $shortDefinition,
                                detailedExplanation: $detailedExplanation,
                                context: $context,
                                firstSeenRevision: $contentRevision,
                                lastUpdatedRevision: $contentRevision,
                                createdAt: datetime(),
                                updatedAt: datetime()
                            })
                            MERGE (i)-[:HAS_CONCEPT]->(c)
                            RETURN c, true AS created
                            UNION
                            WITH existing
                            WHERE existing IS NOT NULL
                            SET existing.name = $name,
                                existing.normalizedName = $normalizedName,
                                existing.fingerprintText = $fingerprintText,
                                existing.shortDefinition = $shortDefinition,
                                existing.detailedExplanation = $detailedExplanation,
                                existing.context = $context,
                                existing.lastUpdatedRevision = $contentRevision,
                                existing.updatedAt = datetime()
                            RETURN existing AS c, false AS created
                        }
                        RETURN c.id AS canonicalId, created
                    """,
                    {
                        "id": concept.get("id"),
                        "instructionalId": instructional_id,
                        "ownerUserId": owner_user_id,
                        "name": concept.get("name"),
                        "normalizedName": self.normalize_name(concept.get("name") or ""),
                        "fingerprintText": " | ".join(
                            item for item in [
                                concept.get("name"),
                                concept.get("shortDefinition"),
                                concept.get("detailedExplanation") or "",
                                concept.get("context") or ""
                            ] if item
                        ),
                        "shortDefinition": concept.get("shortDefinition"),
                        "detailedExplanation": concept.get("detailedExplanation") or None,
                        "context": concept.get("context") or None,
                        "contentRevision": content_revision
                    }
                )
                record = upsert.single()
                if not record:
                    raise RuntimeError("No Instructional node found while saving canonical concept")
                canonical_id = record.get("canonicalId")
                _append_unique(added_entity_ids if record.get("created") else updated_entity_ids, canonical_id)

                session.run(
                    """
                        MATCH (c:CanonicalConcept {id: $canonicalId})
                        OPTIONAL MATCH (c)-[rel:SOURCED_FROM]->()
                        DELETE rel
                    """,
                    {"canonicalId": canonical_id}
                ).consume()

                for source in concept.get("sources") or []:
                    session.run(
                        """
                            MATCH (c:CanonicalConcept {id: $canonicalId})
                            MATCH (s:SemanticSegment {segmentId: $segmentId})
                            MERGE (c)-[:SOURCED_FROM]->(s)
                        """,
                        {"canonicalId": canonical_id, "segmentId": source.get("segmentId")}
                    ).consume()

            for technique in techniques:
                upsert = session.run(
                    """
                        MATCH (i:Instructional {instructionalId: $instructionalId})
                        OPTIONAL MATCH (existing:CanonicalTechnique {
                            instructionalId: $instructionalId,
                            normalizedName: $normalizedName
                        })
                        WITH i, existing
                        CALL {
                            WITH i, existing
                            WHERE existing IS NULL
                            CREATE (t:CanonicalTechnique {
                                id: $id,
                                instructionalId: $instructionalId,
                                ownerUserId: $ownerUserId,
                                name: $name,
                                normalizedName: $normalizedName,
                                fingerprintText: $fingerprintText,
                                type: $type,
                                positionStart: $positionStart,
                                positionEnd: $positionEnd,
                                steps: $steps,
                                firstSeenRevision: $contentRevision,
                                lastUpdatedRevision: $contentRevision,
                                createdAt: datetime(),
                                updatedAt: datetime()
                            })
                            MERGE (i)-[:HAS_TECHNIQUE]->(t)
                            RETURN t, true AS created
                            UNION
                            WITH existing
                            WHERE existing IS NOT NULL
                            SET existing.name = $name,
                                existing.normalizedName = $normalizedName,
                                existing.fingerprintText = $fingerprintText,
                                existing.type = $type,
                                existing.positionStart = $positionStart,
                                existing.positionEnd = $positionEnd,
                                existing.steps = $steps,
                                existing.lastUpdatedRevision = $contentRevision,
                                existing.updatedAt = datetime()
                            RETURN existing AS t, false AS created
                        }
                        RETURN t.id AS canonicalId, created
                    """,
                    {
                        "id": technique.get("id"),
                        "instructionalId": instructional_id,
                        "ownerUserId": owner_user_id,
                        "name": technique.get("name"),
                        "normalizedName": self.normalize_name(technique.get("name") or ""),
                        "fingerprintText": " | ".join(
                            item for item in [
                                technique.get("name"),
                                technique.get("positionStart") or "",
                                technique.get("positionEnd") or "",
                                json.dumps(technique.get("steps") or [])
                            ] if item
                        ),
                        "type": technique.get("type"),
                        "positionStart": technique.get("positionStart") or None,
                        "positionEnd": technique.get("positionEnd") or None,
                        "steps": json.dumps(technique.get("steps") or []),
                        "contentRevision": content_revision
                    }
                )
                record = upsert.single()
                if not record:
                    raise RuntimeError("No Instructional node found while saving canonical technique")
                canonical_id = record.get("canonicalId")
                _append_unique(added_entity_ids if record.get("created") else updated_entity_ids, canonical_id)

                session.run(
                    """
                        MATCH (t:CanonicalTechnique {id: $canonicalId})
                        OPTIONAL MATCH (t)-[source:SOURCED_FROM]->()
                        DELETE source
                    """,
                    {"canonicalId": canonical_id}
                ).consume()

                for source in technique.get("sources") or []:
                    session.run(
                        """
                            MATCH (t:CanonicalTechnique {id: $canonicalId})
                            MATCH (s:SemanticSegment {segmentId: $segmentId})
                            MERGE (t)-[:SOURCED_FROM]->(s)
                        """,
                        {"canonicalId": canonical_id, "segmentId": source.get("segmentId")}
                    ).consume()

            for drill in drills:
                upsert = session.run(
                    """
                        MATCH (i:Instructional {instructionalId: $instructionalId})
                        OPTIONAL MATCH (existing:CanonicalDrill {
                            instructionalId: $instructionalId,
                            normalizedName: $normalizedName
                        })
                        WITH i, existing
                        CALL {
                            WITH i, existing
                            WHERE existing IS NULL
                            CREATE (d:CanonicalDrill {
                                id: $id,
                                instructionalId: $instructionalId,
                                ownerUserId: $ownerUserId,
                                name: $name,
                                normalizedName: $normalizedName,
                                fingerprintText: $fingerprintText,
                                goal: $goal,
                                rolesDescription: $rolesDescription,
                                constraints: $constraints,
                                repScheme: $repScheme,
                                firstSeenRevision: $contentRevision,
                                lastUpdatedRevision: $contentRevision,
                                createdAt: datetime(),
                                updatedAt: datetime()
                            })
                            MERGE (i)-[:HAS_DRILL]->(d)
                            RETURN d, true AS created
                            UNION
                            WITH existing
                            WHERE existing IS NOT NULL
                            SET existing.name = $name,
                                existing.normalizedName = $normalizedName,
                                existing.fingerprintText = $fingerprintText,
                                existing.goal = $goal,
                                existing.rolesDescription = $rolesDescription,
                                existing.constraints = $constraints,
                                existing.repScheme = $repScheme,
                                existing.lastUpdatedRevision = $contentRevision,
                                existing.updatedAt = datetime()
                            RETURN existing AS d, false AS created
                        }
                        RETURN d.id AS canonicalId, created
                    """,
                    {
                        "id": drill.get("id"),
                        "instructionalId": instructional_id,
                        "ownerUserId": owner_user_id,
                        "name": drill.get("name"),
                        "normalizedName": self.normalize_name(drill.get("name") or ""),
                        "fingerprintText": " | ".join(
                            item for item in [
                                drill.get("name"),
                                drill.get("goal"),
                                drill.get("rolesDescription") or "",
                                drill.get("constraints") or "",
                                drill.get("repScheme") or ""
                            ] if item
                        ),
                        "goal": drill.get("goal"),
                        "rolesDescription": drill.get("rolesDescription") or None,
                        "constraints": drill.get("constraints") or None,
                        "repScheme": drill.get("repScheme") or None,
                        "contentRevision": content_revision
                    }
                )
                record = upsert.single()
                if not record:
                    raise RuntimeError("No Instructional node found while saving canonical drill")
                canonical_id = record.get("canonicalId")
                _append_unique(added_entity_ids if record.get("created") else updated_entity_ids, canonical_id)

                session.run(
                    """
                        MATCH (d:CanonicalDrill {id: $canonicalId})
                        OPTIONAL MATCH (d)-[source:SOURCED_FROM]->()
                        DELETE source
                    """,
                    {"canonicalId": canonical_id}
                ).consume()

                for source in drill.get("sources") or []:
                    session.run(
                        """
                            MATCH (d:CanonicalDrill {id: $canonicalId})
                            MATCH (s:SemanticSegment {segmentId: $segmentId})
                            MERGE (d)-[:SOURCED_FROM]->(s)
                        """,
                        {"canonicalId": canonical_id, "segmentId": source.get("segmentId")}
                    ).consume()

            def delete_orphans(label: str) -> List[str]:
                result = session.run(
                    f"""
                        MATCH (n:{label} {{instructionalId: $instructionalId}})
                        WHERE NOT (n)-[:SOURCED_FROM]->(:SemanticSegment)
                        RETURN collect(n.id) AS ids
                    """,
                    {"instructionalId": instructional_id}
                )
                ids = [item for item in (result.single().get("ids") or []) if item]
                if ids:
                    session.run(f"MATCH (n:{label}) WHERE n.id IN $ids DETACH DELETE n", {"ids": ids}).consume()
                return ids

            deleted_entity_ids = [
                *delete_orphans("CanonicalConcept"),
                *delete_orphans("CanonicalTechnique"),
                *delete_orphans("CanonicalDrill")
            ]

        affected_entity_ids: List[str] = []
        for entity_id in [*added_entity_ids, *updated_entity_ids, *deleted_entity_ids]:
            _append_unique(affected_entity_ids, entity_id)

        return {
            "addedEntityIds": added_entity_ids,
            "updatedEntityIds": updated_entity_ids,
            "deletedEntityIds": deleted_entity_ids,
            "affectedEntityIds": affected_entity_ids
        }

    def get_canonical_entities(self, instructional_id: str) -> Dict[str, List[Dict[str, Any]]]:
        with self.get_driver().session(database=self.database) as session:
            concepts_result = session.run(
                """
                MATCH (c:CanonicalConcept {instructionalId: $instructionalId})
                OPTIONAL MATCH (c)-[:SOURCED_FROM]->(s:SemanticSegment)
                RETURN c, collect(DISTINCT {segmentId: s.segmentId, extractedId: null}) as sources
                """,
                {"instructionalId": instructional_id}
            )
            concepts = []
            for record in concepts_result:
                node = _node_properties(record.get("c"))
                sources = record.get("sources") or []
                concept = {
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "shortDefinition": node.get("shortDefinition"),
                    "relatedConceptIds": [],
                    "sources": [source for source in sources if source and source.get("segmentId")]
                }
                if node.get("detailedExplanation"):
                    concept["detailedExplanation"] = node.get("detailedExplanation")
                if node.get("context"):
                    concept["context"] = node.get("context")
                concepts.append(concept)

            techniques_result = session.run(
                """
                MATCH (t:CanonicalTechnique {instructionalId: $instructionalId})
                OPTIONAL MATCH (t)-[:SOURCED_FROM]->(s:SemanticSegment)
                OPTIONAL MATCH (t)-[:USES_CONCEPT]->(c:CanonicalConcept)
                RETURN t, collect(DISTINCT {segmentId: s.segmentId, extractedId: null}) as sources, collect(DISTINCT c.id) as conceptIds
                """,
                {"instructionalId": instructional_id}
            )
            techniques = []
            for record in techniques_result:
                node = _node_properties(record.get("t"))
                sources = record.get("sources") or []
                concept_ids = [item for item in (record.get("conceptIds") or []) if item]
                steps = []
                if node.get("steps"):
                    try:
                        steps = json.loads(node.get("steps")) if isinstance(node.get("steps"), str) else node.get("steps")
                    except json.JSONDecodeError:
                        steps = []
                technique = {
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "type": node.get("type"),
                    "steps": steps,
                    "sources": [source for source in sources if source and source.get("segmentId")]
                }
                if node.get("positionStart"):
                    technique["positionStart"] = node.get("positionStart")
                if node.get("positionEnd"):
                    technique["positionEnd"] = node.get("positionEnd")
                if concept_ids:
                    technique["conceptIds"] = concept_ids
                techniques.append(technique)

            drills_result = session.run(
                """
                MATCH (d:CanonicalDrill {instructionalId: $instructionalId})
                OPTIONAL MATCH (d)-[:SOURCED_FROM]->(s:SemanticSegment)
                OPTIONAL MATCH (d)-[:TRAINS_TECHNIQUE]->(t:CanonicalTechnique)
                OPTIONAL MATCH (d)-[:APPLIES_CONCEPT]->(c:CanonicalConcept)
                RETURN d, collect(DISTINCT {segmentId: s.segmentId, extractedId: null}) as sources,
                       collect(DISTINCT t.id) as techniqueIds, collect(DISTINCT c.id) as conceptIds
                """,
                {"instructionalId": instructional_id}
            )
            drills = []
            for record in drills_result:
                node = _node_properties(record.get("d"))
                sources = record.get("sources") or []
                technique_ids = [item for item in (record.get("techniqueIds") or []) if item]
                concept_ids = [item for item in (record.get("conceptIds") or []) if item]
                drill = {
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "goal": node.get("goal"),
                    "sources": [source for source in sources if source and source.get("segmentId")]
                }
                if node.get("rolesDescription"):
                    drill["rolesDescription"] = node.get("rolesDescription")
                if node.get("constraints"):
                    drill["constraints"] = node.get("constraints")
                if node.get("repScheme"):
                    drill["repScheme"] = node.get("repScheme")
                if technique_ids:
                    drill["techniqueIds"] = technique_ids
                if concept_ids:
                    drill["conceptIds"] = concept_ids
                drills.append(drill)

        return {"concepts": concepts, "techniques": techniques, "drills": drills}

    def get_lessons_by_instructional(self, instructional_id: str) -> List[Dict[str, Any]]:
        cypher = """
            MATCH (i:Instructional {instructionalId: $instructionalId})-[:HAS_LESSON]->(l:Lesson)
            RETURN
                l.lessonId as lessonId,
                l.instructionalId as instructionalId,
                l.title as title,
                coalesce(l.order, 0) as order,
                coalesce(l.durationSeconds, 0) as durationSeconds,
                coalesce(l.videoUrl, "") as videoUrl,
                coalesce(l.transcript, "") as transcript,
                coalesce(l.volume, 1) as volume,
                coalesce(l.createdAt, "") as createdAt,
                coalesce(l.updatedAt, l.createdAt, "") as updatedAt
            ORDER BY l.order, l.lessonId
        """
        with self.get_driver().session(database=self.database) as session:
            result = session.run(cypher, {"instructionalId": instructional_id})
            return [
                {
                    "id": record.get("lessonId"),
                    "instructionalId": record.get("instructionalId") or instructional_id,
                    "title": record.get("title"),
                    "order": int(record.get("order") or 0),
                    "durationSeconds": int(record.get("durationSeconds") or 0),
                    "videoUrl": record.get("videoUrl") or "",
                    "transcript": record.get("transcript") or "",
                    "volume": int(record.get("volume") or 1),
                    "createdAt": _to_plain_value(record.get("createdAt") or ""),
                    "updatedAt": _to_plain_value(record.get("updatedAt") or "")
                }
                for record in result
            ]

    def link_instructional_to_documents(self, instructional_id: str) -> None:
        self.run_query(
            """
                MATCH (i:Instructional {instructionalId: $instructionalId})
                MATCH (d:Document)
                WHERE d.fileName STARTS WITH $instructionalId
                MERGE (i)-[:HAS_DOCUMENT]->(d)
            """,
            {"instructionalId": instructional_id}
        )

    def update_lesson_metadata(self, params: Dict[str, Any]) -> None:
        set_clauses = []
        values: Dict[str, Any] = {
            "instructionalId": params["instructionalId"],
            "lessonId": params["lessonId"]
        }

        if params.get("title") is not None:
            set_clauses.append("l.title = $title")
            values["title"] = params.get("title")
        if params.get("order") is not None:
            set_clauses.append("l.order = $order")
            values["order"] = params.get("order")
        if params.get("volume") is not None:
            set_clauses.append("l.volume = $volume")
            values["volume"] = params.get("volume")
        set_clauses.append("l.updatedAt = datetime()")

        self.run_query(
            f"""
                MATCH (i:Instructional {{instructionalId: $instructionalId}})-[:HAS_LESSON]->(l:Lesson {{lessonId: $lessonId}})
                SET {', '.join(set_clauses)}
            """,
            values
        )

    def update_lesson_orders(self, instructional_id: str, orders: List[Dict[str, Any]]) -> None:
        self.run_query(
            """
                UNWIND $orders as row
                MATCH (i:Instructional {instructionalId: $instructionalId})-[:HAS_LESSON]->(l:Lesson {lessonId: row.lessonId})
                SET l.order = row.order,
                    l.updatedAt = datetime()
            """,
            {"instructionalId": instructional_id, "orders": orders}
        )


router = APIRouter(prefix="/grapple/graph")
_SERVICE = GrappleGraphService()


async def _run_operation(message: str, operation, *args):
    try:
        data = await asyncio.to_thread(operation, *args)
        return _api_success(data)
    except Exception as error:
        return _api_failure(message, error)


@router.post("/accessible")
async def get_accessible_graph(request: AccessibleGraphRequest):
    return await _run_operation(
        "Unable to get accessible graph",
        _SERVICE.get_accessible_graph,
        request.instructionalIds,
        request.includeStructural
    )


@router.post("/element-detail")
async def get_graph_element_detail(request: ElementDetailRequest):
    return await _run_operation("Unable to get graph element detail", _SERVICE.get_graph_element_detail, request.elementId)


@router.post("/save-transcript")
async def save_transcript_to_neo4j(request: SaveTranscriptRequest):
    return await _run_operation("Unable to save transcript", _SERVICE.save_transcript_to_neo4j, request.model_dump())


@router.post("/save-segments")
async def save_segments_to_neo4j(request: SaveSegmentsRequest):
    return await _run_operation("Unable to save semantic segments", _SERVICE.save_segments_to_neo4j, request.model_dump())


@router.post("/lesson-transcripts")
async def get_lesson_transcripts_by_instructional(request: InstructionalRequest):
    return await _run_operation(
        "Unable to get lesson transcripts",
        _SERVICE.get_lesson_transcripts_by_instructional,
        request.instructionalId
    )


@router.post("/mark-lessons-ingested")
async def mark_lessons_ingested(request: MarkLessonsIngestedRequest):
    return await _run_operation("Unable to mark lessons ingested", _SERVICE.mark_lessons_ingested, request.rows)


@router.post("/mark-lessons-segmented")
async def mark_lessons_segmented(request: MarkLessonsSegmentedRequest):
    return await _run_operation("Unable to mark lessons segmented", _SERVICE.mark_lessons_segmented, request.rows)


@router.post("/save-canonical-entities")
async def save_canonical_entities(request: SaveCanonicalEntitiesRequest):
    return await _run_operation("Unable to save canonical entities", _SERVICE.save_canonical_entities, request.model_dump())


@router.post("/canonical-entities")
async def get_canonical_entities(request: InstructionalRequest):
    return await _run_operation("Unable to get canonical entities", _SERVICE.get_canonical_entities, request.instructionalId)


@router.post("/lessons")
async def get_lessons_by_instructional(request: InstructionalRequest):
    return await _run_operation("Unable to get instructional lessons", _SERVICE.get_lessons_by_instructional, request.instructionalId)


@router.post("/link-instructional-documents")
async def link_instructional_to_documents(request: InstructionalRequest):
    return await _run_operation(
        "Unable to link instructional to documents",
        _SERVICE.link_instructional_to_documents,
        request.instructionalId
    )


@router.post("/update-lesson-metadata")
async def update_lesson_metadata(request: UpdateLessonMetadataRequest):
    return await _run_operation("Unable to update lesson metadata", _SERVICE.update_lesson_metadata, request.model_dump())


@router.post("/update-lesson-orders")
async def update_lesson_orders(request: UpdateLessonOrdersRequest):
    return await _run_operation(
        "Unable to update lesson orders",
        _SERVICE.update_lesson_orders,
        request.instructionalId,
        request.orders
    )
