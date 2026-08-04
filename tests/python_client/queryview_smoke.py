#!/usr/bin/env python3

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import os
import time
from typing import List

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


SEALED_ROW_COUNT = 128
GROWING_ROW_COUNT = 32
TOTAL_ROW_COUNT = SEALED_ROW_COUNT + GROWING_ROW_COUNT
VECTOR_DIMENSION = 4
SEARCH_ID = 150
ITERATOR_LIMIT = 12
ITERATOR_BATCH_SIZE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the QueryView Plain Query/Search smoke test."
    )
    parser.add_argument("--host", default=os.getenv("MILVUS_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.getenv("MILVUS_PORT", "19530"))
    parser.add_argument("--connect-timeout", type=int, default=180)
    parser.add_argument(
        "--collection",
        default=f"queryview_smoke_{int(time.time())}_{os.getpid()}",
    )
    return parser.parse_args()


def vector_for(row_id: int) -> List[float]:
    return [
        float(row_id),
        float(row_id % 7),
        float(row_id % 11),
        float(row_id % 13),
    ]


def connect_with_retry(host: str, port: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = None

    while time.monotonic() < deadline:
        try:
            connections.connect(
                alias="default",
                host=host,
                port=port,
                timeout=5,
            )
            utility.list_collections(timeout=5)
            return
        except Exception as exc:  # noqa: BLE001 - report the final connection error.
            last_error = exc
            connections.disconnect(alias="default")
            time.sleep(2)

    raise RuntimeError(
        f"Milvus did not become ready at {host}:{port} within "
        f"{timeout_seconds}s: {last_error}"
    )


def insert_rows(collection: Collection, start: int, end: int) -> None:
    ids = list(range(start, end))
    values = [row_id * 10 for row_id in ids]
    vectors = [vector_for(row_id) for row_id in ids]
    collection.insert([ids, values, vectors])


def run_smoke(args: argparse.Namespace) -> None:
    connect_with_retry(args.host, args.port, args.connect_timeout)

    id_field = FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=False,
    )
    value_field = FieldSchema(name="value", dtype=DataType.INT64)
    vector_field = FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=VECTOR_DIMENSION,
    )
    schema = CollectionSchema(
        fields=[id_field, value_field, vector_field],
        description="QueryView streaming-reduction smoke collection",
    )

    collection = None
    try:
        print(f"Creating two-shard collection {args.collection}", flush=True)
        collection = Collection(
            name=args.collection,
            schema=schema,
            shards_num=2,
            consistency_level="Strong",
        )
        collection.create_index(
            field_name="vector",
            index_params={
                "index_type": "FLAT",
                "metric_type": "L2",
                "params": {},
            },
        )

        print(f"Inserting and sealing {SEALED_ROW_COUNT} rows", flush=True)
        insert_rows(collection, 0, SEALED_ROW_COUNT)
        collection.flush()
        collection.load()

        print(f"Inserting {GROWING_ROW_COUNT} growing rows", flush=True)
        insert_rows(collection, SEALED_ROW_COUNT, TOTAL_ROW_COUNT)

        query_results = collection.query(
            expr="id >= 0",
            output_fields=["id"],
        )
        actual_ids = {row["id"] for row in query_results}
        expected_ids = set(range(TOTAL_ROW_COUNT))
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            unexpected = sorted(actual_ids - expected_ids)
            raise AssertionError(
                "Strong Query returned an unexpected ID set: "
                f"count={len(actual_ids)}, missing={missing}, unexpected={unexpected}"
            )

        search_results = collection.search(
            data=[vector_for(SEARCH_ID)],
            anns_field="vector",
            param={"metric_type": "L2", "params": {}},
            limit=ITERATOR_LIMIT,
            output_fields=["id", "vector"],
        )
        actual_top_id = search_results[0][0].id
        if actual_top_id != SEARCH_ID:
            raise AssertionError(
                f"Search top-1 ID was {actual_top_id}, expected {SEARCH_ID}"
            )

        batch_search_ids = [hit.id for hit in search_results[0]]
        for hit in search_results[0]:
            actual_vector = hit.entity.get("vector")
            expected_vector = vector_for(hit.id)
            if actual_vector is None or len(actual_vector) != VECTOR_DIMENSION:
                raise AssertionError(
                    f"Batch Search did not requery vector for ID {hit.id}: {actual_vector}"
                )
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-5)
                for actual, expected in zip(actual_vector, expected_vector)
            ):
                raise AssertionError(
                    f"Batch Search returned an unexpected vector for ID {hit.id}: "
                    f"actual={actual_vector}, expected={expected_vector}"
                )

        search_iterator = collection.search_iterator(
            data=[vector_for(SEARCH_ID)],
            anns_field="vector",
            param={"metric_type": "L2", "params": {}},
            batch_size=ITERATOR_BATCH_SIZE,
            limit=ITERATOR_LIMIT,
            output_fields=["id", "vector"],
        )
        iterator_search_ids = []
        try:
            while True:
                page = search_iterator.next()
                if len(page) == 0:
                    break
                for hit in page:
                    iterator_search_ids.append(hit.id)
                    actual_vector = hit.entity.get("vector")
                    expected_vector = vector_for(hit.id)
                    if actual_vector is None or len(actual_vector) != VECTOR_DIMENSION:
                        raise AssertionError(
                            f"Search iterator did not requery vector for ID {hit.id}: "
                            f"{actual_vector}"
                        )
                    if any(
                        not math.isclose(
                            actual,
                            expected,
                            rel_tol=0.0,
                            abs_tol=1e-5,
                        )
                        for actual, expected in zip(actual_vector, expected_vector)
                    ):
                        raise AssertionError(
                            f"Search iterator returned an unexpected vector for ID {hit.id}: "
                            f"actual={actual_vector}, expected={expected_vector}"
                        )
        finally:
            search_iterator.close()

        if iterator_search_ids != batch_search_ids:
            raise AssertionError(
                "Search iterator returned a different ordered ID list: "
                f"iterator={iterator_search_ids}, batch={batch_search_ids}"
            )

        print(
            "QUERYVIEW_SMOKE_PASS "
            f"query_rows={len(actual_ids)} search_top1={actual_top_id} "
            f"iterator_rows={len(iterator_search_ids)}",
            flush=True,
        )
    finally:
        if collection is not None:
            try:
                collection.release()
            except Exception:  # noqa: BLE001 - continue with collection cleanup.
                pass
        try:
            if utility.has_collection(args.collection):
                utility.drop_collection(args.collection)
        finally:
            connections.disconnect(alias="default")


if __name__ == "__main__":
    run_smoke(parse_args())
