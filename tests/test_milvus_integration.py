# -*- coding: utf-8 -*-
"""连接真实 Milvus 的核心功能测试。

运行前先启动服务：docker compose up -d etcd minio milvus
可通过 MILVUS_TEST_URI 指定独立测试实例；不会使用或删除业务 collection。
"""
import asyncio
import os
import tempfile
import unittest
import uuid
from typing import List
from unittest.mock import patch

from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

import core.knowledge as knowledge_module
from core.milvus_store import create_milvus_store, get_physical_collection_name
from infra.config import settings


class DeterministicEmbeddings(Embeddings):
    """8 维确定性向量，避免功能测试消耗真实 Embedding API。"""

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * 8
        if "营业收入" in text:
            vector[0] = 1.0
        elif "净利润" in text:
            vector[1] = 1.0
        elif "现金流" in text:
            vector[2] = 1.0
        else:
            vector[7] = 0.1
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class MilvusIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_uri = settings.MILVUS_URI
        settings.MILVUS_URI = os.getenv("MILVUS_TEST_URI", settings.MILVUS_URI)
        try:
            client = MilvusClient(uri=settings.MILVUS_URI, timeout=5)
            client.list_collections()
        except Exception as exc:
            raise unittest.SkipTest(
                f"无法连接 Milvus {settings.MILVUS_URI}: {exc}；"
                "请先执行 docker compose up -d etcd minio milvus"
            )

    @classmethod
    def tearDownClass(cls):
        settings.MILVUS_URI = cls.original_uri
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.close()
        except RuntimeError:
            pass

    def setUp(self):
        self.embeddings = DeterministicEmbeddings()
        self.logical_name = f"test_{uuid.uuid4().hex}"
        self.physical_name = get_physical_collection_name(self.logical_name)
        self.store = create_milvus_store(self.logical_name, self.embeddings)
        self.docs = [
            Document(
                page_content="2024年营业收入为8000亿元",
                metadata={"source": "2024年报.pdf", "report_period": "2024年报", "page": 10},
            ),
            Document(
                page_content="2023年营业收入为6000亿元",
                metadata={"source": "2023年报.pdf", "report_period": "2023年报", "page": 11},
            ),
            Document(
                page_content="2024年净利润为400亿元",
                metadata={"source": "2024年报.pdf", "report_period": "2024年报", "page": 20},
            ),
            Document(
                page_content="2024年现金流为500亿元",
                metadata={"source": "2024年报.pdf", "report_period": "2024年报", "page": 30},
            ),
        ]

    def tearDown(self):
        self.store.client.drop_collection(self.physical_name)
        self.store.client.close()

    def _add_documents(self):
        return self.store.add_documents(
            self.docs,
            ids=[f"doc_{i}" for i in range(len(self.docs))],
        )

    def test_write_and_vector_search(self):
        inserted_ids = self._add_documents()
        self.assertEqual(len(inserted_ids), 4)

        results = self.store.similarity_search("营业收入", k=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all("营业收入" in doc.page_content for doc in results))
        self.assertEqual(
            {doc.metadata["report_period"] for doc in results},
            {"2024年报", "2023年报"},
        )
        self.assertIn("source", results[0].metadata)

    def test_conditional_search_and_official_client_query(self):
        self._add_documents()
        results = self.store.similarity_search(
            "营业收入", k=4, expr='report_period == "2024年报"'
        )
        self.assertEqual(len(results), 3)
        self.assertTrue(
            all(doc.metadata["report_period"] == "2024年报" for doc in results)
        )

        rows = self.store.client.query(
            collection_name=self.physical_name,
            filter='report_period == "2024年报"',
            output_fields=["text", "report_period", "source"],
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["report_period"] == "2024年报" for row in rows))

    def test_repeat_write_same_ids_is_idempotent(self):
        self._add_documents()
        self._add_documents()
        rows = self.store.client.query(
            collection_name=self.physical_name,
            filter="",
            limit=100,
            output_fields=["pk"],
        )
        self.assertEqual(len(rows), 4)

    def test_delete_by_ids(self):
        self._add_documents()
        self.store.delete(ids=["doc_2"])
        results = self.store.similarity_search("净利润", k=4)
        self.assertTrue(all("净利润" not in doc.page_content for doc in results))

    def test_new_instance_reads_persisted_data(self):
        self._add_documents()
        another_store = create_milvus_store(self.logical_name, self.embeddings)
        results = another_store.similarity_search("现金流", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("500亿元", results[0].page_content)

    def test_hnsw_cosine_index_created_on_server(self):
        self._add_documents()
        index_names = self.store.client.list_indexes(self.physical_name)
        self.assertTrue(index_names)
        description = self.store.client.describe_index(self.physical_name, index_names[0])
        self.assertEqual(description["index_type"], "HNSW")
        self.assertEqual(description["metric_type"], "COSINE")

    def test_incremental_index_skip_and_cleanup(self):
        docs = self.docs[:2]
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "record_manager.db").replace("\\", "/")
            manager = SQLRecordManager(
                f"test/milvus/{self.logical_name}", db_url=f"sqlite:///{db_path}"
            )
            manager.create_schema()

            first = index(
                docs,
                manager,
                self.store,
                cleanup="full",
                source_id_key="source",
                key_encoder="sha256",
            )
            self.assertEqual(first["num_added"], 2)
            self.assertEqual(first["num_skipped"], 0)

            second = index(
                docs,
                manager,
                self.store,
                cleanup="full",
                source_id_key="source",
                key_encoder="sha256",
            )
            self.assertEqual(second["num_added"], 0)
            self.assertEqual(second["num_skipped"], 2)

            third = index(
                docs[:1],
                manager,
                self.store,
                cleanup="full",
                source_id_key="source",
                key_encoder="sha256",
            )
            self.assertEqual(third["num_skipped"], 1)
            self.assertEqual(third["num_deleted"], 1)
            manager.engine.dispose()

        remaining = self.store.client.query(
            collection_name=self.physical_name,
            filter="",
            limit=100,
            output_fields=["text"],
        )
        self.assertEqual(len(remaining), 1)
        self.assertIn("2024年营业收入", remaining[0]["text"])

    def test_create_indexes_business_interface(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document_path = os.path.join(temp_dir, "sample.txt")
            with open(document_path, "w", encoding="utf-8") as file:
                file.write("2024年营业收入为8000亿元\n2024年净利润为400亿元")

            db_path = os.path.join(temp_dir, "records.db").replace("\\", "/")
            managers = []

            def manager_factory(namespace, db_url):
                manager = SQLRecordManager(namespace, db_url=f"sqlite:///{db_path}")
                managers.append(manager)
                return manager

            with patch.object(knowledge_module, "SQLRecordManager", side_effect=manager_factory):
                retriever = knowledge_module.create_indexes(
                    self.logical_name,
                    temp_dir,
                    self.embeddings,
                )
            for manager in managers:
                manager.engine.dispose()

        self.assertIsNotNone(retriever)
        results = retriever.invoke("营业收入")
        self.assertTrue(results)
        self.assertTrue(any("8000亿元" in doc.page_content for doc in results))

    def test_invalid_filter_raises_milvus_error(self):
        self._add_documents()
        with self.assertRaises(MilvusException):
            self.store.similarity_search("营业收入", expr="report_period ==")


if __name__ == "__main__":
    unittest.main()
