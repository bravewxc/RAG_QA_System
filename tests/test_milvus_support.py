# -*- coding: utf-8 -*-
"""Milvus 配置与重试机制的离线单元测试。"""
import os
import tempfile
import unittest
from unittest.mock import patch

from pymilvus.exceptions import MilvusException

from core.combine_client import STRUCTURED_DATA_PATH
from core.knowledge import create_indexes
from core.milvus_store import _build_connection_args, get_physical_collection_name
from infra.config import settings
from infra.models import get_ali_rerank
from infra.retry_utils import is_retryable_milvus_error, with_milvus_retry


class MilvusSupportTest(unittest.TestCase):
    def test_physical_collection_name_is_valid(self):
        self.assertEqual(get_physical_collection_name("0123abc"), "kb_0123abc")
        self.assertTrue(get_physical_collection_name("报表库").startswith("kb_"))

    def test_remote_connection_uses_token_without_leaking_empty_auth(self):
        with patch.multiple(
            settings,
            MILVUS_URI="https://example-milvus:19530",
            MILVUS_DB_NAME="finance",
            MILVUS_TOKEN="token-for-test",
            MILVUS_USER="",
            MILVUS_PASSWORD="",
            MILVUS_SECURE=True,
        ):
            args = _build_connection_args()
        self.assertEqual(args["uri"], "https://example-milvus:19530")
        self.assertEqual(args["db_name"], "finance")
        self.assertEqual(args["token"], "token-for-test")
        self.assertTrue(args["secure"])
        self.assertNotIn("user", args)
        self.assertNotIn("password", args)

    def test_local_file_connection_prepares_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            uri = os.path.join(temp_dir, "nested", "test.db")
            with patch.multiple(settings, MILVUS_URI=uri):
                args = _build_connection_args()
            self.assertTrue(os.path.isabs(args["uri"]))
            self.assertTrue(os.path.isdir(os.path.dirname(args["uri"])))
            self.assertNotIn("db_name", args)

    def test_milvus_retry_predicate(self):
        self.assertTrue(
            is_retryable_milvus_error(MilvusException(503, "service unavailable"))
        )
        self.assertTrue(
            is_retryable_milvus_error(MilvusException(1, "deadline exceeded"))
        )
        self.assertFalse(
            is_retryable_milvus_error(MilvusException(400, "invalid schema"))
        )

    def test_retry_recovers_from_transient_error(self):
        attempts = {"count": 0}

        @with_milvus_retry
        def flaky_operation():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise MilvusException(503, "service unavailable")
            return "ok"

        with patch("tenacity.nap.time.sleep", return_value=None):
            self.assertEqual(flaky_operation(), "ok")
        self.assertEqual(attempts["count"], 3)

    def test_rerank_boost_field_can_be_assigned(self):
        rerank = get_ali_rerank(top_n=4, boost_keywords_map={"审计意见": ["审计报告"]})
        self.assertEqual(rerank.boost_keywords_map, {"审计意见": ["审计报告"]})

    def test_structured_data_path_exists(self):
        self.assertTrue(os.path.isfile(STRUCTURED_DATA_PATH), STRUCTURED_DATA_PATH)

    def test_create_indexes_signature_is_unchanged(self):
        import inspect

        parameters = list(inspect.signature(create_indexes).parameters)
        self.assertEqual(parameters, ["collection_name", "dir_path", "embedding_function"])


if __name__ == "__main__":
    unittest.main()
