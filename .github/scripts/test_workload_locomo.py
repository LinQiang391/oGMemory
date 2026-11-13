import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location('workload_locomo', Path(__file__).with_name('workload_locomo.py'))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ReceiptTests(unittest.TestCase):
    def complete(self):
        return {'request_id': 'oc-request', 'request_status': 'SUCCEEDED', 'accepted': True,
                'source_status': 'ARCHIVED', 'semantic_status': 'COMPLETE', 'projection_status': 'DONE', 'ok': True, 'terminal': True}

    def test_complete_contract(self):
        self.assertTrue(module.receipt_complete(self.complete()))
        for key, value in [('request_status', 'ACCEPTED'), ('semantic_status', 'PENDING'),
                           ('projection_status', 'PENDING'), ('source_status', 'BUFFERED'), ('accepted', 1)]:
            item = self.complete()
            item[key] = value
            self.assertFalse(module.receipt_complete(item))

    def test_qa_receipt_coverage_rejects_missing_duplicate_and_partial(self):
        proofs = [{'phase': 'qa', 'session_id': 's', 'receipts': [self.complete()]}]
        self.assertEqual(module.qa_receipt_coverage(proofs, {'s'}), 1)
        for values, expected in [(proofs, {'s', 'missing'}), (proofs + proofs, {'s'}),
                                 ([{'phase': 'qa', 'session_id': 's', 'receipts': []}], {'s'})]:
            with self.assertRaises(RuntimeError):
                module.qa_receipt_coverage(values, expected)

    def test_exact_receipt_polling(self):
        with module.ReceiptProxy('http://127.0.0.1:1') as proxy:
            proxy.writes['session'] = {'oc-request': {'identity': {'accountId': 'a', 'userId': 'u', 'agentId': 'main'}}}
            response = Mock()
            response.json.return_value = self.complete()
            with patch.object(module.requests, 'get', return_value=response) as get:
                self.assertEqual(len(proxy.wait('session', 1, 0)), 1)
                self.assertEqual(get.call_args.kwargs['params']['clientRequestId'], 'oc-request')

    def test_wrong_request_rejected(self):
        with module.ReceiptProxy('http://127.0.0.1:1') as proxy:
            proxy.writes['session'] = {'other': {'identity': {}}}
            response = Mock()
            response.json.return_value = self.complete()
            with patch.object(module.requests, 'get', return_value=response), self.assertRaises(RuntimeError):
                proxy.wait('session', 1, 0)

    def test_proxy_preserves_params_and_observes_id(self):
        response = Mock(status_code=200, headers={'Content-Type': 'application/json'})
        receipt = self.complete()
        response.json.return_value = receipt
        response.content = json.dumps(receipt).encode()
        body = {'sessionId': 'session', 'clientRequestId': 'oc-request',
                'accountId': 'a', 'userId': 'u', 'agentId': 'main', 'messages': [{'role': 'user', 'content': 'test'}]}
        import urllib.request
        with module.ReceiptProxy('http://127.0.0.1:1') as proxy, patch.object(module.requests, 'post', return_value=response) as post:
            request = urllib.request.Request(proxy.url + '/api/v1/after_turn', data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=3) as result:
                self.assertEqual(json.load(result), receipt)
            self.assertEqual(json.loads(post.call_args.kwargs['data']), body)
            self.assertIn('oc-request', proxy.writes['session'])


if __name__ == '__main__':
    sys.exit(not unittest.main(exit=False).result.wasSuccessful())
